import base64
import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from transformers import HfArgumentParser

from human_plan.vila_train.args import VLAModelArguments, VLADataArguments, VLATrainingArguments
from human_plan.preprocessing.preprocessing import (
    preprocess_vla,
    preprocess_multimodal_vla,
)
from human_plan.preprocessing.prompting_format import preprocess_language_instruction
from human_plan.utils.action_tokenizer import build_action_tokenizer
from llava import conversation as conversation_lib
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_image_ndarray_v2


@dataclass
class ServerArguments:
    host: str = "0.0.0.0"
    port: int = 8000


def _numpy_deserialize(dct: Dict[str, Any]) -> Any:
    if "__numpy__" in dct:
        buf = base64.b64decode(dct["__numpy__"])
        arr = np.frombuffer(buf, dtype=np.dtype(dct["dtype"]))
        return arr.reshape(dct["shape"])
    return dct


def _decode_simple_payload(payload: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    payload = json.loads(json.dumps(payload), object_hook=_numpy_deserialize)
    image_dict = payload["image"]
    image = np.asarray(image_dict["rgb_head_stereo_left"], dtype=np.uint8)
    state = np.asarray(payload["state"]["states"], dtype=np.float32).reshape(-1)
    task_desc = payload.get("instruction", "finish the task")
    return image, state, task_desc


def _encode_numpy(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.ascontiguousarray(arr)
    return {
        "__numpy__": base64.b64encode(arr.tobytes()).decode("utf-8"),
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
    }


def _load_model_for_eval(model_args, data_args, training_args):
    model_name_or_path = model_args.model_name_or_path
    model_name = model_name_or_path.rstrip("/").split("/")[-1]
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_name_or_path,
        model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    model_args.predict_future_step = data_args.predict_future_step
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    data_args.action_tokenizer = build_action_tokenizer(
        model_args.action_tokenizer,
        tokenizer,
        model_args,
    )
    model.config.invalid_token_idx = data_args.action_tokenizer.invalid_token_idx
    model.config.input_placeholder_token_idx = data_args.action_tokenizer.input_placeholder_token_idx
    model.config.input_placeholder_start_token_idx = data_args.action_tokenizer.input_placeholder_start_token_idx
    model.config.input_placeholder_end_token_idx = data_args.action_tokenizer.input_placeholder_end_token_idx
    model.config.merge_hand = data_args.merge_hand
    model.config.invalid_token_weight = training_args.invalid_token_weight

    data_args.traj_action_output_dim = model.traj_decoder.out_dim
    data_args.proprio_size = model.config.proprio_size
    data_args.image_processor = image_processor
    return model, tokenizer, model_args, data_args, training_args


def _eval_single_sample(raw_data, model):
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    def _to_model_float(t: torch.Tensor) -> torch.Tensor:
        return t.to(device=device, dtype=model_dtype)

    def _to_device(t: torch.Tensor) -> torch.Tensor:
        return t.to(device)

    batch = {
        "input_ids": _to_device(raw_data["input_ids"].unsqueeze(0)),
        "labels": _to_device(raw_data["labels"].unsqueeze(0)),
        "attention_mask": _to_device(
            raw_data["input_ids"].unsqueeze(0).ne(model.tokenizer.pad_token_id)
        ),
        "images": _to_model_float(raw_data["image"]),
        "raw_proprio_inputs": _to_model_float(raw_data["proprio_input"]),
        "raw_proprio_inputs_2d": _to_model_float(raw_data["proprio_input_2d"]),
        "raw_proprio_inputs_3d": _to_model_float(raw_data["proprio_input_3d"]),
        "raw_proprio_inputs_rot": _to_model_float(raw_data["proprio_input_rot"]),
        "raw_proprio_inputs_handdof": _to_model_float(raw_data["proprio_input_handdof"]),
        "raw_proprio_inputs_hand_finger_tip": _to_model_float(raw_data["proprio_input_hand_finger_tip"]),
        "raw_ee_movement_masks": _to_model_float(raw_data["ee_movement_mask"]),
        "raw_action_labels": _to_model_float(raw_data["raw_action_label"].unsqueeze(0)),
        "raw_action_masks": _to_model_float(raw_data["raw_action_mask"].unsqueeze(0)),
    }
    with torch.inference_mode():
        outputs = model(**batch, inference=True)
    pred = outputs.prediction.detach().float().cpu().numpy()
    return pred.reshape(-1, pred.shape[-1])


def build_raw_data_dict(
    image: np.ndarray,
    state: np.ndarray,
    task_desc: str,
    data_args: VLADataArguments,
    model_args: VLAModelArguments,
    tokenizer,
) -> Dict[str, torch.Tensor]:
    valid_his_len = 0
    language_instruction = preprocess_language_instruction(
        task_desc, valid_his_len, data_args
    )
    language_instruction = preprocess_multimodal_vla(
        language_instruction, data_args
    )

    action_dim = data_args.traj_action_output_dim
    raw_label = torch.zeros((data_args.predict_future_step, action_dim), dtype=torch.float32)
    raw_mask = torch.ones((data_args.predict_future_step, action_dim), dtype=torch.bool)

    proprio_input = torch.tensor(state, dtype=torch.float32).reshape(1, -1)

    data_dict = preprocess_vla(
        language_instruction,
        proprio_input,
        raw_label,
        raw_mask,
        data_args.action_tokenizer,
        tokenizer,
        mask_input=getattr(data_args, "mask_input", False),
        mask_ignore=getattr(data_args, "mask_ignore", False),
        raw_action_label=getattr(data_args, "raw_action_label", False),
        traj_action_output_dim=getattr(data_args, "traj_action_output_dim", getattr(model_args, "traj_action_output_dim", None)),
        input_placeholder_diff_index=getattr(data_args, "input_placeholder_diff_index", False),
        sep_query_token=getattr(model_args, "sep_query_token", False),
        language_response=None,
        include_response=getattr(data_args, "include_response", False),
        include_repeat_instruction=getattr(data_args, "include_repeat_instruction", False),
        raw_language_label=task_desc,
    )

    image_tensor = process_image_ndarray_v2(
        image, data_args, reverse_channel_order=False
    )
    data_dict["image"] = torch.stack([image_tensor], dim=0)
    data_dict["raw_image_obs"] = image
    data_dict["proprio_input_2d"] = torch.zeros((1, 4), dtype=torch.float32)
    data_dict["proprio_input_3d"] = torch.zeros((1, 6), dtype=torch.float32)
    data_dict["proprio_input_rot"] = torch.zeros((1, 6), dtype=torch.float32)
    data_dict["proprio_input_handdof"] = torch.zeros((1, 30), dtype=torch.float32)
    data_dict["proprio_input_hand_finger_tip"] = torch.zeros((1, 30), dtype=torch.float32)
    data_dict["ee_movement_mask"] = torch.ones((1, 2), dtype=torch.float32)
    data_dict["raw_width"] = image.shape[1]
    data_dict["raw_height"] = image.shape[0]
    data_dict["language_label"] = task_desc
    return data_dict


def decode_image_b64(image_b64: str) -> np.ndarray:
    image_bytes = base64.b64decode(image_b64)
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.array(img)


def make_handler(model, tokenizer, model_args, data_args):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Dict, status=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send_json({"status": "ok"})
                return
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self):
            if self.path not in {"/predict", "/act"}:
                self._send_json({"error": "not found"}, status=404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                self._send_json({"error": "empty body"}, status=400)
                return
            body = self.rfile.read(length)
            try:
                payload = json.loads(body)
            except Exception:
                self._send_json({"error": "invalid json"}, status=400)
                return

            try:
                if self.path == "/act":
                    image, state, task_desc = _decode_simple_payload(payload)
                else:
                    image = decode_image_b64(payload["image_b64"])
                    state = np.asarray(payload["state"], dtype=np.float32)
                    task_desc = payload.get("task_desc", "finish the task")
            except Exception as exc:
                self._send_json({"error": f"bad payload: {exc}"}, status=400)
                return

            raw_data = build_raw_data_dict(
                image=image,
                state=state,
                task_desc=task_desc,
                data_args=data_args,
                model_args=model_args,
                tokenizer=tokenizer,
            )
            pred = _eval_single_sample(raw_data, model)
            if self.path == "/act":
                self._send_json(
                    {
                        "action": _encode_numpy(pred.astype(np.float32)),
                        "err": 0.0,
                        "traj_image": _encode_numpy(np.zeros((1, 1, 3), dtype=np.uint8)),
                    }
                )
            else:
                self._send_json({"action": pred.tolist()})

        def log_message(self, format, *args):  # quiet
            return

    return Handler


def main():
    parser = HfArgumentParser((VLAModelArguments, VLADataArguments, VLATrainingArguments, ServerArguments))
    model_args, data_args, training_args, server_args = parser.parse_args_into_dataclasses()

    model, tokenizer, model_args, data_args, training_args = _load_model_for_eval(
        model_args, data_args, training_args
    )
    model.eval()

    server = HTTPServer((server_args.host, server_args.port), make_handler(model, tokenizer, model_args, data_args))
    print(f"LeRobot policy server listening on {server_args.host}:{server_args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
