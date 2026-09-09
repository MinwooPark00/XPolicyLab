#!/usr/bin/env python3
import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.encoder.dinosiglip_vit import DinoSigLIPViTBackbone
from models.hrdt_runner import HRDTRunner


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    if checkpoint_path.suffix == ".safetensors":
        import safetensors.torch

        missing, unexpected = safetensors.torch.load_model(model, checkpoint_path, strict=False)
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
        return

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = None
    for key in ("state_dict", "model", "module"):
        if isinstance(ckpt, dict) and key in ckpt:
            state = ckpt[key]
            break
    if state is None:
        state = ckpt

    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")


def decode_image_b64(image_b64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_b64)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def numpy_deserialize(dct: Dict[str, Any]) -> Any:
    if "__numpy__" in dct:
        np_obj = np.frombuffer(base64.b64decode(dct["__numpy__"]), np.lib.format.descr_to_dtype(dct["dtype"]))
        shape = dct["shape"]
        return np_obj.reshape(shape) if shape else np_obj[0]
    return dct


def numpy_serialize(arr: np.ndarray) -> Dict[str, Any]:
    return {
        "__numpy__": base64.b64encode(arr.data if arr.flags["C_CONTIGUOUS"] else arr.tobytes()).decode(),
        "dtype": np.lib.format.dtype_to_descr(arr.dtype),
        "shape": arr.shape,
    }


def convert_numpy_in_dict(data: Any, func) -> Any:
    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {key: convert_numpy_in_dict(value, func) for key, value in data.items()}
    if isinstance(data, list):
        return [convert_numpy_in_dict(item, func) for item in data]
    if isinstance(data, np.ndarray):
        return func(data)
    return data


def _predict(model, vision_encoder, image_transform, device, weight_dtype, images_np, state, lang_embeds=None):
    all_pixel_values = [image_transform(Image.fromarray(image)) for image in images_np]
    example = all_pixel_values[0]
    images = {
        key: torch.stack([pv[key] for pv in all_pixel_values], dim=0).unsqueeze(0).to(
            device=device, dtype=weight_dtype
        )
        for key in example
    }
    image_key = next(iter(images))
    batch_size, _, channels, height, width = images[image_key].shape
    for key in images:
        images[key] = images[key].view(-1, channels, height, width)
    image_features = vision_encoder(images).detach()
    image_features = image_features.view((batch_size, -1, vision_encoder.embed_dim))

    state_tokens = torch.as_tensor(state, dtype=torch.float32)
    if state_tokens.ndim == 1:
        state_tokens = state_tokens.unsqueeze(0)
    if state_tokens.ndim == 2:
        state_tokens = state_tokens.unsqueeze(0)
    state_tokens = state_tokens.to(device=device, dtype=weight_dtype)

    lang_tokens: Optional[torch.Tensor] = None
    lang_attn_mask: Optional[torch.Tensor] = None
    if lang_embeds is not None:
        lang_tokens = torch.as_tensor(lang_embeds, dtype=torch.float32)
        if lang_tokens.ndim == 2:
            lang_tokens = lang_tokens.unsqueeze(0)
        lang_tokens = lang_tokens.to(device=device, dtype=weight_dtype)
        lang_attn_mask = torch.ones((lang_tokens.shape[0], lang_tokens.shape[1]), device=device, dtype=torch.bool)

    with torch.no_grad():
        pred_actions = model.predict_action(
            state_tokens=state_tokens,
            image_tokens=image_features,
            lang_tokens=lang_tokens,
            lang_attn_mask=lang_attn_mask,
        ).to(dtype=torch.float32)
    return pred_actions.squeeze(0).cpu().numpy()


def _pad_state(state: np.ndarray, target_dim: int) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.ndim == 2 and state.shape[0] == 1:
        state = state[0]
    if state.shape[-1] >= target_dim:
        return state[..., :target_dim]
    return np.pad(state, (0, target_dim - state.shape[-1]), mode="constant")


def _pick_simple_images(image_dict: Dict[str, Any]) -> list[np.ndarray]:
    preferred_keys = [
        "rgb_head_stereo_left",
        "front_stereo_left",
        "observation.images.egocentric",
        "rs_view",
    ]
    for key in preferred_keys:
        if key in image_dict:
            img = np.asarray(image_dict[key], dtype=np.uint8)
            return [img]
    if not image_dict:
        raise KeyError("request.image is empty")
    return [np.asarray(next(iter(image_dict.values())), dtype=np.uint8)]


def build_app(model, vision_encoder, image_transform, device, weight_dtype, state_dim: int, action_exec_horizon: int):
    app = FastAPI()

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse(content={"status": "ok"})

    @app.post("/predict")
    def predict(payload: Dict[str, Any]) -> JSONResponse:
        try:
            images_b64 = payload["images_b64"]
            state = np.asarray(payload["state"], dtype=np.float32)
            lang_embeds = payload.get("lang_embeds")
            images_np = [np.asarray(decode_image_b64(image_b64), dtype=np.uint8) for image_b64 in images_b64]
            pred = _predict(model, vision_encoder, image_transform, device, weight_dtype, images_np, state, lang_embeds)
            return JSONResponse(content={"action": pred.tolist()})
        except Exception as exc:
            return JSONResponse(content={"error": f"inference failed: {exc}"}, status_code=500)

    @app.post("/act")
    def act(payload: Dict[str, Any]) -> JSONResponse:
        try:
            request = convert_numpy_in_dict(payload, numpy_deserialize)
            images_np = _pick_simple_images(request["image"])
            state = _pad_state(np.asarray(request["state"]["states"], dtype=np.float32), state_dim)
            pred = _predict(model, vision_encoder, image_transform, device, weight_dtype, images_np, state)
            if action_exec_horizon > 0:
                pred = pred[:action_exec_horizon]
            response = convert_numpy_in_dict(
                {
                    "action": pred.astype(np.float32),
                    "err": 0.0,
                    "traj_image": np.zeros((1, 1, 3), dtype=np.uint8),
                },
                numpy_serialize,
            )
            return JSONResponse(content=response)
        except Exception as exc:
            return JSONResponse(content={"status": str(exc)}, status_code=500)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--config_path", default="configs/hrdt_finetune_lerobot.yaml")
    parser.add_argument("--vision_encoder", default="dino-siglip")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--action_exec_horizon", type=int, default=16)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    weight_dtype = torch.bfloat16

    print(f"[hrdt_serve] loading config from {args.config_path}", flush=True)
    print(
        f"[hrdt_serve] startup args checkpoint={args.checkpoint_path} vision_encoder={args.vision_encoder} "
        f"device={args.device} host={args.host} port={args.port}",
        flush=True,
    )
    print("[hrdt_serve] constructing vision encoder", flush=True)
    vision_encoder = DinoSigLIPViTBackbone(
        vision_backbone_id=args.vision_encoder,
        image_resize_strategy="letterbox" if config["dataset"]["image_aspect_ratio"] == "pad" else "resize-naive",
        default_image_size=384,
    )
    print("[hrdt_serve] vision encoder constructed", flush=True)
    image_transform = vision_encoder.get_image_transform()
    print("[hrdt_serve] moving vision encoder to target device", flush=True)
    vision_encoder.eval().to(device=device, dtype=weight_dtype)
    print("[hrdt_serve] vision encoder ready", flush=True)

    print("[hrdt_serve] constructing HRDT runner", flush=True)
    model = HRDTRunner(
        state_dim=config["common"]["state_dim"],
        action_dim=config["common"]["action_dim"],
        pred_horizon=config["common"]["action_chunk_size"],
        config=config["model"],
        act_pos_emb_config=[("state", 1), ("action", config["common"]["action_chunk_size"])],
        img_pos_emb_config=[
            ("image", (config["common"]["img_history_size"], config["common"]["num_cameras"], -vision_encoder.num_patches)),
        ],
        lang_pos_emb_config=[("language", -config["dataset"]["tokenizer_max_length"])],
        max_img_len=config["common"]["img_history_size"] * config["common"]["num_cameras"] * vision_encoder.num_patches,
        max_lang_len=config["dataset"]["tokenizer_max_length"],
        training_mode="lang",
        mode="finetune",
        dtype=weight_dtype,
    )
    print("[hrdt_serve] HRDT runner constructed", flush=True)
    print(f"[hrdt_serve] loading checkpoint from {args.checkpoint_path}", flush=True)
    load_checkpoint(model, Path(args.checkpoint_path))
    print("[hrdt_serve] checkpoint loaded", flush=True)
    print("[hrdt_serve] moving model to target device", flush=True)
    model.eval().to(device=device, dtype=weight_dtype)
    print("[hrdt_serve] model ready", flush=True)

    print("[hrdt_serve] building FastAPI app", flush=True)
    app = build_app(
        model,
        vision_encoder,
        image_transform,
        device,
        weight_dtype,
        int(config["common"]["state_dim"]),
        args.action_exec_horizon,
    )
    print("[hrdt_serve] app ready", flush=True)
    print(f"H-RDT policy server listening on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
