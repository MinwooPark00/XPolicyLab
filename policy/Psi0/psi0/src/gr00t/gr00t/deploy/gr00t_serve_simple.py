import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import tyro
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

from psi.deploy.helpers import RequestMessage, ResponseMessage


@dataclass
class ServerConfig:
    """Configuration for running a simple HTTP Gr00t inference server."""

    model_path: str
    """Path to the model checkpoint directory"""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag"""

    device: str = "cuda"
    """Device to run the model on"""

    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = 5555
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = True
    """Whether to use the sim policy wrapper"""

    action_exec_horizon: int | None = None
    """Optional action horizon to truncate the returned action chunk"""


class Server:
    def __init__(self, cfg: ServerConfig):
        if cfg.model_path.startswith("/") and not os.path.exists(cfg.model_path):
            raise FileNotFoundError(f"Model path {cfg.model_path} does not exist")

        self.policy = Gr00tPolicy(
            embodiment_tag=cfg.embodiment_tag,
            model_path=cfg.model_path,
            device=cfg.device,
            strict=cfg.strict,
        )

        if cfg.use_sim_policy_wrapper:
            self.policy = Gr00tSimPolicyWrapper(self.policy, strict=cfg.strict)

        self.modality_configs = self.policy.get_modality_config()
        self.action_exec_horizon = cfg.action_exec_horizon
        self.last_serve_time = time.monotonic()

    @staticmethod
    def _ensure_btd(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:  # (D,)
            return arr[None, None, ...]
        if arr.ndim == 2:  # (T, D)
            return arr[None, ...]
        if arr.ndim == 3:  # (B, T, D)
            return arr
        raise ValueError(f"Array must be 1-3D, got shape {arr.shape}")

    @staticmethod
    def _to_batched_video(value: Any) -> np.ndarray:
        arr = np.asarray(value)
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)

        if arr.ndim == 3:  # (H, W, C)
            arr = arr[None, None, ...]
        elif arr.ndim == 4:  # (T, H, W, C)
            arr = arr[None, ...]
        elif arr.ndim != 5:
            raise ValueError(f"Video array must be 3-5D, got shape {arr.shape}")
        return arr

    @staticmethod
    def _to_batched_state(value: Any) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        return Server._ensure_btd(arr)

    @staticmethod
    def _split_state_from_psi(
        proprio_joint_positions: np.ndarray,
        amo_policy_command: np.ndarray,
    ) -> dict[str, np.ndarray]:
        proprio = Server._ensure_btd(np.asarray(proprio_joint_positions, dtype=np.float32))
        command = Server._ensure_btd(np.asarray(amo_policy_command, dtype=np.float32))

        left_hand = np.concatenate(
            [
                proprio[..., 29:32],
                proprio[..., 34:36],
                proprio[..., 32:34],
            ],
            axis=-1,
        )
        right_hand = proprio[..., 36:43]
        left_arm = proprio[..., 15:22]
        right_arm = proprio[..., 22:29]
        rpy = np.concatenate([proprio[..., 13:15], proprio[..., 12:13]], axis=-1)
        height = command[..., 6:7]
        base_height_cmd = command[..., 6:7]

        return {
            "left_hand": left_hand,
            "right_hand": right_hand,
            "left_arm": left_arm,
            "right_arm": right_arm,
            "rpy": rpy,
            "height": height,
            "_base_height_cmd": base_height_cmd,
        }

    @staticmethod
    def _action_to_psi_format(action: dict[str, Any]) -> np.ndarray:
        def _pick(key: str) -> np.ndarray:
            raw = action.get(key) or action.get(f"action.{key}")
            if raw is None:
                raise KeyError(f"Missing action key '{key}'")
            return Server._ensure_btd(np.asarray(raw, dtype=np.float32))

        left_hand = _pick("left_hand")
        right_hand = _pick("right_hand")
        left_arm = _pick("left_arm")
        right_arm = _pick("right_arm")
        rpy = _pick("rpy")
        height = _pick("height")
        torso_vx = _pick("torso_vx")
        torso_vy = _pick("torso_vy")
        torso_vyaw = _pick("torso_vyaw")
        target_yaw = _pick("target_yaw")

        psi_action = np.concatenate(
            [
                left_hand,
                right_hand,
                left_arm,
                right_arm,
                rpy,
                height,
                torso_vx,
                torso_vy,
                torso_vyaw,
                target_yaw,
            ],
            axis=-1,
        )
        return psi_action

    def _build_observation(self, request: RequestMessage) -> dict[str, Any]:
        image_dict = request.image
        state_dict = request.state
        instruction = request.instruction

        observation: dict[str, Any] = {}

        # g1_locomanip_config expects rs_view
        video_key = self.modality_configs["video"].modality_keys[0]
        if video_key in image_dict:
            raw_video = image_dict[video_key]
        elif "rgb_head_stereo_left" in image_dict:
            raw_video = image_dict["rgb_head_stereo_left"]
        elif len(image_dict) == 1:
            raw_video = next(iter(image_dict.values()))
        else:
            raise KeyError(f"Missing video key '{video_key}' in request.image")
        observation[f"video.{video_key}"] = self._to_batched_video(raw_video)

        # State modalities: accept either psi-style input or SIMPLE "states"
        if "proprio_joint_positions" in state_dict and "amo_policy_command" in state_dict:
            state_parts = self._split_state_from_psi(
                state_dict["proprio_joint_positions"],
                state_dict["amo_policy_command"],
            )
        elif "states" in state_dict:
            # SIMPLE psi0 baseline sends 32-dim states:
            # [left_hand_thumb(3), left_hand_middle(2), left_hand_index(2),
            #  right_hand(7), left_arm(7), right_arm(7), rpyh(4)]
            states = self._ensure_btd(np.asarray(state_dict["states"], dtype=np.float32))
            if states.shape[-1] != 32:
                raise ValueError(f"Expected 'states' with 32 dims, got {states.shape[-1]}")

            left_hand = np.concatenate(
                [states[..., 0:3], states[..., 3:5], states[..., 5:7]],
                axis=-1,
            )
            right_hand = states[..., 7:14]
            left_arm = states[..., 14:21]
            right_arm = states[..., 21:28]
            rpy = states[..., 28:31]
            height = states[..., 31:32]

            state_parts = {
                "left_hand": left_hand,
                "right_hand": right_hand,
                "left_arm": left_arm,
                "right_arm": right_arm,
                "rpy": rpy,
                "height": height,
                "_base_height_cmd": height,
            }
        else:
            raise KeyError(
                "Missing psi-style state ('proprio_joint_positions' + 'amo_policy_command') "
                "or SIMPLE 'states' in request.state"
            )

        for state_key in self.modality_configs["state"].modality_keys:
            if state_key not in state_parts:
                raise KeyError(f"Missing state key '{state_key}' in psi mapping")
            observation[f"state.{state_key}"] = state_parts[state_key]

        language_key = self.modality_configs["language"].modality_keys[0]
        observation[language_key] = (instruction,)

        return observation

    def predict_action(self, payload: dict[str, Any]) -> JSONResponse:
        try:
            request = RequestMessage.deserialize(payload)
            observation = self._build_observation(request)
            action, _info = self.policy.get_action(observation)

            psi_action = self._action_to_psi_format(action)
            if self.action_exec_horizon is not None and psi_action.ndim >= 2:
                psi_action = psi_action[:, : self.action_exec_horizon]
            # SIMPLE client expects (T, D), not (B, T, D)
            if psi_action.ndim == 3 and psi_action.shape[0] == 1:
                psi_action = psi_action[0]
            # if we got a single step (D,), add time dimension.
            if psi_action.ndim == 1:
                psi_action = psi_action[None, :]

            self.last_serve_time = time.monotonic()
            response = ResponseMessage(psi_action, 0.0)
            return JSONResponse(content=response.serialize())
        except Exception as exc:
            return JSONResponse(content={"status": str(exc)})

    def run(self, host: str = "0.0.0.0", port: int = 5555) -> None:
        app = FastAPI()
        app.post("/act")(self.predict_action)
        app.get("/health")(lambda: JSONResponse(content={"status": "ok"}))
        print(f"Gr00t HTTP server listening on {host}:{port}")
        try:
            uvicorn.run(app, host=host, port=port)
        except Exception as exc:
            print(f"Server crashed: {exc}")
        finally:
            print("Server stopped.")
            sys.exit(1)


def serve(cfg: ServerConfig) -> None:
    server = Server(cfg)
    server.run(cfg.host, cfg.port)


def main() -> None:
    print(f"Args: {sys.argv}")
    config = tyro.cli(ServerConfig, args=sys.argv[1:])
    serve(config)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    config = tyro.cli(ServerConfig)
    serve(config)
