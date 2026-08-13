"""XPolicyLab adapter for standalone MHBench GauDP."""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_POLICY_DIR = Path(__file__).resolve().parent
_XPL_ROOT = Path(__file__).resolve().parents[2]
if str(_XPL_ROOT) not in sys.path:
    sys.path.insert(0, str(_XPL_ROOT))

from XPolicyLab.model_template import ModelTemplate  # noqa: E402
from XPolicyLab.policy.GauDP.gaudp.gaussian import (  # noqa: E402
    freeze_gaussian_encoder,
    load_gaussian_checkpoint,
)
from XPolicyLab.policy.GauDP.gaudp.policy import GauDPPolicy  # noqa: E402
from XPolicyLab.policy.GauDP.gaudp.runner import GauDPRunner  # noqa: E402
from XPolicyLab.policy.GauDP.gaudp.schema import (  # noqa: E402
    ROBOT_NAMES,
    pack_xpolicy_action,
    proprio_from_observation,
)
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root  # noqa: E402
from XPolicyLab.utils.process_data import get_batch_size, get_robot_action_dim_info  # noqa: E402

IMAGE_SIZE = (240, 320)


def _to_chw_float(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"camera color must be HWC RGB, got {image.shape}")
    tensor = torch.as_tensor(image).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    tensor = F.interpolate(tensor, size=IMAGE_SIZE, mode="bilinear", align_corners=False)
    return tensor[0].numpy()


def encode_observation(observation: dict, use_scene: bool) -> tuple[np.ndarray, np.ndarray]:
    vision = observation["vision"]
    camera_keys = ["cam_left_wrist", "cam_right_wrist"]
    if use_scene:
        camera_keys.append("cam_head")
    images = np.stack([_to_chw_float(vision[key]["color"]) for key in camera_keys])
    state = np.concatenate([proprio_from_observation(observation, robot) for robot in ROBOT_NAMES])
    return images.astype(np.float32), state.astype(np.float32)


def _checkpoint_file(root: Path, stage: str, preference: str) -> Path:
    if root.is_file():
        if stage == "policy":
            return root
        root = root.parent.parent
    preferred = root / stage / f"{preference}.ckpt"
    fallback = root / stage / ("last.ckpt" if preference == "best" else "best.ckpt")
    if preferred.is_file():
        return preferred
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"no {stage} checkpoint found at {preferred} or {fallback}")


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        if model_cfg.get("action_type") != "ee":
            raise NotImplementedError("GauDP supports action_type=ee only")
        self.model_cfg = model_cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_scene = bool(model_cfg.get("use_scene", False))
        if bool(int(os.environ.get("GAUDP_USE_SCENE", "0"))):
            self.use_scene = True

        info = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        if len(info["arm_dim"]) != 2:
            raise ValueError("MHBench GauDP requires a dual-arm XPolicyLab robot config")
        self.ee_dim = info["ee_dim"]
        self.batch_size = get_batch_size(model_cfg["env_cfg_type"])

        root = resolve_checkpoint_root(
            model_cfg,
            _POLICY_DIR / "checkpoints",
            policy_dir=_POLICY_DIR,
            must_exist=True,
        )
        preference = str(model_cfg.get("checkpoint", "best"))
        policy_path = _checkpoint_file(root, "policy", preference)
        gaussian_path = _checkpoint_file(root, "gaussian", preference)
        payload = torch.load(policy_path, map_location="cpu", weights_only=False)
        if payload.get("format") != "mhbench-gaudp-policy-v1":
            raise ValueError(f"unsupported GauDP policy checkpoint: {policy_path}")
        config = dict(payload["config"])
        expected_views = 3 if self.use_scene else 2
        if int(config["num_views"]) != expected_views:
            raise ValueError(
                f"checkpoint uses {config['num_views']} views, but evaluation requested {expected_views}; "
                "GAUDP_USE_SCENE and data conversion/training must match"
            )
        self.model = GauDPPolicy(**config)
        load_gaussian_checkpoint(self.model.gaussian_encoder, gaussian_path, strict=True)
        missing, unexpected = self.model.load_state_dict(payload["state_dict"], strict=False)
        non_gaussian_missing = [key for key in missing if not key.startswith("gaussian_encoder.")]
        if non_gaussian_missing or unexpected:
            raise RuntimeError(
                f"policy state mismatch: missing={non_gaussian_missing}, unexpected={unexpected}"
            )
        freeze_gaussian_encoder(self.model.gaussian_encoder)
        self.model.to(self.device).eval()
        self.runner = GauDPRunner(self.model.n_obs_steps)
        self._env_indices: list[int] | None = None
        print(f"[GauDP] loaded {policy_path} and {gaussian_path} on {self.device}")

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        indices = []
        for observation in obs_list:
            env_idx = int(observation.get("env_idx", 0))
            images, state = encode_observation(observation, self.use_scene)
            self.runner.update(env_idx, images, state)
            indices.append(env_idx)
        self._env_indices = indices

    def get_action(self):
        if not self._env_indices:
            raise RuntimeError("get_action() called before update_obs()")
        return self.get_action_batch([self._env_indices[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        indices = self._env_indices if env_idx_list is None else list(env_idx_list)
        if not indices:
            raise RuntimeError("get_action_batch() called before update_obs_batch()")
        images, state = self.runner.batch(indices, self.device)
        action = self.model.predict_action(images, state).cpu().numpy()
        return [
            [pack_xpolicy_action(action[batch, step], self.ee_dim) for step in range(action.shape[1])]
            for batch in range(action.shape[0])
        ]

    def reset(self):
        self.runner.reset()
        self._env_indices = None
