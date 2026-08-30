"""XPolicyLab adapter for standalone MHBench GauDP.

GauDP is centralized and joint-space: one network reads both robots' 86D
URDF-ordered joint state plus the two ego views, and predicts the pair's 70D
absolute joint-target action -- the same contract the GR00T adapter drives the
environment with (`mhbench_raw_action.<robot>.{joint_targets, height,
base_vel}`, `upper_body_mode="joint"`). The wrist-pose/Pink path GauDP used to
be evaluated under is gone from this adapter; see `gaudp/schema.py`.

The two cameras stay `ego_a`, `ego_b` in that order (`GAUDP_USE_SCENE=1` adds
`scene` as a third view and needs a three-view Gaussian checkpoint and a
re-extracted feature cache).
"""

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
    ACTION_DIM,
    ACTION_SCHEMA,
    PROPRIO_DIM,
    ROBOT_NAMES,
    STATE_SCHEMA,
    pack_xpolicy_action,
    proprio_from_observation,
)
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root  # noqa: E402

POLICY_CHECKPOINT_FORMAT = "mhbench-gaudp-policy-v2"
"""The joint-space checkpoint format. `-v1` was the 42D/44D wrist-pose policy.

A v1 checkpoint is refused rather than partially loaded: its diffusion U-Net is
44 channels wide against 70 here, its observation encoder was conditioned on
21D-per-robot pelvis/EEF poses, and its normalizer buffers are fitted to that
contract. Nothing in it transfers, so a silent `strict=False` load would serve
a policy that predicts wrist poses into a joint-target action term.
"""

IMAGE_SIZE = (240, 320)

_CLIP_OFF = ("", "none", "null", "off", "false", "0")
_CLIP_ON = ("fitted", "true", "1")


def _require_finite(name: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if not np.isfinite(value).all():
        bad = int(value.size - np.isfinite(value).sum())
        raise ValueError(f"{name} contains {bad} NaN/Inf value(s)")
    return value


def _clip_enabled(value, name: str) -> bool:
    """Whether a serving option requests checkpoint-fitted range clamping."""
    if value is None:
        return False
    token = str(value).strip().lower()
    if token in _CLIP_OFF:
        return False
    if token in _CLIP_ON:
        return True
    raise ValueError(f"{name} must be 'fitted' or None, got {value!r}")


def _clip_fitted_state(state: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Clamp an online joint observation to the policy normalizer's fit range.

    Several handover joints barely move in the demonstrations. Their min-max
    ranges are as small as 9e-6 rad, so ordinary closed-loop simulator drift
    otherwise becomes a state conditioning value hundreds of times larger than
    anything used for training.
    """
    state = _require_finite("encoded proprioception", np.asarray(state, dtype=np.float32))
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    if low.shape != state.shape or high.shape != state.shape:
        raise ValueError(
            f"obs_clip bounds are {low.shape}/{high.shape}, but encoded state is {state.shape}"
        )
    return np.clip(state, low, high)


def _to_chw_float(image: np.ndarray) -> np.ndarray:
    image = _require_finite("camera color", image)
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
    return (
        _require_finite("encoded observation images", images).astype(np.float32),
        _require_finite("encoded proprioception", state).astype(np.float32),
    )


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


def _check_checkpoint_contract(payload: dict, policy_path: Path) -> None:
    """Refuse anything that is not a joint-space v2 checkpoint, by name."""
    recorded = str(payload.get("format", ""))
    if recorded == "mhbench-gaudp-policy-v1":
        raise ValueError(
            f"{policy_path} is a v1 GauDP checkpoint (42D pelvis/EEF state, 44D wrist-pose "
            "action). GauDP now trains and serves the centralized GR00T joint contract "
            f"({PROPRIO_DIM}D state / {ACTION_DIM}D absolute joint targets); no weights carry "
            "over, so this run has to be retrained in joint space."
        )
    if recorded != POLICY_CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported GauDP policy checkpoint format {recorded!r} at {policy_path}; "
            f"expected {POLICY_CHECKPOINT_FORMAT!r}"
        )
    state_dim = int(payload.get("state_dim", -1))
    action_dim = int(payload.get("action_dim", -1))
    if state_dim != PROPRIO_DIM or action_dim != ACTION_DIM:
        raise ValueError(
            f"{policy_path} was trained on {state_dim}D state / {action_dim}D action; "
            f"this adapter serves {PROPRIO_DIM}D / {ACTION_DIM}D"
        )
    state_schema = tuple(payload.get("state_schema", ()))
    action_schema = tuple(payload.get("action_schema", ()))
    if state_schema != STATE_SCHEMA or action_schema != ACTION_SCHEMA:
        raise ValueError(
            f"{policy_path} records state={state_schema} action={action_schema}, which is not "
            f"the centralized GR00T ordering state={STATE_SCHEMA} action={ACTION_SCHEMA}"
        )


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        if model_cfg.get("action_type") != "joint":
            raise NotImplementedError(
                "GauDP is a joint-space policy: it predicts the centralized 70D absolute "
                "joint-target action, so deploy.yml/serve must set action_type=joint"
            )
        self.model_cfg = model_cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_scene = bool(model_cfg.get("use_scene", False))
        if bool(int(os.environ.get("GAUDP_USE_SCENE", "0"))):
            self.use_scene = True

        # Nothing is read out of XPolicyLab's env_cfg tree: the joint contract
        # is fixed by mhbench_keys.py, and `env_cfg_type` is now
        # `unitree_g1x2_centralized` (the checkpoint's name, matching every
        # other centralized baseline), which has no `env_cfg/<type>.yml` --
        # those describe XPolicyLab's own scenes, not MHBench's Isaac ones.
        # `get_robot_action_dim_info`/`get_batch_size` used to be called here
        # and would now raise FileNotFoundError at server start.

        root = resolve_checkpoint_root(
            model_cfg,
            _POLICY_DIR / "checkpoints",
            policy_dir=_POLICY_DIR,
            must_exist=True,
        )
        preference = str(model_cfg.get("checkpoint", "best"))
        policy_path = _checkpoint_file(root, "policy", preference)
        payload = torch.load(policy_path, map_location="cpu", weights_only=False)
        _check_checkpoint_contract(payload, policy_path)
        configured_gaussian = os.environ.get("GAUDP_GAUSSIAN_CKPT") or model_cfg.get("gaussian_checkpoint")
        recorded = str(payload.get("gaussian_checkpoint", ""))
        candidates = []
        if configured_gaussian:
            candidates.append(Path(str(configured_gaussian)).expanduser())
        if recorded:
            candidates.append(Path(recorded).expanduser())
            candidates.append(root / "gaussian" / Path(recorded).name)
            candidates.append(_POLICY_DIR / "weights" / Path(recorded).name)
        for candidate in candidates:
            if candidate.is_file():
                gaussian_path = candidate.resolve()
                break
        else:
            # Old policy checkpoints recorded only a filename. Retain their
            # original run-local best/last lookup as the final fallback.
            try:
                gaussian_path = _checkpoint_file(root, "gaussian", preference)
            except FileNotFoundError as local_error:
                raise FileNotFoundError(
                    f"could not resolve the Gaussian checkpoint recorded by {policy_path}: {recorded!r}. "
                    "Set GAUDP_GAUSSIAN_CKPT or deploy.yml gaussian_checkpoint to the exact checkpoint "
                    "used for offline feature extraction."
                ) from local_error
        config = dict(payload["config"])
        expected_views = 3 if self.use_scene else 2
        if int(config["num_views"]) != expected_views:
            raise ValueError(
                f"checkpoint uses {config['num_views']} views, but evaluation requested {expected_views}; "
                "GAUDP_USE_SCENE and data conversion/training must match"
            )
        expected_cameras = ["ego_a", "ego_b"] + (["scene"] if self.use_scene else [])
        recorded_cameras = list(payload.get("camera_order", []))
        if recorded_cameras != expected_cameras:
            raise ValueError(
                f"checkpoint camera order is {recorded_cameras}, expected {expected_cameras}"
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
        self._obs_bounds: tuple[np.ndarray, np.ndarray] | None = None
        if _clip_enabled(model_cfg.get("obs_clip", "fitted"), "obs_clip"):
            low = self.model.normalizer.state_min.detach().cpu().numpy().astype(np.float32)
            high = self.model.normalizer.state_max.detach().cpu().numpy().astype(np.float32)
            gain = 2.0 / np.maximum(high - low, 1e-9)
            self._obs_bounds = (low, high)
            print(
                f"[GauDP][obs_clip] clamping joint state to the fitted range; "
                f"largest normalization gain {gain.max():.0f} per radian on dim {int(gain.argmax())}"
            )
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
            if self._obs_bounds is not None:
                state = _clip_fitted_state(state, *self._obs_bounds)
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
        _require_finite("predicted action", action)
        return [
            [pack_xpolicy_action(action[batch, step]) for step in range(action.shape[1])]
            for batch in range(action.shape[0])
        ]

    def reset(self):
        self.runner.reset()
        self._env_indices = None
