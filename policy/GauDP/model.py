"""XPolicyLab adapter for standalone MHBench GauDP.

GauDP is centralized and joint-space: one network reads every robot's 43D
URDF-ordered joint state plus their ego views, and predicts all of their 35D
absolute joint-target actions at once (86D / 70D for a two-robot task, 129D /
105D for a three-robot one) -- the same contract the GR00T adapter drives the
environment with (`mhbench_raw_action.<robot>.{joint_targets, height,
base_vel}`, `upper_body_mode="joint"`). See `gaudp/schema.py`.

The checkpoint decides both the robot count (its recorded state width) and the
views (its `camera_order`); serving reproduces what was trained. `use_scene` /
`GAUDP_USE_SCENE` only cross-check that the scene camera is among them.
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
    ACTION_SCHEMA,
    CAMERA_SLOT,
    ROBOT_ACTION_DIM,
    SCENE_VIEW,
    STATE_SCHEMA,
    joint_state_from_observation,
    pack_xpolicy_action,
    robot_count_from_state_dim,
    robot_names,
    robots_in_observation,
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


def encode_observation(
    observation: dict, camera_order: list[str], robots: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """The checkpoint's views, in its order, and the scene's robots' joint state."""
    vision = observation["vision"]
    images = []
    for camera in camera_order:
        slot = CAMERA_SLOT[camera]
        if slot not in vision:
            raise KeyError(
                f"this checkpoint was trained on view {camera!r}, which MHBench delivers as "
                f"vision[{slot!r}]; this observation carries {sorted(vision)}. Add {camera} to the "
                f"client's --obs_cameras (eval/runner.sbatch: EVAL_OBS_CAMERAS)."
            )
        images.append(_to_chw_float(vision[slot]["color"]))
    images = np.stack(images)
    state = joint_state_from_observation(observation, robots)
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


def _checkpoint_preference(model_cfg: dict) -> str:
    """Resolve the shared eval override for GauDP's best/last checkpoints."""
    preference = str(model_cfg.get("checkpoint_num") or model_cfg.get("checkpoint", "best"))
    if preference not in {"best", "last"}:
        raise ValueError(
            "GauDP checkpoint_num/checkpoint must be 'best' or 'last', "
            f"got {preference!r}"
        )
    return preference


def _check_checkpoint_contract(payload: dict, policy_path: Path) -> int:
    """Refuse anything that is not a joint-space v2 checkpoint, by name.

    Returns the robot count the checkpoint drives, read off its state width. A
    checkpoint written before `robot_count` was recorded is a two-robot one.
    """
    recorded = str(payload.get("format", ""))
    if recorded == "mhbench-gaudp-policy-v1":
        raise ValueError(
            f"{policy_path} is a v1 GauDP checkpoint (42D pelvis/EEF state, 44D wrist-pose "
            "action). GauDP now trains and serves the centralized GR00T joint contract "
            "(86D state / 70D absolute joint targets for two robots); no weights carry "
            "over, so this run has to be retrained in joint space."
        )
    if recorded != POLICY_CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported GauDP policy checkpoint format {recorded!r} at {policy_path}; "
            f"expected {POLICY_CHECKPOINT_FORMAT!r}"
        )
    state_dim = int(payload.get("state_dim", -1))
    action_dim = int(payload.get("action_dim", -1))
    try:
        count = robot_count_from_state_dim(state_dim)
    except ValueError as error:
        raise ValueError(
            f"{policy_path} was trained on {state_dim}D state / {action_dim}D action; this adapter "
            f"serves 43D / 35D per robot -- 86D / 70D for two robots, 129D / 105D for three"
        ) from error
    if action_dim != count * ROBOT_ACTION_DIM:
        raise ValueError(
            f"{policy_path} pairs {state_dim}D state ({count} robots) with {action_dim}D action; "
            f"expected {count * ROBOT_ACTION_DIM}D"
        )
    declared = int(payload.get("robot_count", count))
    if declared != count:
        raise ValueError(
            f"{policy_path} records robot_count={declared} but its {state_dim}D state is {count} robots"
        )
    state_schema = tuple(payload.get("state_schema", ()))
    action_schema = tuple(payload.get("action_schema", ()))
    if state_schema != STATE_SCHEMA or action_schema != ACTION_SCHEMA:
        raise ValueError(
            f"{policy_path} records state={state_schema} action={action_schema}, which is not "
            f"the centralized GR00T ordering state={STATE_SCHEMA} action={ACTION_SCHEMA}"
        )
    return count


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        if model_cfg.get("action_type") != "joint":
            raise NotImplementedError(
                "GauDP is a joint-space policy: it predicts the centralized absolute "
                "joint-target action, so deploy.yml/serve must set action_type=joint"
            )
        self.model_cfg = model_cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Only a cross-check now: the checkpoint's camera_order decides the views.
        requested_scene = model_cfg.get("use_scene")
        if os.environ.get("GAUDP_USE_SCENE"):
            requested_scene = bool(int(os.environ["GAUDP_USE_SCENE"]))

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
        preference = _checkpoint_preference(model_cfg)
        policy_path = _checkpoint_file(root, "policy", preference)
        payload = torch.load(policy_path, map_location="cpu", weights_only=False)
        robot_count = _check_checkpoint_contract(payload, policy_path)
        self.robots = robot_names(robot_count)
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
        recorded_cameras = list(payload.get("camera_order", []))
        unknown = [camera for camera in recorded_cameras if camera not in CAMERA_SLOT]
        if not recorded_cameras or unknown:
            raise ValueError(
                f"{policy_path} records camera_order={recorded_cameras}; known views are {sorted(CAMERA_SLOT)}"
            )
        if len(recorded_cameras) != int(config["num_views"]):
            raise ValueError(
                f"{policy_path} lists {len(recorded_cameras)} cameras {recorded_cameras} but its "
                f"config says num_views={config['num_views']}"
            )
        self.camera_order = recorded_cameras
        self.use_scene = SCENE_VIEW in recorded_cameras
        if requested_scene is not None and bool(requested_scene) != self.use_scene:
            raise ValueError(
                f"use_scene={requested_scene} but the checkpoint's views are {recorded_cameras}; "
                "the checkpoint decides, and use_scene/GAUDP_USE_SCENE only cross-check it"
            )
        if "robot_count" in config and int(config.pop("robot_count")) != robot_count:
            raise ValueError(f"{policy_path} config and state width disagree on the robot count")
        self.model = GauDPPolicy(**config, robot_count=robot_count)
        load_gaussian_checkpoint(
            self.model.gaussian_encoder, gaussian_path, strict=True, expect_views=len(self.camera_order)
        )
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
        # The vision recipe is whatever the checkpoint recorded; a checkpoint from
        # before those keys existed rebuilds through GauDPPolicy's legacy defaults.
        # `.eval()` above is what makes a configured crop the deterministic centre
        # crop rather than the training-time random one.
        print(
            f"[GauDP][vision] crop_shape={config.get('crop_shape', 'legacy:None')} "
            f"image_norm={config.get('image_norm', 'legacy:symmetric')} "
            f"group_norm_divisor={config.get('group_norm_divisor', 'legacy:None')}"
        )
        print(
            f"[GauDP] {len(self.robots)} robots {list(self.robots)}, views {self.camera_order} "
            f"-> slots {[CAMERA_SLOT[camera] for camera in self.camera_order]}"
        )
        print(f"[GauDP] loaded {policy_path} and {gaussian_path} on {self.device}")

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        indices = []
        for observation in obs_list:
            env_idx = int(observation.get("env_idx", 0))
            present = robots_in_observation(observation)
            if present != self.robots:
                raise ValueError(
                    f"the scene has robots {list(present)} but this checkpoint drives {list(self.robots)} "
                    f"({len(self.robots) * 43}D state / {len(self.robots) * ROBOT_ACTION_DIM}D action); "
                    "a centralized policy cannot serve a different robot count"
                )
            images, state = encode_observation(observation, self.camera_order, self.robots)
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
            [pack_xpolicy_action(action[batch, step], self.robots) for step in range(action.shape[1])]
            for batch in range(action.shape[0])
        ]

    def reset(self):
        self.runner.reset()
        self._env_indices = None
