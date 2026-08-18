from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name, resolve_checkpoint_root

_POLICY_DIR = Path(__file__).resolve().parent
_GR00T_ROOT = _POLICY_DIR / "gr00t_n17"
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"

if str(_GR00T_ROOT) not in sys.path:
    sys.path.insert(0, str(_GR00T_ROOT))

from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.policy import Gr00tPolicy  # noqa: E402

VIDEO_KEY_CANDIDATES = {
    "front": ["cam_head", "cam_high", "head_camera", "top_camera"],
    "left_wrist": ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"],
    "right_wrist": ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"],
}


def _load_modality_config(env_cfg_type: str) -> None:
    config_path = _POLICY_DIR / "configs" / f"{env_cfg_type}_config.py"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Modality config not found: {config_path}. Run process_data.sh for env_cfg_type={env_cfg_type} first."
        )
    spec = importlib.util.spec_from_file_location(f"gr00t_modality_{env_cfg_type}", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load modality config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _extract_step_number(value: Any) -> int | None:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


DEFAULT_COSMOS_MODEL_REPO = "nvidia/Cosmos-Reason2-2B"


def _resolve_relative_path(raw_path: str | Path, base_dir: Path) -> Path:
    """Resolve a deploy.yml path relative to base_dir."""
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        raise ValueError(
            f"Absolute paths are not supported: {path}. "
            f"Use a path relative to {base_dir} or set it in deploy.yml."
        )
    return (base_dir / path).resolve()


def _is_hf_repo_id(value: str) -> bool:
    if value.startswith((".", "/")) or "://" in value:
        return False
    parts = value.split("/")
    return len(parts) >= 2 and all(parts)


def _resolve_cosmos_model(model_cfg: dict[str, Any]) -> str:
    """Return HuggingFace repo id or a local path for Cosmos (processor backbone)."""
    raw_path = model_cfg.get("cosmos_model_path")
    if raw_path is None or raw_path == "":
        return DEFAULT_COSMOS_MODEL_REPO

    raw = str(raw_path)
    for candidate in (
        _POLICY_DIR / raw,
        _CHECKPOINTS_DIR / raw,
        _POLICY_DIR / "checkpoints" / raw,
    ):
        if (candidate / "config.json").is_file():
            return str(candidate.resolve())

    if _is_hf_repo_id(raw):
        return raw

    return str(_resolve_relative_path(raw, _POLICY_DIR))


@contextmanager
def _override_processor_cosmos_model(checkpoint_dir: Path, cosmos_model: str) -> Iterator[None]:
    """Replace baked-in absolute Cosmos paths in processor_config.json during load."""
    config_path = checkpoint_dir / "processor_config.json"
    if not config_path.is_file():
        yield
        return

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    processor_kwargs = data.setdefault("processor_kwargs", {})
    previous = processor_kwargs.get("model_name")
    if previous == cosmos_model:
        yield
        return

    processor_kwargs["model_name"] = cosmos_model
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    try:
        yield
    finally:
        if previous is not None:
            processor_kwargs["model_name"] = previous
        else:
            processor_kwargs.pop("model_name", None)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def _resolve_checkpoint_dir(model_cfg: dict[str, Any]) -> Path:
    # Shared precedence: model_dir key > ckpt_name-as-path >
    # {bench}-{ckpt}-{env}-{action}-{seed} concat > checkpoints/<ckpt_name>.
    root = resolve_checkpoint_root(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
        explicit_keys=("model_dir",),
    )
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")

    search_roots = [root]
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith("checkpoint-"):
            search_roots.append(child)

    candidates = []
    for search_root in search_roots:
        candidates.extend(sorted(search_root.glob("checkpoint-*"), key=lambda p: p.name))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directories under {root}")

    checkpoint_num = model_cfg.get("checkpoint_num")
    if checkpoint_num in (None, "last"):
        return max(candidates, key=lambda p: _extract_step_number(p.name) or -1)

    desired = _extract_step_number(checkpoint_num)
    if desired is not None:
        for candidate in candidates:
            if _extract_step_number(candidate.name) == desired:
                return candidate.resolve()

    explicit = root / f"checkpoint-{checkpoint_num}"
    if explicit.is_dir():
        return explicit.resolve()
    for search_root in search_roots:
        nested = search_root / f"checkpoint-{checkpoint_num}"
        if nested.is_dir():
            return nested.resolve()

    raise FileNotFoundError(
        f"Checkpoint step {checkpoint_num!r} not found under {root}. "
        f"Available: {[p.name for p in candidates]}"
    )


def _ensure_hwc_uint8(image: Any) -> np.ndarray:

    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return image
    if image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return image

    raise ValueError(f"Unsupported image shape: {image.shape}")


def _extract_image(observation: dict[str, Any], candidate_names: list[str]) -> np.ndarray:
    vision = observation.get("vision", {})
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "colors", "rgb"):
                if image_key in image:
                    return _ensure_hwc_uint8(image[image_key])
        else:
            return _ensure_hwc_uint8(image)
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def _to_rgb_hwc(image: np.ndarray) -> np.ndarray:
    """XPolicyLab obs images are RGB; match LeRobot video training."""
    return _ensure_hwc_uint8(image)


def _as_1d(value: Any, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError(f"Expected length {length}, got {arr.shape}")
    return arr


def _extract_prompt(observation: dict[str, Any], default_prompt: str) -> str:
    for key in ("instruction", "instructions"):
        if key not in observation:
            continue
        value = observation[key]
        if isinstance(value, dict):
            general = value.get("general")
            if isinstance(general, list) and general:
                first = general[0]
                if isinstance(first, dict):
                    conversations = first.get("conversations", [])
                    for turn in conversations:
                        if turn.get("from") == "human" and turn.get("value"):
                            text = str(turn["value"])
                            marker = "Generate robot actions for the task:\n"
                            if marker in text:
                                text = text.split(marker, 1)[1]
                            return text.replace(" /no_cot", "").strip()
        if isinstance(value, list):
            if not value:
                continue
            value = value[0]
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default_prompt


def _pack_arm_state(observation: dict[str, Any], side: str) -> np.ndarray:
    state = observation.get("state", {})
    prefix = f"{side}_"
    joint = _as_1d(state[f"{prefix}arm_joint_state"], 6)
    gripper = _as_1d(state[f"{prefix}ee_joint_state"], 1)
    return np.concatenate([joint, gripper], axis=0).astype(np.float32)


def _encode_observation(obs: dict[str, Any], default_prompt: str) -> dict[str, Any]:
    images = {
        video_key: _to_rgb_hwc(_extract_image(obs, candidates))
        for video_key, candidates in VIDEO_KEY_CANDIDATES.items()
    }
    prompt = _extract_prompt(obs, default_prompt)
    left_arm = _pack_arm_state(obs, "left")
    right_arm = _pack_arm_state(obs, "right")

    return {
        "video": {
            key: np.asarray(image, dtype=np.uint8)[None, None, ...]
            for key, image in images.items()
        },
        "state": {
            "left_arm": left_arm[None, None, :],
            "right_arm": right_arm[None, None, :],
        },
        "language": {
            "annotation.human.task_description": [[prompt]],
        },
    }


def _gr00t_action_to_env(action: dict[str, np.ndarray], action_type: str) -> list[dict[str, np.ndarray]]:
    left_arm = np.asarray(action["left_arm"][0], dtype=np.float32)
    right_arm = np.asarray(action["right_arm"][0], dtype=np.float32)
    horizon = left_arm.shape[0]

    if action_type != "joint":
        raise ValueError(
            f"GR00T_N17 RoboDojo arx_x5 is trained with joint-space relative actions (action_type=joint). "
            f"Got action_type={action_type!r}."
        )

    action_list: list[dict[str, np.ndarray]] = []
    for step in range(horizon):
        left = left_arm[step]
        right = right_arm[step]
        action_list.append(
            {
                "left_arm_joint_state": left[:6].astype(np.float32),
                "left_ee_joint_state": left[6:7].astype(np.float32),
                "right_arm_joint_state": right[:6].astype(np.float32),
                "right_ee_joint_state": right[6:7].astype(np.float32),
            }
        )
    return action_list


# --------------------------------------------------------------------------
# MHBench (two-humanoid, decentralized): one server hosts BOTH robots' policies
# and answers with the env's joint-space `mhbench_raw_action`
# (`scripts/mhbench_xpolicylab_env.py`: {joint_targets(31), base_vel(3),
# height(1)} per robot, Pink IK bypassed since MHBench a7d24c6).
# --------------------------------------------------------------------------

# observation.state's 43-D URDF-group layout (mhbench_keys.JOINT_GROUPS /
# mhbench.g1.actions.mhbench_state_joint_names()) -- the order
# MHBenchTaskEnv.get_obs() delivers `joint_pos` in.
MHBENCH_STATE_SLICES = {
    "left_leg": slice(0, 6), "right_leg": slice(6, 12), "waist": slice(12, 15),
    "left_arm": slice(15, 22), "left_hand": slice(22, 29),
    "right_arm": slice(29, 36), "right_hand": slice(36, 43),
}

# The 31 joint-target columns of the env action, in
# mhbench.g1.actions.gr00t_joint_names() order == the policy's own action
# groups concatenated in this sequence (mhbench_keys.ACTION_JOINT_GROUPS).
MHBENCH_JOINT_TARGET_GROUPS = ("left_arm", "right_arm", "left_hand", "right_hand", "waist")

# The sentences each robot's policy was trained on (meta/tasks.jsonl, indices
# 1/2 -- the shared sentence at index 0 belongs to the centralized policy).
MHBENCH_TASK_PROMPTS = {
    "cocarry": {
        "robot_a": "Hold your end of the board with both hands and side-step to your right, keeping the board level, until it rests on the stands.",
        "robot_b": "Hold your end of the board with both hands and side-step to your left, keeping the board level, until it rests on the stands.",
    },
    "handover": {
        "robot_a": "Pick up the bottle from the counter with your left hand, transfer it to your right hand, and hand it across to your partner.",
        "robot_b": "Receive the bottle with your right hand, transfer it to your left hand, and set it down at the far end of the counter.",
    },
}

# Which env camera slot carries each robot's own head camera
# (MHBenchTaskEnv.get_obs maps ego_a/ego_b/scene onto XPolicyLab's slot names).
MHBENCH_CAMERA_SLOT = {"robot_a": "cam_left_wrist", "robot_b": "cam_right_wrist"}

# The training run token: checkpoints were trained as
# mhbench-<task>_robot_<r>-unitree_g1x2_decentralized-joint-<seed>.
MHBENCH_TRAIN_ENV_CFG_TYPE = "unitree_g1x2_decentralized"


def _is_mhbench(model_cfg: dict[str, Any]) -> bool:
    return str(model_cfg.get("bench_name") or "") == "mhbench"


def _resolve_mhbench_model_dir(model_cfg: dict[str, Any], robot: str) -> Path:
    """The merged (standalone) model dir for one robot's decentralized policy.

    Training wrote PEFT adapters; `merge_lora_checkpoint.py` folded each into a
    `merged-<step>` dir next to its `checkpoint-<step>`. Explicit override:
    deploy key `model_dir_<robot>`.
    """
    explicit = model_cfg.get(f"model_dir_{robot}")
    if explicit:
        path = Path(explicit)
        if not path.is_dir():
            raise FileNotFoundError(f"model_dir_{robot} does not exist: {path}")
        return path

    task = str(model_cfg.get("ckpt_name") or "").strip()
    if not task:
        raise ValueError("mhbench mode needs ckpt_name=<task> (e.g. cocarry) or model_dir_<robot>")
    run_cfg = dict(model_cfg)
    run_cfg["ckpt_name"] = f"{task}_{robot}"  # e.g. cocarry_robot_a
    run_cfg["env_cfg_type"] = MHBENCH_TRAIN_ENV_CFG_TYPE
    run_name = build_run_dir_name(run_cfg)
    if run_name is None:
        raise ValueError("bench_name/ckpt_name/action_type/seed required to name the run dir")
    root = _CHECKPOINTS_DIR / run_name
    search_roots = [root, root / run_name]  # train.sh doubles the run dir
    wanted = str(model_cfg.get("merged_checkpoint", model_cfg.get("checkpoint_num", "last")))
    candidates = [d for r in search_roots if r.is_dir() for d in sorted(r.glob("merged-*")) if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"no merged-* model dir under {root} -- run baselines/scripts/merge_lora_checkpoint.py first"
        )
    if wanted in ("last", "None", ""):
        return max(candidates, key=lambda d: _extract_step_number(d.name) or -1)
    for d in candidates:
        if str(_extract_step_number(d.name)) == wanted:
            return d
    raise FileNotFoundError(f"merged-{wanted} not found; available: {[d.name for d in candidates]}")


def _encode_mhbench_observation(obs: dict[str, Any], robot: str, prompt: str) -> dict[str, Any]:
    """MHBenchTaskEnv obs -> one robot's Gr00tPolicy observation dict."""
    slot = MHBENCH_CAMERA_SLOT[robot]
    image = obs["vision"][slot]["color"]
    image = np.ascontiguousarray(_ensure_hwc_uint8(image))[None, None, ...]  # (1,1,H,W,3)

    robot_state = obs["mhbench_state"][robot]
    joints = np.asarray(robot_state["joint_pos"], dtype=np.float32).reshape(-1)
    if joints.shape[0] != 43:
        raise ValueError(f"{robot} joint_pos has {joints.shape[0]} dims, expected 43")
    state = {
        f"{robot}_{group}": joints[sl][None, None, :].astype(np.float32)
        for group, sl in MHBENCH_STATE_SLICES.items()
    }

    return {
        "video": {("ego_a" if robot == "robot_a" else "ego_b"): image},
        "state": state,
        "language": {f"annotation.human.task_description_{robot}": [[prompt]]},
    }


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        if _is_mhbench(model_cfg):
            self._init_mhbench(model_cfg)
            return
        self._mhbench = False
        self.model_cfg = model_cfg
        self.action_type = model_cfg.get("action_type", "joint")
        self.default_prompt = model_cfg.get("default_prompt", model_cfg.get("task_name", "Perform the robot manipulation task."))
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.device = model_cfg.get("device", "cuda:0" if self._has_cuda() else "cpu")

        _load_modality_config(self.env_cfg_type)
        checkpoint_dir = _resolve_checkpoint_dir(model_cfg)
        embodiment_tag = model_cfg.get("embodiment_tag", "NEW_EMBODIMENT")
        cosmos_model = _resolve_cosmos_model(model_cfg)

        with _override_processor_cosmos_model(checkpoint_dir, cosmos_model):
            self.policy = Gr00tPolicy(
                model_path=str(checkpoint_dir),
                embodiment_tag=embodiment_tag,
                device=self.device,
                strict=True,
            )
        self.model = self.policy
        self.action_horizon = len(self.policy.modality_configs["action"].delta_indices)

        self._obs_list: list[dict[str, Any]] = []
        self._latest_env_idx_list: list[int] = [0]

        print(f"[GR00T_N17] Loaded checkpoint from {checkpoint_dir}")
        print(f"[GR00T_N17] cosmos_model={cosmos_model}")
        print(f"[GR00T_N17] action_horizon={self.action_horizon}, embodiment_tag={embodiment_tag}")

    def _init_mhbench(self, model_cfg: dict[str, Any]) -> None:
        """Two decentralized Gr00tPolicy instances, one per robot, one server."""
        self._mhbench = True
        self.model_cfg = model_cfg
        self.device = model_cfg.get("device", "cuda:0" if self._has_cuda() else "cpu")
        task = str(model_cfg.get("ckpt_name") or "").strip()
        embodiment_tag = model_cfg.get("embodiment_tag", "NEW_EMBODIMENT")

        default_prompt = model_cfg.get("default_prompt", "Perform the robot manipulation task.")
        task_prompts = MHBENCH_TASK_PROMPTS.get(task, {})
        self._prompts = {
            robot: str(model_cfg.get(f"prompt_{robot}") or task_prompts.get(robot) or default_prompt)
            for robot in ("robot_a", "robot_b")
        }

        self._policies: dict[str, Gr00tPolicy] = {}
        for robot in ("robot_a", "robot_b"):
            model_dir = _resolve_mhbench_model_dir(model_cfg, robot)
            policy = Gr00tPolicy(
                model_path=str(model_dir),
                embodiment_tag=embodiment_tag,
                device=self.device,
                strict=True,
            )
            expected_cam = "ego_a" if robot == "robot_a" else "ego_b"
            video_keys = policy.modality_configs["video"].modality_keys
            if list(video_keys) != [expected_cam]:
                raise RuntimeError(
                    f"{robot} checkpoint at {model_dir} was trained on video keys {video_keys}, "
                    f"expected ['{expected_cam}'] -- wrong checkpoint pairing?"
                )
            self._policies[robot] = policy
            print(f"[GR00T_N17][mhbench] {robot}: {model_dir}")
            print(f"[GR00T_N17][mhbench] {robot} prompt: {self._prompts[robot]!r}")

        self.model = self._policies["robot_a"]
        self.action_horizon = len(self._policies["robot_a"].modality_configs["action"].delta_indices)
        self.exec_horizon = max(1, min(int(model_cfg.get("exec_horizon") or self.action_horizon), self.action_horizon))
        self._obs_list = []
        self._latest_env_idx_list = [0]
        print(f"[GR00T_N17][mhbench] action_horizon={self.action_horizon} exec_horizon={self.exec_horizon}")

    def _get_action_mhbench(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        per_robot: dict[str, list[dict[str, np.ndarray]]] = {}
        for robot, policy in self._policies.items():
            encoded = _encode_mhbench_observation(obs, robot, self._prompts[robot])
            action, _ = policy.get_action(encoded)
            groups = {
                key[len(robot) + 1 :]: np.asarray(value[0], dtype=np.float32)[: self.exec_horizon]
                for key, value in action.items()
                if key.startswith(robot + "_")
            }
            missing = [
                g for g in (*MHBENCH_JOINT_TARGET_GROUPS, "navigate_command", "base_height_command")
                if g not in groups
            ]
            if missing:
                raise KeyError(f"{robot} action is missing groups {missing}; got {sorted(groups)}")
            steps = min(len(groups[g]) for g in groups)
            per_robot[robot] = [
                {
                    # The policy predicts the same Pink-solved joint targets the
                    # env action consumes -- packing is a pure group reorder.
                    "joint_targets": np.concatenate(
                        [groups[g][t].reshape(-1) for g in MHBENCH_JOINT_TARGET_GROUPS]
                    ).astype(np.float32),
                    "base_vel": groups["navigate_command"][t].reshape(3).astype(np.float32),
                    "height": groups["base_height_command"][t].reshape(1).astype(np.float32),
                }
                for t in range(steps)
            ]
        steps = min(len(per_robot["robot_a"]), len(per_robot["robot_b"]))
        return [
            {"mhbench_raw_action": {"robot_a": per_robot["robot_a"][t], "robot_b": per_robot["robot_b"][t]}}
            for t in range(steps)
        ]

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        if self._mhbench:
            self._obs_list = list(obs_list)  # encoded lazily per robot in get_action
        else:
            self._obs_list = [_encode_observation(obs, self.default_prompt) for obs in obs_list]

    def get_action(self, **kwargs):
        if not self._obs_list:
            raise AssertionError("update_obs or update_obs_batch first!")
        return self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if not self._obs_list:
            raise AssertionError("update_obs or update_obs_batch first!")

        if self._mhbench:
            return [self._get_action_mhbench(obs) for obs in self._obs_list]

        action_list = []
        for encoded_obs in self._obs_list:
            gr00t_action, _ = self.policy.get_action(encoded_obs, **kwargs)
            action_list.append(_gr00t_action_to_env(gr00t_action, self.action_type))
        return action_list

    def reset(self):
        self._obs_list = []
        self._latest_env_idx_list = [0]
        if self._mhbench:
            for policy in self._policies.values():
                policy.reset()
        else:
            self.policy.reset()