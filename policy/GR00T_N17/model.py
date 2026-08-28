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
        "robot_a": "Hold your side of the basket with both hands and side-step to your right, keeping it level, until it rests on the green patch.",
        "robot_b": "Hold your side of the basket with both hands and side-step to your left, keeping it level, until it rests on the green patch.",
    },
    "frame_hang": {
        "robot_a": "Take the carrying handle on the left side of the frame with one hand, lift level with your partner, guide the loop onto the peg, and release once it is seated.",
        "robot_b": "Take the carrying handle on the right side of the frame with one hand, lift level with your partner, guide the loop onto the peg, and release once it is seated.",
    },
    "door_passage": {
        "robot_a": "Walk to the door, pull it open, and hold it open for your partner.",
        "robot_b": "Lift the trophy off the plinth by its handles, carry it through the open doorway, and stand it on the green stand.",
    },
    "handover": {
        "robot_a": "Pick up the bottle directly in front of you with your right hand and hand it to your partner's right hand.",
        "robot_b": "Receive the bottle with your right hand, place it on the target directly in front of you, and release it.",
    },
}

# The runner names a task by its dataset spelling (the word the checkpoints were
# built under); the tables above are keyed by the scene. Both reach the same
# sentences -- a lookup that misses would prompt the policy with a generic
# sentence it never trained on, and say nothing.
MHBENCH_TASK_ALIASES = {
    "framehang": "frame_hang",
    "doorpassage": "door_passage",
    "handover_easy": "handover",
    "handovereasy": "handover",
}


def _mhbench_task_key(task: str) -> str:
    return MHBENCH_TASK_ALIASES.get(task, task)

# Which env camera slot carries each dataset camera: MHBenchTaskEnv.get_obs
# packs ego_a/ego_b/scene into XPolicyLab's standard bimanual slot names.
MHBENCH_VIDEO_SLOT = {
    "ego_a": "cam_left_wrist", "ego_b": "cam_right_wrist", "scene": "cam_head",
}
MHBENCH_EGO_VIEW = {"robot_a": "ego_a", "robot_b": "ego_b"}

# The shared instruction a centralized policy is trained on (meta/tasks.jsonl
# index 0). Add a task here from its own dataset rather than guessing it -- a
# policy prompted with a sentence it never saw is a silent failure.
MHBENCH_DUO_PROMPTS = {
    "cocarry": "Carry the laundry basket together and set it down level on the green patch on the far shelf.",
    "door_passage": "Open the door and carry the trophy through the doorway onto the stand in the far room.",
    "frame_hang": "Lift the framed painting off the floor together, hang the loop on its top rail on the wall peg, and let go once it is hanging square.",
    "handover": "Pass the bottle directly from Robot A's right hand to Robot B's right hand, then place it on the target.",
}

# The training run token per mode: checkpoints were trained as
# mhbench-<task>_robot_<r>-unitree_g1x2_decentralized-joint-<seed>, or
# mhbench-<task>-unitree_g1x2_centralized-joint-<seed> for one policy driving both.
MHBENCH_TRAIN_ENV_CFG_TYPE = "unitree_g1x2_decentralized"
MHBENCH_CENTRALIZED_ENV_CFG_TYPE = "unitree_g1x2_centralized"


def _is_mhbench(model_cfg: dict[str, Any]) -> bool:
    return str(model_cfg.get("bench_name") or "") == "mhbench"


def _resolve_mhbench_model_dir(model_cfg: dict[str, Any], robot: str | None) -> Path:
    """The merged (standalone) model dir for one mhbench policy.

    `robot` names one robot's own decentralized policy, ``"shared"`` for the
    one multitask policy that drives either, or None for the centralized one
    driving both. Training wrote PEFT adapters; a full finetune or
    `merge_lora_checkpoint.py` leaves a `merged-<step>` dir beside the
    `checkpoint-<step>`. Explicit override: deploy key `model_dir_<robot>`, or
    `model_dir` for the shared and centralized runs, which are one directory.
    """
    explicit_key = "model_dir" if robot in (None, "shared") else f"model_dir_{robot}"
    explicit = model_cfg.get(explicit_key)
    if explicit:
        path = Path(explicit)
        if not path.is_dir():
            raise FileNotFoundError(f"{explicit_key} does not exist: {path}")
        return path

    task = str(model_cfg.get("ckpt_name") or "").strip()
    if not task:
        raise ValueError(f"mhbench mode needs ckpt_name=<task> (e.g. cocarry) or {explicit_key}")
    run_cfg = dict(model_cfg)
    if robot is None:
        run_cfg["env_cfg_type"] = MHBENCH_CENTRALIZED_ENV_CFG_TYPE
    elif robot == "shared":
        # ckpt_name is already the run's own name (`multitask`) -- the policy
        # is not named for a task or a robot, because it was trained on all of
        # them at once.
        run_cfg["env_cfg_type"] = MHBENCH_TRAIN_ENV_CFG_TYPE
    else:
        run_cfg["ckpt_name"] = f"{task}_{robot}"  # e.g. cocarry_robot_a
        run_cfg["env_cfg_type"] = MHBENCH_TRAIN_ENV_CFG_TYPE
    run_name = build_run_dir_name(run_cfg)
    if run_name is None:
        raise ValueError("bench_name/ckpt_name/action_type/seed required to name the run dir")
    # train.sh appends CKPT_TAG to the run directory for runs that train the same
    # data differently (full instead of LoRA, an extra camera). Carry it here too,
    # or evaluating one would quietly load the other.
    ckpt_tag = str(model_cfg.get("ckpt_tag") or "").strip()
    if ckpt_tag:
        run_name = f"{run_name}-{ckpt_tag}"
    root = _CHECKPOINTS_DIR / run_name
    search_roots = [root, root / run_name]  # train.sh doubles the run dir
    wanted = str(model_cfg.get("merged_checkpoint", model_cfg.get("checkpoint_num", "last")))
    candidates = [d for r in search_roots if r.is_dir() for d in sorted(r.glob("merged-*")) if d.is_dir()]
    if not candidates:
        # A full finetune writes the whole model into checkpoint-<step>, so there
        # is nothing to merge and no merged-<step> to find. Only a LoRA run needs
        # one, and it says so by leaving an adapter_config.json behind.
        checkpoints = [d for r in search_roots if r.is_dir() for d in sorted(r.glob("checkpoint-*")) if d.is_dir()]
        candidates = [d for d in checkpoints if not (d / "adapter_config.json").is_file()]
        if not candidates:
            raise FileNotFoundError(
                f"no loadable model dir under {root}: "
                + (
                    "its checkpoints hold LoRA adapters -- run "
                    "baselines/scripts/merge_lora_checkpoint.sbatch first"
                    if checkpoints
                    else "it has no checkpoint-* at all"
                )
            )
    if wanted in ("last", "None", ""):
        return max(candidates, key=lambda d: _extract_step_number(d.name) or -1)
    for d in candidates:
        if str(_extract_step_number(d.name)) == wanted:
            return d
    raise FileNotFoundError(f"merged-{wanted} not found; available: {[d.name for d in candidates]}")


def _mhbench_view(obs: dict[str, Any], view: str) -> np.ndarray:
    """One dataset camera as the (1, 1, H, W, 3) frame Gr00tPolicy expects."""
    slot = MHBENCH_VIDEO_SLOT.get(view)
    if slot is None:
        raise KeyError(f"no env camera slot for view {view!r}; known: {sorted(MHBENCH_VIDEO_SLOT)}")
    return np.ascontiguousarray(_ensure_hwc_uint8(obs["vision"][slot]["color"]))[None, None, ...]


def _mhbench_state(obs: dict[str, Any], robot: str) -> dict[str, np.ndarray]:
    """One robot's 43 joint angles, split into the policy's seven state groups."""
    joints = np.asarray(obs["mhbench_state"][robot]["joint_pos"], dtype=np.float32).reshape(-1)
    if joints.shape[0] != 43:
        raise ValueError(f"{robot} joint_pos has {joints.shape[0]} dims, expected 43")
    return {
        f"{robot}_{group}": joints[sl][None, None, :].astype(np.float32)
        for group, sl in MHBENCH_STATE_SLICES.items()
    }


def _encode_mhbench_observation(obs: dict[str, Any], robot: str, prompt: str) -> dict[str, Any]:
    """MHBenchTaskEnv obs -> one robot's Gr00tPolicy observation dict."""
    return {
        "video": {MHBENCH_EGO_VIEW[robot]: _mhbench_view(obs, MHBENCH_EGO_VIEW[robot])},
        "state": _mhbench_state(obs, robot),
        "language": {f"annotation.human.task_description_{robot}": [[prompt]]},
    }


def _mhbench_shared_state(obs: dict[str, Any], robot: str) -> dict[str, np.ndarray]:
    """One robot's 43 joint angles under the shared policy's unprefixed keys.

    The flattened all-task dataset has one robot per row, so its state groups
    are `left_leg` ... `right_hand` rather than `robot_a_left_leg`. Whose
    joints they are is decided here, by which robot's block of the observation
    is read -- and told to the policy by the instruction, not by the key names.
    """
    return {
        group: value
        for key, value in _mhbench_state(obs, robot).items()
        for group in (key[len(robot) + 1 :],)
    }


def _encode_mhbench_shared_observation(obs: dict[str, Any], robot: str, prompt: str) -> dict[str, Any]:
    """MHBenchTaskEnv obs -> the shared multitask policy's observation dict."""
    return {
        "video": {"ego": _mhbench_view(obs, MHBENCH_EGO_VIEW[robot])},
        "state": _mhbench_shared_state(obs, robot),
        "language": {"annotation.human.task_description": [[prompt]]},
    }


def _encode_mhbench_duo_observation(
    obs: dict[str, Any], prompt: str, views: list[str]
) -> dict[str, Any]:
    """MHBenchTaskEnv obs -> the centralized policy's observation dict.

    `views` comes from the loaded checkpoint rather than a constant, so a policy
    trained with the room camera (MHBENCH_SCENE_CAMERA=1) and one trained on the
    two ego views alone are both served without a second switch to set.
    """
    return {
        "video": {view: _mhbench_view(obs, view) for view in views},
        "state": {**_mhbench_state(obs, "robot_a"), **_mhbench_state(obs, "robot_b")},
        "language": {"annotation.human.task_description": [[prompt]]},
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
        """One server for one MHBench task, decentralized (default) or centralized.

        Decentralized holds two Gr00tPolicy instances, one per robot; centralized
        holds the single policy that drives both. `mhbench_mode` in deploy.yml or
        the eval overrides picks between them.
        """
        self._mhbench = True
        self.model_cfg = model_cfg
        self.device = model_cfg.get("device", "cuda:0" if self._has_cuda() else "cpu")
        task = str(model_cfg.get("ckpt_name") or "").strip()
        embodiment_tag = model_cfg.get("embodiment_tag", "NEW_EMBODIMENT")
        self._mode = str(model_cfg.get("mhbench_mode") or "decentralized").strip().lower()
        if self._mode not in ("decentralized", "centralized"):
            raise ValueError(
                f"mhbench_mode must be 'decentralized' or 'centralized', got {self._mode!r}"
            )

        default_prompt = model_cfg.get("default_prompt", "Perform the robot manipulation task.")
        # What "decentralized" means for this checkpoint: `shared` is one
        # multitask policy driving both agents, told them apart by the
        # instruction each is given; `per_robot` is the older pair, one
        # checkpoint per (task, robot). eval_policy.sbatch sends whichever the
        # serving hook declares.
        self._style = str(model_cfg.get("mhbench_decentralized_style") or "per_robot").strip().lower()
        if self._style not in ("shared", "per_robot"):
            raise ValueError(
                f"mhbench_decentralized_style must be 'shared' or 'per_robot', got {self._style!r}"
            )

        if self._mode == "centralized":
            self._init_mhbench_centralized(model_cfg, task, embodiment_tag, default_prompt)
            return

        # Fallbacks only. The instruction each agent is actually given comes off
        # the wire (`mhbench_instruction`, from scripts/_task_text.py), so these
        # cover a deploy.py driving this model outside the eval client -- and a
        # deploy.yml `prompt_<robot>` still wins over both.
        task_prompts = MHBENCH_TASK_PROMPTS.get(_mhbench_task_key(task), {})
        self._prompt_overrides = {
            robot: model_cfg.get(f"prompt_{robot}") for robot in ("robot_a", "robot_b")
        }
        self._prompts = {
            robot: str(self._prompt_overrides[robot] or task_prompts.get(robot) or default_prompt)
            for robot in ("robot_a", "robot_b")
        }

        if self._style == "shared":
            self._init_mhbench_shared(model_cfg, embodiment_tag)
            return

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

    def _init_mhbench_shared(self, model_cfg: dict[str, Any], embodiment_tag: str) -> None:
        """One policy, both agents -- the shipped decentralized checkpoint.

        Trained on the flattened all-task dataset, so it is queried once per
        robot with that robot's own camera and state and that agent's own
        instruction. One set of weights on the card instead of two, which is
        also why a 96 GB policy fits where the pair did not.
        """
        model_dir = _resolve_mhbench_model_dir(model_cfg, "shared")
        policy = Gr00tPolicy(
            model_path=str(model_dir),
            embodiment_tag=embodiment_tag,
            device=self.device,
            strict=True,
        )
        video_keys = list(policy.modality_configs["video"].modality_keys)
        if video_keys != ["ego"]:
            raise RuntimeError(
                f"shared checkpoint at {model_dir} was trained on video keys {video_keys}, "
                "expected ['ego'] -- a per-robot checkpoint is being served as the shared one. "
                "MHBENCH_DECENTRALIZED_STYLE=per_robot evaluates those."
            )
        self._policy_shared = policy
        self.model = policy
        self.action_horizon = len(policy.modality_configs["action"].delta_indices)
        self.exec_horizon = max(1, min(int(model_cfg.get("exec_horizon") or self.action_horizon), self.action_horizon))
        self._obs_list = []
        self._latest_env_idx_list = [0]
        print(f"[GR00T_N17][mhbench] shared (multitask): {model_dir}")
        print(f"[GR00T_N17][mhbench] action_horizon={self.action_horizon} exec_horizon={self.exec_horizon}")

    def _init_mhbench_centralized(
        self, model_cfg: dict[str, Any], task: str, embodiment_tag: str, default_prompt: str
    ) -> None:
        """The single policy that sees both robots and answers for both."""
        self._prompt_duo_override = model_cfg.get("prompt_duo")
        prompt = self._prompt_duo_override or MHBENCH_DUO_PROMPTS.get(_mhbench_task_key(task))
        if not prompt:
            raise ValueError(
                f"no shared instruction registered for task {task!r}: add it to "
                "MHBENCH_DUO_PROMPTS from that dataset's meta/tasks.jsonl (index 0), "
                "or pass prompt_duo. Falling back to a generic sentence would prompt "
                "the policy with words it never trained on."
            )
        self._prompt_duo = str(prompt)

        model_dir = _resolve_mhbench_model_dir(model_cfg, None)
        policy = Gr00tPolicy(
            model_path=str(model_dir),
            embodiment_tag=embodiment_tag,
            device=self.device,
            strict=True,
        )
        self._views = list(policy.modality_configs["video"].modality_keys)
        unknown = [view for view in self._views if view not in MHBENCH_VIDEO_SLOT]
        if unknown:
            raise RuntimeError(
                f"checkpoint at {model_dir} was trained on video keys {unknown}, which the "
                f"env does not carry; it has {sorted(MHBENCH_VIDEO_SLOT)}"
            )
        self._policy_duo = policy
        self.model = policy
        self.action_horizon = len(policy.modality_configs["action"].delta_indices)
        self.exec_horizon = max(1, min(int(model_cfg.get("exec_horizon") or self.action_horizon), self.action_horizon))
        self._obs_list = []
        self._latest_env_idx_list = [0]
        print(f"[GR00T_N17][mhbench] centralized: {model_dir}")
        print(f"[GR00T_N17][mhbench] views={self._views} prompt={self._prompt_duo!r}")
        print(f"[GR00T_N17][mhbench] action_horizon={self.action_horizon} exec_horizon={self.exec_horizon}")

    def _pack_robot_action(
        self, action: dict[str, np.ndarray], robot: str, *, prefixed: bool = True
    ) -> list[dict[str, np.ndarray]]:
        """One robot's slice of a policy output -> the env's `mhbench_raw_action`.

        `prefixed` is False for the shared multitask policy, whose output keys
        are `left_arm` rather than `robot_a_left_arm`: its dataset has one robot
        per row, so there is only ever one set of groups to name.
        """
        prefix = f"{robot}_" if prefixed else ""
        groups = {
            key[len(prefix) :]: np.asarray(value[0], dtype=np.float32)[: self.exec_horizon]
            for key, value in action.items()
            if key.startswith(prefix)
        }
        missing = [
            g for g in (*MHBENCH_JOINT_TARGET_GROUPS, "navigate_command", "base_height_command")
            if g not in groups
        ]
        if missing:
            raise KeyError(f"{robot} action is missing groups {missing}; got {sorted(groups)}")
        steps = min(len(groups[g]) for g in groups)
        return [
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

    def _mhbench_policies(self) -> list[Gr00tPolicy]:
        """Every policy this server holds: one, or the per-robot pair."""
        if self._mode == "centralized":
            return [self._policy_duo]
        if self._style == "shared":
            return [self._policy_shared]
        return list(self._policies.values())

    def _mhbench_prompt(self, obs: dict[str, Any], robot: str) -> str:
        """This agent's instruction: a deploy override, else what the env sent,
        else the sentence table.

        The env is the authority (scripts/_task_text.py, one copy for training
        and serving); the table is a fallback for a deploy.py driving this model
        outside the eval client, and a stale entry there can no longer decide
        what a policy is told."""
        override = self._prompt_overrides.get(robot)
        if override:
            return str(override)
        wire = (obs.get("mhbench_instruction") or {}).get(robot)
        return str(wire) if wire else self._prompts[robot]

    def _get_action_mhbench(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        if self._mode == "centralized":
            prompt = self._prompt_duo_override or (obs.get("mhbench_instruction") or {}).get("duo") or self._prompt_duo
            encoded = _encode_mhbench_duo_observation(obs, str(prompt), self._views)
            action, _ = self._policy_duo.get_action(encoded)
            per_robot = {robot: self._pack_robot_action(action, robot) for robot in ("robot_a", "robot_b")}
        elif self._style == "shared":
            # One policy, queried once per agent. Same weights, same camera
            # geometry -- the instruction is the only thing that differs, which
            # is what makes this a decentralized pair rather than one policy
            # doing the task twice.
            per_robot = {}
            for robot in ("robot_a", "robot_b"):
                encoded = _encode_mhbench_shared_observation(obs, robot, self._mhbench_prompt(obs, robot))
                action, _ = self._policy_shared.get_action(encoded)
                per_robot[robot] = self._pack_robot_action(action, robot, prefixed=False)
        else:
            per_robot = {}
            for robot, policy in self._policies.items():
                encoded = _encode_mhbench_observation(obs, robot, self._mhbench_prompt(obs, robot))
                action, _ = policy.get_action(encoded)
                per_robot[robot] = self._pack_robot_action(action, robot)
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
            for policy in self._mhbench_policies():
                policy.reset()
        else:
            self.policy.reset()
