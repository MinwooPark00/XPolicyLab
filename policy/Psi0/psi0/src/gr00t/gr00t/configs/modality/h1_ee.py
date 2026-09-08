import json
import os
from pathlib import Path

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


DATASET_PATH = os.environ.get("DATASET_PATH")
if not DATASET_PATH:
    raise RuntimeError("DATASET_PATH must be set to load modality.json")
META_PATH = Path(DATASET_PATH) / "meta" / "modality.json"
if not META_PATH.exists():
    raise RuntimeError(f"Missing modality.json at {META_PATH}")
try:
    with META_PATH.open("r") as f:
        MODALITY_META = json.load(f)
except Exception as exc:
    raise RuntimeError(f"Failed to load modality.json at {META_PATH}") from exc

TRAIN_STATE_KEYS = [
    "observation.arm_joints",
    "observation.hand_joints",
]
OPTIONAL_STATE_KEYS = ["observation.leg_joints"]
EXPECTED_ACTION_KEYS = [
    "action.wrists.left.xyz",
    "action.wrists.left.rpy",
    "action.wrists.right.xyz",
    "action.wrists.right.rpy",
    "action.hands.left_thumb.xyz",
    "action.hands.left_thumb.rpy",
    "action.hands.left_index.xyz",
    "action.hands.left_index.rpy",
    "action.hands.left_middle.xyz",
    "action.hands.left_middle.rpy",
    "action.hands.right_thumb.xyz",
    "action.hands.right_thumb.rpy",
    "action.hands.right_index.xyz",
    "action.hands.right_index.rpy",
    "action.hands.right_middle.xyz",
    "action.hands.right_middle.rpy",
]
EXPECTED_VIDEO_KEYS = ["egocentric"]
EXPECTED_ANNOTATION_KEYS = ["human.task_description"]

state_keys = list(MODALITY_META.get("state", {}).keys())
raw_action_keys = list(MODALITY_META.get("action", {}).keys())
video_keys = list(MODALITY_META.get("video", {}).keys())
annotation_keys = list(MODALITY_META.get("annotation", {}).keys())
missing_state_keys = sorted(set(TRAIN_STATE_KEYS) - set(state_keys))
unexpected_state_keys = sorted(
    set(state_keys) - set(TRAIN_STATE_KEYS) - set(OPTIONAL_STATE_KEYS)
)
if missing_state_keys or unexpected_state_keys:
    raise RuntimeError(
        "modality.json state keys mismatch: "
        f"missing={missing_state_keys}, unexpected={unexpected_state_keys}, actual={state_keys}"
    )
state_keys = TRAIN_STATE_KEYS
missing_action_keys = sorted(set(EXPECTED_ACTION_KEYS) - set(raw_action_keys))
if missing_action_keys:
    raise RuntimeError(
        "modality.json action keys mismatch: "
        f"missing={missing_action_keys}, actual={raw_action_keys}"
    )
action_keys = EXPECTED_ACTION_KEYS

if set(video_keys) != set(EXPECTED_VIDEO_KEYS):
    raise RuntimeError(f"modality.json video keys mismatch: {video_keys}")
if set(annotation_keys) != set(EXPECTED_ANNOTATION_KEYS):
    raise RuntimeError(f"modality.json annotation keys mismatch: {annotation_keys}")

ACTION_HORIZON = int(os.environ.get("ACTION_HORIZON", "16"))
if ACTION_HORIZON <= 0:
    raise RuntimeError(f"ACTION_HORIZON must be > 0, got {ACTION_HORIZON}")

h1_ee_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=video_keys,
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=state_keys,
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, ACTION_HORIZON)),
        modality_keys=action_keys,
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in action_keys
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[f"annotation.{key}" for key in annotation_keys],
    ),
}

register_modality_config(h1_ee_config, embodiment_tag=EmbodimentTag.H1_EE_A16)
