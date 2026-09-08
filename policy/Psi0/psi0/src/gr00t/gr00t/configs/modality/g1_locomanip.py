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
    raise RuntimeError("DATASET_PATH must be set to load SIMPLE modality.json")
META_PATH = Path(DATASET_PATH) / "meta" / "modality.json"
if not META_PATH.exists():
    raise RuntimeError(f"Missing modality.json at {META_PATH}")
try:
    MODALITY_META = json.load(META_PATH.open("r"))
except Exception:
    raise RuntimeError(f"Failed to load modality.json at {META_PATH}")

EXPECTED_STATE_KEYS = [
    "left_hand",
    "right_hand",
    "left_arm",
    "right_arm",
    "rpy",
    "height",
]
EXPECTED_ACTION_KEYS = [
    "left_hand",
    "right_hand",
    "left_arm",
    "right_arm",
    "rpy",
    "height",
    "torso_vx",
    "torso_vy",
    "torso_vyaw",
    "target_yaw",
]

state_keys = list(MODALITY_META.get("state", {}).keys())
action_keys = list(MODALITY_META.get("action", {}).keys())
if set(state_keys) != set(EXPECTED_STATE_KEYS):
    raise RuntimeError(f"modality.json state keys mismatch: {state_keys}")
if set(action_keys) != set(EXPECTED_ACTION_KEYS):
    raise RuntimeError(f"modality.json action keys mismatch: {action_keys}")

# Horizon is a training choice, not the action dimension.
ACTION_HORIZON = 16

g1_locomanip_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["rs_view"],
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
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(g1_locomanip_config, embodiment_tag=EmbodimentTag.G1_LOCO_DOWNSTREAM)
