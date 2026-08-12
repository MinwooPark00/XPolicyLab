"""GR00T modality config for the MHBench two-humanoid *decentralized* dataset
-- one policy per robot, trained as two separate finetuning runs against that
robot's own scripts/data_convertion.py --format groot --type decentralized
output (a separate "..._lerobot_robot_a" / "..._lerobot_robot_b" dataset each).

Matches the converter's per-robot output:
  - video: that robot's own ego view only (ego_a for robot_a, ego_b for
    robot_b) -- no third-person "scene" camera by default.
  - state: pelvis root_pose (7D) + left/right eef_pos (3D each) + left/right
    eef_rot (4D each) -- 21D. eef_rot won't appear in the converted
    meta/modality.json until the recorder writes it (see
    scripts/data_convertion.py's DEFAULT_STATE_FIELDS/build_state_schema);
    listing it here is harmless either way, GR00T only loads keys present in
    modality.json.
  - action: one field, 22D (14D arm EEF pose + 4D hand signals + 3D base
    velocity + 1D height, via --compress-hands).

Which robot this registers is selected by the MHBENCH_ROBOT env var
(robot_a/robot_b, default robot_a) -- set it before calling finetune.sh, since
each run only ever needs one robot's config in that process:

    MHBENCH_ROBOT=robot_a bash examples/finetune.sh \
        --base-model-path <path> \
        --dataset-path <path/to/..._lerobot_robot_a> \
        --embodiment-tag new_embodiment \
        --modality-config-path configs/unitree_g1x2_decentralized_config.py \
        --output-dir <dir_a>

    MHBENCH_ROBOT=robot_b bash examples/finetune.sh \
        --base-model-path <path> \
        --dataset-path <path/to/..._lerobot_robot_b> \
        --embodiment-tag new_embodiment \
        --modality-config-path configs/unitree_g1x2_decentralized_config.py \
        --output-dir <dir_b>
"""

import os

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# See unitree_g1x2_centralized_config.py for why this is 50 and why every
# action group is NON_EEF/DEFAULT rather than an EEF pose format.
ACTION_HORIZON = 50

_STATE_FIELDS = ("root_pose", "left_eef_pos", "right_eef_pos", "left_eef_rot", "right_eef_rot")

robot = os.environ.get("MHBENCH_ROBOT", "robot_a")
if robot not in ("robot_a", "robot_b"):
    raise ValueError(f"MHBENCH_ROBOT must be 'robot_a' or 'robot_b', got {robot!r}")
ego_cam = "ego_a" if robot == "robot_a" else "ego_b"

unitree_g1x2_decentralized_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[ego_cam],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[f"{robot}_{f}" for f in _STATE_FIELDS],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[robot],
        action_configs=[
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

register_modality_config(
    unitree_g1x2_decentralized_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
