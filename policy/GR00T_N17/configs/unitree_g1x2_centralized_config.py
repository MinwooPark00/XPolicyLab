"""GR00T modality config for the MHBench two-humanoid *centralized* dataset.

Matches scripts/data_convertion.py --format groot --type centralized:
  - video: both robots' own ego view (ego_a, ego_b) -- no third-person "scene"
    camera by default (pass --cameras to the converter to add it back).
  - state: pelvis root_pose (7D) + left/right eef_pos (3D each) + left/right
    eef_rot (4D each) per robot -- 21D/robot, 42D combined. eef_rot won't
    appear in the converted meta/modality.json until the recorder writes it
    (see scripts/data_convertion.py's DEFAULT_STATE_FIELDS/build_state_schema);
    listing it here is harmless either way, GR00T only loads keys present in
    modality.json.
  - action: one "robot_a"/"robot_b" field each, 22D per robot (14D arm EEF
    pose + 4D hand signals + 3D base velocity + 1D height, via
    --compress-hands), 44D combined.

Usage:
    bash examples/finetune.sh \
        --base-model-path <path> \
        --dataset-path <path/to/..._lerobot> \
        --embodiment-tag new_embodiment \
        --modality-config-path configs/unitree_g1x2_centralized_config.py \
        --output-dir <dir>
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# How many future action steps to predict per query; tunable, matches this
# project's ACT chunk_size (policy/ACT/train.sh --chunk_size 50) and the
# pre-registered "unitree_g1_full_body_with_waist_height_nav_cmd" config in
# gr00t/configs/data/embodiment_configs.py.
ACTION_HORIZON = 50

_STATE_FIELDS = ("root_pose", "left_eef_pos", "right_eef_pos", "left_eef_rot", "right_eef_rot")

# rep=ABSOLUTE (the converter writes absolute poses/signals, not deltas);
# type=NON_EEF, format=DEFAULT for every group, not just the non-pose ones --
# ActionFormat has no plain xyz+quat option (only xyz+rot6d/xyz+rotvec), and
# the pre-registered G1 config in this same repo uses NON_EEF/DEFAULT for its
# arm groups too, so this follows that precedent rather than reshaping our
# 7D-per-eef quat data into a representation nothing here asked for.
_ROBOT_ACTION_CONFIG = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)

unitree_g1x2_centralized_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_a", "ego_b"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[f"robot_a_{f}" for f in _STATE_FIELDS] + [f"robot_b_{f}" for f in _STATE_FIELDS],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["robot_a", "robot_b"],
        action_configs=[_ROBOT_ACTION_CONFIG, _ROBOT_ACTION_CONFIG],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    unitree_g1x2_centralized_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
