"""GR00T modality config for the MHBench two-humanoid *centralized* dataset.

Matches scripts/data_convertion.py --format groot --type centralized:
  - video: both robots' own ego view (ego_a, ego_b) -- no third-person "scene"
    camera by default (pass --cameras to the converter to add it back).
  - state: pelvis root_pose (7D) + left/right eef_pos (3D each) per robot --
    13D/robot, 26D combined. eef_rot is NOT listed: the source HDF5 has no
    per-hand end-effector *orientation* field at all (only position, plus a
    whole-robot root_rot quaternion that isn't per-hand) -- see
    datasets/data/*.hdf5's obs/ group. It's not something the converter can
    produce by re-running; it would need a new recorder term upstream in the
    simulator. Unlike XPolicyLab's other process_data.sh-driven adapters,
    GR00T here does NOT silently skip a modality_key missing from
    meta/modality.json -- lerobot_episode_loader.py's get_dataset_statistics()
    does a direct dict lookup with no guard, so a declared-but-absent key is a
    hard KeyError, not a no-op. Keep this list exactly in sync with
    meta/modality.json's actual "state" keys.
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

# How many future action steps to predict per query. NOT freely tunable up:
# the GR00T-N1.7-3B checkpoint's own architecture config fixes
# model_config.action_horizon=40 (gr00t_n1d7/setup.py passes it straight into
# processing_gr00t_n1d7.py's max_action_horizon), so requesting more here
# than the model supports pads with a negative dimension and crashes
# (RuntimeError: Trying to create tensor with negative dimension -10: [-10,
# ...] -- literally max_action_horizon(40) - this value). 50 (matching ACT's
# chunk_size / the pre-registered "unitree_g1_full_body_..." config, both for
# different base models) does not fit here; 40 is this checkpoint's ceiling.
ACTION_HORIZON = 40

_STATE_FIELDS = ("root_pose", "left_eef_pos", "right_eef_pos")

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
