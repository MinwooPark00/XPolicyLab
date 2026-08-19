"""GR00T modality config for the MHBench two-humanoid *centralized* dataset --
one policy driving both robots, against the same ``datasets/<task>`` export the
decentralized configs read (the modality config does the slicing).

Thin shim, like ``unitree_g1x2_decentralized_config.py``: the key lists live in
MHBench's ``configs/gr00t/mhbench_modality.py``, the authority shared with the
exporter. For the duo target that means:

  - video: both robots' ego views (``ego_a``, ``ego_b``), and the fixed room
    camera (``scene``) only when asked for -- see below.
  - state: 86 = both robots' 43 joint angles.
  - action: 70 = both robots' 35 (arms 14 + hands 14 + waist 3 + base height 1
    + navigation 3), arms RELATIVE and the rest ABSOLUTE.
  - language: the pair's shared instruction
    (``annotation.human.task_description``).

MHBENCH_SCENE_CAMERA (default 0) adds the room camera. It is off by default
because the two ego views are what a deployed pair actually has; the room
camera is a third-person view of the scene that no robot carries. Every view is
just another image in the same VLM prompt -- N1.7 has no per-camera encoder --
so adding it costs image tokens (VRAM and step time) rather than new weights.

MHBENCH_ACTION_HORIZON (default 40 = ``ACTION_HORIZON_FINETUNE_SAFE``): the
released N1.7-3B checkpoint is a 40-step model. ``meta/relative_stats.json``
must be generated at the same horizon (gr00t/data/stats.py with this config).
"""

import os
import sys
from pathlib import Path

# baselines/XPolicyLab/policy/GR00T_N17/configs -> the MHBench checkout root
_MHBENCH_ROOT = Path(__file__).resolve().parents[5]
_MHBENCH_GR00T_CONFIG_DIR = Path(
    os.environ.get("MHBENCH_CONFIG_DIR", _MHBENCH_ROOT / "configs" / "gr00t")
)
if str(_MHBENCH_GR00T_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_MHBENCH_GR00T_CONFIG_DIR))

import mhbench_keys  # noqa: E402
import mhbench_modality  # noqa: E402

from gr00t.configs.data.embodiment_configs import register_modality_config  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.data.types import ModalityConfig  # noqa: E402


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"{name} must be a boolean flag (0/1), got {os.environ[name]!r}")


scene_camera = _env_flag("MHBENCH_SCENE_CAMERA")

action_horizon = int(
    os.environ.get("MHBENCH_ACTION_HORIZON", mhbench_keys.ACTION_HORIZON_FINETUNE_SAFE)
)

unitree_g1x2_centralized_config = mhbench_modality.build(
    robot=None, action_horizon=action_horizon
)

if not scene_camera:
    # Derived from the per-robot lists rather than by dropping "scene" by name,
    # so this keeps meaning what it says if the camera set is ever renamed.
    ego_views = [key for robot in ("robot_a", "robot_b") for key in mhbench_keys.video_keys(robot)]
    duo_views = unitree_g1x2_centralized_config["video"].modality_keys
    missing = [key for key in ego_views if key not in duo_views]
    if missing:
        raise ValueError(f"ego views {missing} are not in the duo camera set {duo_views}")
    unitree_g1x2_centralized_config["video"] = ModalityConfig(
        delta_indices=unitree_g1x2_centralized_config["video"].delta_indices,
        modality_keys=ego_views,
    )

register_modality_config(
    unitree_g1x2_centralized_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
