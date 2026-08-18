"""GR00T modality config for the MHBench two-humanoid *decentralized* dataset
-- one policy per robot, trained as two separate finetuning runs against the
same ``datasets/<task>/lerobot`` export (the modality config does the
per-robot slicing; no repacked per-robot dataset exists any more).

This file is a thin shim: the key lists live in MHBench's own
``configs/gr00t/mhbench_modality.py`` (the single authority shared with the
exporter), so the two cannot drift apart. Per robot that means:

  - video: that robot's own ego view only (``ego_a`` / ``ego_b``).
  - state: 43 joint angles (G1 contract, seven groups).
  - action: 35 = arms 14 + hands 14 + waist 3 + base height 1 + navigation 3,
    arms RELATIVE, everything else ABSOLUTE -- NVIDIA's own
    ``unitree_g1x2_full_body_with_waist_height_nav_cmd`` placement.
  - language: that robot's own instruction
    (``annotation.human.task_description_robot_{a,b}``).

Which robot this registers is selected by the MHBENCH_ROBOT env var
(robot_a/robot_b, default robot_a) -- set it before calling train.sh /
finetune.sh, since each run only ever needs one robot's config in a process.

The action horizon is MHBENCH_ACTION_HORIZON (default 40 =
``mhbench_keys.ACTION_HORIZON_FINETUNE_SAFE``): the released N1.7-3B
checkpoint is a 40-step model and ``launch_finetune.py`` has no override, so
the bench's nominal 50 cannot ride this path. ``meta/relative_stats.json``
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

robot = os.environ.get("MHBENCH_ROBOT", "robot_a")
if robot not in ("robot_a", "robot_b"):
    raise ValueError(f"MHBENCH_ROBOT must be 'robot_a' or 'robot_b', got {robot!r}")

action_horizon = int(
    os.environ.get("MHBENCH_ACTION_HORIZON", mhbench_keys.ACTION_HORIZON_FINETUNE_SAFE)
)

unitree_g1x2_decentralized_config = mhbench_modality.build(
    robot=robot, action_horizon=action_horizon
)

register_modality_config(
    unitree_g1x2_decentralized_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)