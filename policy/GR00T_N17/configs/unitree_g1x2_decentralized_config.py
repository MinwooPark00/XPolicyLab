"""GR00T modality config for the MHBench two-humanoid *decentralized* dataset
-- one policy per agent. Two shapes of it, selected by MHBENCH_ROBOT:

``shared`` (the shipped one)
    One policy over every task and both roles, trained on the flattened
    all-task export (``baselines/data/multitask/lerobot``, written by
    ``scripts/build_multitask_lerobot.py``). Each row of that dataset is
    already ONE robot, so its keys carry no ``robot_a_``/``robot_b_`` prefix
    and there is nothing to slice: video ``ego``, state ``left_leg`` ...
    ``right_hand``, language ``annotation.human.task_description``. What tells
    the two agents apart is the instruction each is given, not the key names.

``robot_a`` / ``robot_b``
    The older single-task per-robot runs, two separate finetunings against the
    same ``datasets/<task>/lerobot`` export -- the modality config does the
    per-robot slicing there, and the keys are prefixed accordingly (video
    ``ego_a``/``ego_b``, language
    ``annotation.human.task_description_robot_{a,b}``).

This file is a thin shim: the key lists live in MHBench's own
``configs/gr00t/mhbench_modality.py`` (the single authority shared with the
exporter), so the two cannot drift apart. Either way one agent sees:

  - video: one ego view.
  - state: 43 joint angles (G1 contract, seven groups).
  - action: 35 = arms 14 + hands 14 + waist 3 + base height 1 + navigation 3,
    arms RELATIVE, everything else ABSOLUTE -- NVIDIA's own
    ``unitree_g1x2_full_body_with_waist_height_nav_cmd`` placement.
  - language: one instruction.

Set MHBENCH_ROBOT before calling train.sh / finetune.sh (default robot_a):
each run only ever needs one config in a process, and
``baselines/scripts/train/GR00T_N17.sh`` sets it from the checkpoint name.

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
if robot not in ("robot_a", "robot_b", mhbench_keys.SHARED):
    raise ValueError(
        f"MHBENCH_ROBOT must be 'robot_a', 'robot_b' or '{mhbench_keys.SHARED}', got {robot!r}"
    )

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