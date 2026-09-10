"""Centralized MHBench joint-space contract shared by GauDP train and eval.

The ordering is identical to XPolicyLab's centralized GR00T adapter. Dataset
actions are absolute joint targets; GR00T's private arm-delta preprocessing is
not part of the environment-facing contract reproduced here.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

ROBOT_NAMES = ("robot_a", "robot_b")

STATE_GROUPS = (
    ("left_leg", 6),
    ("right_leg", 6),
    ("waist", 3),
    ("left_arm", 7),
    ("left_hand", 7),
    ("right_arm", 7),
    ("right_hand", 7),
)
ACTION_GROUPS = (
    ("left_arm", 7),
    ("right_arm", 7),
    ("left_hand", 7),
    ("right_hand", 7),
    ("waist", 3),
    ("base_height_command", 1),
    ("navigate_command", 3),
)

ROBOT_PROPRIO_DIM = 43
PROPRIO_DIM = 86
ROBOT_ACTION_DIM = 35
ACTION_DIM = 70
JOINT_TARGET_DIM = 31


def _group_slices(groups: tuple[tuple[str, int], ...]) -> dict[str, slice]:
    result = {}
    start = 0
    for name, width in groups:
        result[name] = slice(start, start + width)
        start += width
    return result


STATE_SLICES = _group_slices(STATE_GROUPS)
ACTION_SLICES = _group_slices(ACTION_GROUPS)
STATE_SCHEMA = tuple(name for name, _ in STATE_GROUPS)
ACTION_SCHEMA = tuple(name for name, _ in ACTION_GROUPS)


def proprio_from_observation(observation: Mapping, robot: str) -> np.ndarray:
    """Read one robot's 43D URDF-ordered joint state from MHBench."""
    if robot not in ROBOT_NAMES:
        raise ValueError(f"unknown robot {robot!r}")
    try:
        value = observation["mhbench_state"][robot]["joint_pos"]
    except KeyError as exc:
        raise KeyError(
            "GauDP requires observation['mhbench_state'][robot_a|robot_b]['joint_pos']; "
            "joint-space policies never substitute actions or EEF poses for proprioception"
        ) from exc
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (ROBOT_PROPRIO_DIM,):
        raise ValueError(f"{robot} joint_pos must be 43D, got {result.shape}")
    return result.copy()


def split_robot_action(action35: np.ndarray) -> dict[str, np.ndarray]:
    """Convert one absolute 35D GR00T-format action to the MHBench env keys."""
    action35 = np.asarray(action35, dtype=np.float32)
    if action35.shape != (ROBOT_ACTION_DIM,):
        raise ValueError(f"per-robot action must be 35D, got {action35.shape}")
    return {
        "joint_targets": action35[:JOINT_TARGET_DIM].copy(),
        "height": action35[JOINT_TARGET_DIM : JOINT_TARGET_DIM + 1].copy(),
        "base_vel": action35[JOINT_TARGET_DIM + 1 :].copy(),
    }


def pack_xpolicy_action(action70: np.ndarray) -> dict[str, dict]:
    """Pack one centralized 70D timestep for ``MHBenchTaskEnv.take_action``."""
    action70 = np.asarray(action70, dtype=np.float32)
    if action70.shape != (ACTION_DIM,):
        raise ValueError(f"centralized action must be 70D, got {action70.shape}")
    return {
        "mhbench_raw_action": {
            robot: split_robot_action(
                action70[index * ROBOT_ACTION_DIM : (index + 1) * ROBOT_ACTION_DIM]
            )
            for index, robot in enumerate(ROBOT_NAMES)
        }
    }


def flat_action_from_xpolicy(action: Mapping) -> np.ndarray:
    """Reference inverse of :func:`pack_xpolicy_action` for contract tests."""
    raw = action["mhbench_raw_action"]
    robots = []
    for robot in ROBOT_NAMES:
        entry = raw[robot]
        joint_targets = np.asarray(entry["joint_targets"], dtype=np.float32).reshape(-1)
        height = np.asarray(entry["height"], dtype=np.float32).reshape(-1)
        base_vel = np.asarray(entry["base_vel"], dtype=np.float32).reshape(-1)
        value = np.concatenate((joint_targets, height, base_vel))
        if value.shape != (ROBOT_ACTION_DIM,):
            raise ValueError(f"{robot} packed action reconstructs to {value.shape}, expected (35,)")
        robots.append(value)
    return np.concatenate(robots).astype(np.float32)


def pose7_xyzw_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert ``xyz + quaternion xyzw`` to a camera-to-world 4x4 matrix."""
    pose = np.asarray(pose, dtype=np.float32).reshape(7)
    x, y, z, w = pose[3:]
    norm = float(np.linalg.norm((x, y, z, w)))
    if norm < 1e-8:
        raise ValueError("zero-norm camera quaternion")
    x, y, z, w = (x / norm, y / norm, z / norm, w / norm)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = rotation
    result[:3, 3] = pose[:3]
    return result
