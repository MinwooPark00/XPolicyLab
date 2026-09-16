"""Centralized MHBench joint-space contract shared by GauDP train and eval.

The ordering is identical to XPolicyLab's centralized GR00T adapter. Dataset
actions are absolute joint targets; GR00T's private arm-delta preprocessing is
not part of the environment-facing contract reproduced here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

# MHBench tasks carry two or three robots. The two-robot names and widths stay
# the module constants they always were -- they are the default of every
# count-taking signature below -- and the count itself travels as an argument,
# so one process can hold a 2-robot and a 3-robot policy at once.
ALL_ROBOT_NAMES = ("robot_a", "robot_b", "robot_c")
ROBOT_NAMES = ALL_ROBOT_NAMES[:2]
DEFAULT_ROBOT_COUNT = 2

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
ROBOT_ACTION_DIM = 35
JOINT_TARGET_DIM = 31
# The two-robot widths. Per-robot is the contract (43D state, 35D action); these
# are what two of them make, and nothing may size a tensor with them unless the
# task really has two robots -- ask the data, through the helpers below.
PROPRIO_DIM = 86
ACTION_DIM = 70


def robot_names(count: int = DEFAULT_ROBOT_COUNT) -> tuple[str, ...]:
    if count not in (2, 3):
        raise ValueError(f"MHBench tasks have two or three robots, not {count}")
    return ALL_ROBOT_NAMES[:count]


def proprio_dim(count: int = DEFAULT_ROBOT_COUNT) -> int:
    return len(robot_names(count)) * ROBOT_PROPRIO_DIM


def action_dim(count: int = DEFAULT_ROBOT_COUNT) -> int:
    return len(robot_names(count)) * ROBOT_ACTION_DIM


def robot_count_from_state_dim(width: int) -> int:
    """How many robots a flat joint-state width is, or a message naming both."""
    for count in (2, 3):
        if int(width) == proprio_dim(count):
            return count
    raise ValueError(
        f"{width}D state is not a whole number of MHBench robots: GauDP's joint contract is "
        f"43D state / 35D action per robot -- 86D state / 70D joint-action for two robots, "
        f"129D state / 105D joint-action for three"
    )


def robot_count_from_action_dim(width: int) -> int:
    for count in (2, 3):
        if int(width) == action_dim(count):
            return count
    raise ValueError(
        f"{width}D action is not a whole number of MHBench robots: GauDP's joint contract is "
        f"43D state / 35D action per robot -- 86D state / 70D joint-action for two robots, "
        f"129D state / 105D joint-action for three"
    )


# One ego view per robot, and the scene camera beside them. The slots are
# MHBenchTaskEnv._VISION_SLOT (scripts/mhbench_xpolicylab_env.py): `cam_third_view`
# is not an upstream slot name, it is what the env packs ego_c into.
EGO_VIEWS = ("ego_a", "ego_b", "ego_c")
SCENE_VIEW = "scene"
CAMERA_SLOT = {
    "ego_a": "cam_left_wrist",
    "ego_b": "cam_right_wrist",
    "ego_c": "cam_third_view",
    SCENE_VIEW: "cam_head",
}


def ego_views(count: int = DEFAULT_ROBOT_COUNT) -> tuple[str, ...]:
    return EGO_VIEWS[: len(robot_names(count))]


def robots_in_observation(observation: Mapping) -> tuple[str, ...]:
    """Which robots the live scene has, in canonical order (the env decides)."""
    present = observation.get("mhbench_state") or {}
    return tuple(robot for robot in ALL_ROBOT_NAMES if robot in present)


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
    if robot not in ALL_ROBOT_NAMES:
        raise ValueError(f"unknown robot {robot!r}")
    try:
        value = observation["mhbench_state"][robot]["joint_pos"]
    except KeyError as exc:
        raise KeyError(
            "GauDP requires observation['mhbench_state'][robot_a|robot_b|robot_c]['joint_pos']; "
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


def joint_state_from_observation(observation: Mapping, robots: Sequence[str]) -> np.ndarray:
    """The scene's robots' joint state, concatenated in the order given."""
    return np.concatenate([proprio_from_observation(observation, robot) for robot in robots])


def pack_xpolicy_action(action: np.ndarray, robots: Sequence[str] | None = None) -> dict[str, dict]:
    """Pack one centralized timestep for ``MHBenchTaskEnv.take_action``.

    ``robots`` names who the columns belong to; without it the robot count is
    read off the action's own width, which is how the contract tests call it.
    """
    action = np.asarray(action, dtype=np.float32)
    if robots is None:
        robots = robot_names(robot_count_from_action_dim(action.shape[-1] if action.ndim else 0))
    expected = len(robots) * ROBOT_ACTION_DIM
    if action.shape != (expected,):
        raise ValueError(
            f"centralized action must be {expected}D for {len(robots)} robots, got {action.shape}"
        )
    return {
        "mhbench_raw_action": {
            robot: split_robot_action(
                action[index * ROBOT_ACTION_DIM : (index + 1) * ROBOT_ACTION_DIM]
            )
            for index, robot in enumerate(robots)
        }
    }


def flat_action_from_xpolicy(action: Mapping, robots: Sequence[str] | None = None) -> np.ndarray:
    """Reference inverse of :func:`pack_xpolicy_action` for contract tests."""
    raw = action["mhbench_raw_action"]
    if robots is None:
        robots = tuple(robot for robot in ALL_ROBOT_NAMES if robot in raw)
        unknown = sorted(set(raw) - set(ALL_ROBOT_NAMES))
        if unknown or not robots:
            raise ValueError(f"packed action names unknown robots {unknown or sorted(raw)}")
    values = []
    for robot in robots:
        entry = raw[robot]
        joint_targets = np.asarray(entry["joint_targets"], dtype=np.float32).reshape(-1)
        height = np.asarray(entry["height"], dtype=np.float32).reshape(-1)
        base_vel = np.asarray(entry["base_vel"], dtype=np.float32).reshape(-1)
        value = np.concatenate((joint_targets, height, base_vel))
        if value.shape != (ROBOT_ACTION_DIM,):
            raise ValueError(f"{robot} packed action reconstructs to {value.shape}, expected (35,)")
        values.append(value)
    return np.concatenate(values).astype(np.float32)


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
