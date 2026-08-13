"""MHBench-specific GauDP observation and action schema.

Training uses xyzw quaternions because that is how MHBench records raw Isaac
observations.  The XPolicyLab environment side-channel exposes wxyz poses, so
the online adapter explicitly converts them back to xyzw here.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

ROBOT_NAMES = ("robot_a", "robot_b")
ROBOT_PROPRIO_DIM = 21
PROPRIO_DIM = 42
ROBOT_ACTION_DIM = 22
ACTION_DIM = 44
RAW_ROBOT_ACTION_DIM = 32

EEF_SOURCE_PATHS = (
    "obs/{robot}_left_eef_pos",
    "obs/{robot}_left_eef_rot",
    "obs/{robot}_right_eef_pos",
    "obs/{robot}_right_eef_rot",
)


def _as_2d(array: np.ndarray, length: int, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.shape[0] != length:
        raise ValueError(f"{name} has {array.shape[0]} frames; expected {length}")
    return array.reshape(length, -1)


def proprio_from_demo(demo, robot: str, length: int) -> np.ndarray:
    """Return pelvis + two EEF poses for one robot as ``(T, 21)``."""
    if robot not in ROBOT_NAMES:
        raise ValueError(f"unknown robot {robot!r}")
    root_pose_path = f"states/robots/{robot}/root_pose"
    if root_pose_path in demo:
        parts = [_as_2d(demo[root_pose_path], length, root_pose_path)]
    else:
        # GauDP.md's observation schema is also supported for datasets that
        # predate the consolidated recorder root_pose field.
        root_paths = (f"obs/{robot}_root_pos", f"obs/{robot}_root_rot")
        if any(path not in demo for path in root_paths):
            raise KeyError(
                f"{demo.name!r} has neither {root_pose_path!r} nor both {root_paths!r}"
            )
        parts = [_as_2d(demo[path], length, path) for path in root_paths]
    for template in EEF_SOURCE_PATHS:
        path = template.format(robot=robot)
        if path not in demo:
            raise KeyError(f"{demo.name!r} is missing required proprio field {path!r}")
        parts.append(_as_2d(demo[path], length, path))
    result = np.concatenate(parts, axis=-1)
    if result.shape != (length, ROBOT_PROPRIO_DIM):
        raise ValueError(f"{robot} proprio must be (T, 21), got {result.shape}")
    return result


def compress_robot_action(action32: np.ndarray) -> np.ndarray:
    """Compress one robot's 32D action to GauDP's 22D representation."""
    action32 = np.asarray(action32, dtype=np.float32)
    if action32.shape[-1] != RAW_ROBOT_ACTION_DIM:
        raise ValueError(f"expected 32D per-robot action, got {action32.shape[-1]}D")
    hand_cols = np.asarray((14, 15, 17, 18))
    return np.concatenate((action32[..., :14], action32[..., hand_cols], action32[..., 28:32]), axis=-1)


def compress_joint_action(action64: np.ndarray) -> np.ndarray:
    """Compress ``[robot_a 32D, robot_b 32D]`` into a centralized 44D action."""
    action64 = np.asarray(action64, dtype=np.float32)
    if action64.shape[-1] != 64:
        raise ValueError(f"GauDP requires raw 64D MHBench actions, got {action64.shape[-1]}D")
    result = np.concatenate((compress_robot_action(action64[..., :32]), compress_robot_action(action64[..., 32:])), axis=-1)
    assert result.shape[-1] == ACTION_DIM
    return result


def pose_wxyz_to_xyzw(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape[-1] < 7:
        raise ValueError(f"pose must have at least 7 values, got {pose.shape}")
    return np.concatenate((pose[..., :3], pose[..., 4:7], pose[..., 3:4]), axis=-1)


def pose_xyzw_to_wxyz(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape[-1] < 7:
        raise ValueError(f"pose must have at least 7 values, got {pose.shape}")
    return np.concatenate((pose[..., :3], pose[..., 6:7], pose[..., 3:6]), axis=-1)


def proprio_from_observation(observation: Mapping, robot: str) -> np.ndarray:
    """Read a real robot's 21D proprio from XPolicyLab's MHBench side-channel."""
    try:
        state = observation["mhbench_state"][robot]
    except KeyError as exc:
        raise KeyError(
            "GauDP requires observation['mhbench_state'][robot_a|robot_b]; "
            "it never substitutes actions for proprioception"
        ) from exc
    poses = (
        pose_wxyz_to_xyzw(state["pelvis_pose"]),
        pose_wxyz_to_xyzw(state["left_eef_pose"]),
        pose_wxyz_to_xyzw(state["right_eef_pose"]),
    )
    result = np.concatenate(poses).astype(np.float32)
    if result.shape != (ROBOT_PROPRIO_DIM,):
        raise ValueError(f"{robot} online proprio must be 21D, got {result.shape}")
    return result


def split_robot_action(action22: np.ndarray) -> dict[str, np.ndarray]:
    action22 = np.asarray(action22, dtype=np.float32)
    if action22.shape != (ROBOT_ACTION_DIM,):
        raise ValueError(f"per-robot action must be 22D, got {action22.shape}")
    return {
        "left_pose": action22[:7].copy(),
        "right_pose": action22[7:14].copy(),
        "hands": action22[14:18].copy(),
        "base_vel": action22[18:21].copy(),
        "height": action22[21:22].copy(),
    }


def pack_xpolicy_action(action44: np.ndarray, ee_dim: list[int] | tuple[int, int]) -> dict:
    """Convert a centralized GauDP timestep into XPolicyLab/MHBench action keys."""
    action44 = np.asarray(action44, dtype=np.float32)
    if action44.shape != (ACTION_DIM,):
        raise ValueError(f"centralized action must be 44D, got {action44.shape}")
    robot_a = split_robot_action(action44[:ROBOT_ACTION_DIM])
    robot_b = split_robot_action(action44[ROBOT_ACTION_DIM:])
    return {
        "left_ee_pose": pose_xyzw_to_wxyz(robot_a["left_pose"]),
        "right_ee_pose": pose_xyzw_to_wxyz(robot_a["right_pose"]),
        "left_ee_joint_state": np.full(int(ee_dim[0]), robot_a["hands"][:2].mean(), dtype=np.float32),
        "right_ee_joint_state": np.full(int(ee_dim[1]), robot_a["hands"][2:].mean(), dtype=np.float32),
        "mhbench_raw_action": {"robot_a": robot_a, "robot_b": robot_b},
    }


def _decompress_hands(hands4: np.ndarray) -> np.ndarray:
    """Mirror MHBenchTaskEnv's exact four-signal to 14-joint mapping."""
    l_idx, l_mid, r_idx, r_mid = np.asarray(hands4, dtype=np.float32)

    def hand(trigger: float, squeeze: float, left: bool) -> np.ndarray:
        thumb = max(trigger, squeeze)
        rotation = 0.5 * trigger - 0.5 * squeeze
        if not left:
            rotation = -rotation
        joints = np.asarray(
            [rotation, -thumb * 0.4, -thumb * 0.7, trigger, trigger, squeeze, squeeze],
            dtype=np.float32,
        )
        return -joints if left else joints

    left = hand(-float(l_idx), -float(l_mid), True)
    right = hand(float(r_idx), float(r_mid), False)
    lt, lp, ld, li, lid, lm, lmd = left
    rt, rp, rd, ri, rid, rm, rmd = right
    return np.asarray([li, lm, lt, ri, rm, rt, lid, lmd, lp, rid, rmd, rp, ld, rd], dtype=np.float32)


def simulator_action_from_xpolicy(action: Mapping) -> np.ndarray:
    """Reference reconstruction of ``mhbench_raw_action`` to simulator 64D."""
    raw = action["mhbench_raw_action"]
    robots = []
    for name in ROBOT_NAMES:
        robot = raw[name]
        robots.append(
            np.concatenate(
                (
                    np.asarray(robot["left_pose"], dtype=np.float32),
                    np.asarray(robot["right_pose"], dtype=np.float32),
                    _decompress_hands(robot["hands"]),
                    np.asarray(robot["base_vel"], dtype=np.float32),
                    np.asarray(robot["height"], dtype=np.float32),
                )
            )
        )
    result = np.concatenate(robots).astype(np.float32)
    if result.shape != (64,):
        raise ValueError(f"simulator action must be 64D, got {result.shape}")
    return result


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
