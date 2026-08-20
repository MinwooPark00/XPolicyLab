import json

import h5py
import numpy as np
import pytest

from XPolicyLab.policy.GauDP.gaudp.dataset import (
    GaussianFrameDataset,
    _OPENCV_TO_ISAAC_CAMERA,
)
from XPolicyLab.policy.GauDP.process_data import _camera_intrinsics, _lerobot_state_action


def _pose_names(robot, prefix):
    # Deliberately use the source's wxyz order. The converter must select by
    # name and emit GauDP's xyz + quaternion_xyzw order.
    return [
        f"{robot}/{prefix}_pos_x",
        f"{robot}/{prefix}_pos_y",
        f"{robot}/{prefix}_pos_z",
        f"{robot}/{prefix}_quat_w",
        f"{robot}/{prefix}_quat_x",
        f"{robot}/{prefix}_quat_y",
        f"{robot}/{prefix}_quat_z",
    ]


def test_lerobot_named_fields_map_to_gaudp_contract():
    robots = ("robot_a", "robot_b")
    root_names = [name for robot in robots for name in _pose_names(robot, "root")]
    eef_names = [
        name
        for robot in robots
        for side in ("left", "right")
        for name in _pose_names(robot, f"{side}_wrist")
    ]
    joint_names = [
        f"{robot}/{side}_hand_{finger}_{joint}_joint"
        for robot in robots
        for side in ("left", "right")
        for finger in ("thumb", "middle", "index")
        for joint in (0, 1)
    ]
    navigation_names = [
        name
        for robot in robots
        for name in (f"{robot}/lin_vel_x", f"{robot}/lin_vel_y", f"{robot}/ang_vel_z")
    ]
    height_names = [f"{robot}/base_height" for robot in robots]
    names = {
        "observation.robots_state": root_names,
        "observation.eef_state": eef_names,
        "action.eef": eef_names,
        "action": joint_names,
        "teleop.navigate_command": navigation_names,
        "teleop.base_height_command": height_names,
    }
    info = {"features": {key: {"names": value} for key, value in names.items()}}
    columns = {
        key: np.arange(1000 * (offset + 1), 1000 * (offset + 1) + len(value), dtype=np.float32)[None]
        for offset, (key, value) in enumerate(names.items())
    }

    state, action = _lerobot_state_action(columns, info)
    assert state.shape == (1, 42)
    assert action.shape == (1, 44)

    def value(key, name):
        return columns[key][0, names[key].index(name)]

    np.testing.assert_array_equal(
        state[0, :7],
        [value("observation.robots_state", name) for name in (
            "robot_a/root_pos_x", "robot_a/root_pos_y", "robot_a/root_pos_z",
            "robot_a/root_quat_x", "robot_a/root_quat_y", "robot_a/root_quat_z",
            "robot_a/root_quat_w",
        )],
    )
    np.testing.assert_array_equal(
        action[0, 14:18],
        [value("action", name) for name in (
            "robot_a/left_hand_index_0_joint", "robot_a/left_hand_middle_0_joint",
            "robot_a/right_hand_index_0_joint", "robot_a/right_hand_middle_0_joint",
        )],
    )


def test_rgb_only_conversion_rejects_reconstruction_but_not_images(tmp_path):
    path = tmp_path / "rgb_only.hdf5"
    with h5py.File(path, "w") as target:
        target.create_dataset("episode_ends", data=np.asarray([1], dtype=np.int64))
        target.create_dataset("rgb_0", data=np.zeros((1, 2, 2, 3), dtype=np.uint8))
        target.create_dataset("rgb_1", data=np.zeros((1, 2, 2, 3), dtype=np.uint8))
        target.attrs["camera_order"] = json.dumps(["ego_a", "ego_b"])
        target.attrs["state_dim"] = 42
        target.attrs["action_dim"] = 44
        target.attrs["source_format"] = "lerobot-v2.1"
        target.attrs["gaussian_supervision"] = False

    with pytest.raises(ValueError, match="no Gaussian reconstruction supervision"):
        GaussianFrameDataset(path, train=True)


def test_current_lerobot_camera_metadata_and_axis_contract():
    matrix = [[160.0, 0.0, 160.0], [0.0, 160.0, 120.0], [0.0, 0.0, 1.0]]
    info = {
        "features": {
            "observation.depth.ego_a": {"info": {"camera.intrinsics": matrix}},
        }
    }
    np.testing.assert_array_equal(_camera_intrinsics(info, "ego_a"), matrix)

    # Dataset contract: OpenCV [right, down, forward] -> Isaac actor
    # [forward, left, up] = [z, -x, -y].
    right = _OPENCV_TO_ISAAC_CAMERA[:3, :3] @ np.asarray([1.0, 0.0, 0.0])
    down = _OPENCV_TO_ISAAC_CAMERA[:3, :3] @ np.asarray([0.0, 1.0, 0.0])
    forward = _OPENCV_TO_ISAAC_CAMERA[:3, :3] @ np.asarray([0.0, 0.0, 1.0])
    np.testing.assert_array_equal(right, [0.0, -1.0, 0.0])
    np.testing.assert_array_equal(down, [0.0, 0.0, -1.0])
    np.testing.assert_array_equal(forward, [1.0, 0.0, 0.0])
