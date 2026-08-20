import numpy as np

from XPolicyLab.policy.GauDP.gaudp.schema import (
    ACTION_DIM,
    PROPRIO_DIM,
    compress_joint_action,
    pack_xpolicy_action,
    pose_wxyz_to_xyzw,
    proprio_from_observation,
    simulator_action_from_xpolicy,
)


def test_dimensions_and_hand_compression():
    raw = np.arange(64, dtype=np.float32)
    compressed = compress_joint_action(raw)
    assert PROPRIO_DIM == 42
    assert ACTION_DIM == 44
    assert compressed.shape == (44,)
    np.testing.assert_array_equal(compressed[14:18], raw[[14, 15, 17, 18]])
    np.testing.assert_array_equal(compressed[36:40], raw[32 + np.array([14, 15, 17, 18])])


def test_wxyz_to_xyzw():
    pose = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
    np.testing.assert_array_equal(pose_wxyz_to_xyzw(pose), [1, 2, 3, 5, 6, 7, 4])


def test_dual_robot_proprio_is_42d_and_converts_quaternions():
    pose = np.array([1, 2, 3, 1, 0, 0, 0], dtype=np.float32)
    observation = {
        "mhbench_state": {
            name: {"pelvis_pose": pose, "left_eef_pose": pose, "right_eef_pose": pose}
            for name in ("robot_a", "robot_b")
        }
    }
    state = np.concatenate(
        [proprio_from_observation(observation, name) for name in ("robot_a", "robot_b")]
    )
    assert state.shape == (42,)
    np.testing.assert_array_equal(state[3:7], [0, 0, 0, 1])


def test_44d_side_channel_preserves_both_robots():
    action = np.arange(44, dtype=np.float32)
    packed = pack_xpolicy_action(action, [7, 7])
    assert set(packed["mhbench_raw_action"]) == {"robot_a", "robot_b"}
    for robot in packed["mhbench_raw_action"].values():
        assert robot["left_pose"].shape == (7,)
        assert robot["right_pose"].shape == (7,)
        assert robot["hands"].shape == (4,)
        assert robot["base_vel"].shape == (3,)
        assert robot["height"].shape == (1,)
    restored = simulator_action_from_xpolicy(packed)
    assert restored.shape == (64,)
    np.testing.assert_array_equal(restored[:14], action[:14])
    np.testing.assert_array_equal(restored[28:32], action[18:22])
    np.testing.assert_array_equal(restored[32:46], action[22:36])
    np.testing.assert_array_equal(restored[60:64], action[40:44])
