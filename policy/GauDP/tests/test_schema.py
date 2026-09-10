import numpy as np
import pytest

from XPolicyLab.policy.GauDP.gaudp.schema import (
    ACTION_DIM,
    ACTION_SCHEMA,
    ACTION_SLICES,
    JOINT_TARGET_DIM,
    PROPRIO_DIM,
    ROBOT_ACTION_DIM,
    ROBOT_NAMES,
    ROBOT_PROPRIO_DIM,
    STATE_SCHEMA,
    STATE_SLICES,
    flat_action_from_xpolicy,
    pack_xpolicy_action,
    proprio_from_observation,
    split_robot_action,
)


def test_contract_matches_the_centralized_groot_embodiment():
    # The one place these numbers are authoritative is
    # configs/gr00t/mhbench_keys.py; GauDP re-states them and this pins the
    # restatement to the same widths and the same order.
    assert STATE_SCHEMA == (
        "left_leg",
        "right_leg",
        "waist",
        "left_arm",
        "left_hand",
        "right_arm",
        "right_hand",
    )
    assert ACTION_SCHEMA == (
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "waist",
        "base_height_command",
        "navigate_command",
    )
    assert ROBOT_PROPRIO_DIM == 43 and PROPRIO_DIM == 86
    assert ROBOT_ACTION_DIM == 35 and ACTION_DIM == 70
    assert sum(s.stop - s.start for s in STATE_SLICES.values()) == ROBOT_PROPRIO_DIM
    assert sum(s.stop - s.start for s in ACTION_SLICES.values()) == ROBOT_ACTION_DIM
    # The first 31 action dims are exactly the joints the env writes; height
    # and navigation are the two locomotion commands after them.
    assert ACTION_SLICES["waist"].stop == JOINT_TARGET_DIM
    assert ACTION_SLICES["base_height_command"] == slice(31, 32)
    assert ACTION_SLICES["navigate_command"] == slice(32, 35)


def test_online_proprio_is_the_recorded_joint_state_untouched():
    joints = {
        "robot_a": np.arange(43, dtype=np.float32),
        "robot_b": np.arange(100, 143, dtype=np.float32),
    }
    observation = {
        "mhbench_state": {robot: {"joint_pos": joints[robot]} for robot in ROBOT_NAMES}
    }
    state = np.concatenate([proprio_from_observation(observation, r) for r in ROBOT_NAMES])
    assert state.shape == (PROPRIO_DIM,)
    np.testing.assert_array_equal(state[:43], joints["robot_a"])
    np.testing.assert_array_equal(state[43:], joints["robot_b"])
    # A copy, so a policy cannot write back into the environment's buffer.
    state[0] = -1.0
    assert joints["robot_a"][0] == 0.0


def test_online_proprio_refuses_a_substitute_for_joint_state():
    pose = np.zeros(7, dtype=np.float32)
    observation = {
        "mhbench_state": {
            robot: {"pelvis_pose": pose, "left_eef_pose": pose, "right_eef_pose": pose}
            for robot in ROBOT_NAMES
        }
    }
    with pytest.raises(KeyError, match="joint_pos"):
        proprio_from_observation(observation, "robot_a")
    with pytest.raises(ValueError, match="43D"):
        proprio_from_observation(
            {"mhbench_state": {"robot_a": {"joint_pos": np.zeros(31, np.float32)}}}, "robot_a"
        )


def test_70d_side_channel_splits_into_the_env_action_keys():
    action = np.arange(ACTION_DIM, dtype=np.float32)
    packed = pack_xpolicy_action(action)
    assert set(packed["mhbench_raw_action"]) == set(ROBOT_NAMES)
    for index, robot in enumerate(ROBOT_NAMES):
        entry = packed["mhbench_raw_action"][robot]
        assert set(entry) == {"joint_targets", "height", "base_vel"}
        assert entry["joint_targets"].shape == (JOINT_TARGET_DIM,)
        assert entry["height"].shape == (1,)
        assert entry["base_vel"].shape == (3,)
        base = index * ROBOT_ACTION_DIM
        np.testing.assert_array_equal(entry["joint_targets"], action[base : base + 31])
        np.testing.assert_array_equal(entry["height"], action[base + 31 : base + 32])
        np.testing.assert_array_equal(entry["base_vel"], action[base + 32 : base + 35])
    np.testing.assert_array_equal(flat_action_from_xpolicy(packed), action)


def test_pack_refuses_the_old_44d_wrist_pose_action():
    with pytest.raises(ValueError, match="70D"):
        pack_xpolicy_action(np.zeros(44, dtype=np.float32))
    with pytest.raises(ValueError, match="35D"):
        split_robot_action(np.zeros(22, dtype=np.float32))
