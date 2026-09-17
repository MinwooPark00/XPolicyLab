import numpy as np

from XPolicyLab.policy.FastWAM.model import Model, _derive_policy_noise_seed


def test_noise_seed_is_stable_and_keyed_by_every_dimension():
    base = _derive_policy_noise_seed(7, "robot_a", 3, 11)
    # Pin the named schedule: this value must also agree across machines.
    assert base == 2404644635935907912
    assert base == _derive_policy_noise_seed(7, "robot_a", 3, 11)
    assert len(
        {
            base,
            _derive_policy_noise_seed(8, "robot_a", 3, 11),
            _derive_policy_noise_seed(7, "robot_b", 3, 11),
            _derive_policy_noise_seed(7, "robot_c", 3, 11),
            _derive_policy_noise_seed(7, "robot_a", 4, 11),
            _derive_policy_noise_seed(7, "robot_a", 3, 12),
        }
    ) == 6
    assert 0 <= base < 2**63


def test_episode_reseed_replays_sequence_and_salt_changes_it():
    model = Model.__new__(Model)
    model._policy_seed_salt = 5
    model.seed(17)
    first = [model._next_policy_noise_seed("robot_a") for _ in range(3)]
    robot_b = model._next_policy_noise_seed("robot_b")

    model.seed(17)
    assert first == [model._next_policy_noise_seed("robot_a") for _ in range(3)]
    assert robot_b == model._next_policy_noise_seed("robot_b")

    model._policy_seed_salt = 6
    model.seed(17)
    assert first[0] != model._next_policy_noise_seed("robot_a")


def test_three_agent_observation_and_dummy_actions_include_robot_c():
    model = Model.__new__(Model)
    model.allow_dummy_policy = False
    model._decentralized = True
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    obs = {
        "vision": {
            "cam_left_wrist": {"color": image},
            "cam_right_wrist": {"color": image},
            "cam_third_view": {"color": image},
        },
        "mhbench_state": {
            robot: {
                "joint_pos": np.full(43, index, dtype=np.float32),
                "pelvis_pose": np.zeros(7, dtype=np.float32),
            }
            for index, robot in enumerate(("robot_a", "robot_b", "robot_c"))
        },
        "mhbench_instruction": {
            "robot_a": "role a",
            "robot_b": "role b",
            "robot_c": "role c",
        },
    }

    encoded = model._encode_mhbench(obs)
    assert set(encoded) == {
        "robot_a", "robot_b", "robot_c", "__mhbench_instructions__"
    }
    assert encoded["robot_c"]["joint_action"]["vector"].shape == (43,)
    assert encoded["robot_c"]["joint_action"]["vector"].flags.writeable
    assert encoded["robot_c"]["images"]["ego"].shape == (240, 320, 3)

    model.allow_dummy_policy = True
    model.replan_steps = 2
    actions = model._mhbench_chunks(encoded)
    assert len(actions) == 2
    assert all(
        set(step["mhbench_raw_action"]) == {"robot_a", "robot_b", "robot_c"}
        for step in actions
    )


class _FakePolicy:
    def __init__(self):
        self.seed = None
        self.seeds_seen = []

    def _infer_action_chunk(self, observation, instruction):
        self.seeds_seen.append(self.seed)
        return np.zeros((1, 35), dtype=np.float32)


def test_three_agent_shared_policy_gets_distinct_role_and_replan_seeds():
    model = Model.__new__(Model)
    model.allow_dummy_policy = False
    model._decentralized = True
    model.replan_steps = 1
    model._policy_seed_salt = 0
    model.seed(23)
    shared = _FakePolicy()
    model._policies = {"robot_a": shared, "robot_b": shared, "robot_c": shared}
    model._mhbench_instruction_for = lambda target, wire: target
    model._debug_dump = lambda *args: None
    obs = {
        "robot_a": object(),
        "robot_b": object(),
        "robot_c": object(),
        "__mhbench_instructions__": {},
    }

    first_actions = model._mhbench_chunks(obs)
    second_actions = model._mhbench_chunks(obs)

    assert set(first_actions[0]["mhbench_raw_action"]) == {"robot_a", "robot_b", "robot_c"}
    assert set(second_actions[0]["mhbench_raw_action"]) == {"robot_a", "robot_b", "robot_c"}

    assert shared.seeds_seen == [
        _derive_policy_noise_seed(23, "robot_a", 0, 0),
        _derive_policy_noise_seed(23, "robot_b", 0, 0),
        _derive_policy_noise_seed(23, "robot_c", 0, 0),
        _derive_policy_noise_seed(23, "robot_a", 1, 0),
        _derive_policy_noise_seed(23, "robot_b", 1, 0),
        _derive_policy_noise_seed(23, "robot_c", 1, 0),
    ]
    assert len(set(shared.seeds_seen)) == 6
