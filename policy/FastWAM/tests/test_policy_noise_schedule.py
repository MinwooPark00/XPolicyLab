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
            _derive_policy_noise_seed(7, "robot_a", 4, 11),
            _derive_policy_noise_seed(7, "robot_a", 3, 12),
        }
    ) == 5
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


class _FakePolicy:
    def __init__(self):
        self.seed = None
        self.seeds_seen = []

    def _infer_action_chunk(self, observation, instruction):
        self.seeds_seen.append(self.seed)
        return np.zeros((1, 35), dtype=np.float32)


def test_shared_policy_gets_distinct_role_and_replan_seeds():
    model = Model.__new__(Model)
    model.allow_dummy_policy = False
    model._decentralized = True
    model.replan_steps = 1
    model._policy_seed_salt = 0
    model.seed(23)
    shared = _FakePolicy()
    model._policies = {"robot_a": shared, "robot_b": shared}
    model._mhbench_instruction_for = lambda target, wire: target
    model._debug_dump = lambda *args: None
    obs = {
        "robot_a": object(),
        "robot_b": object(),
        "__mhbench_instructions__": {},
    }

    model._mhbench_chunks(obs)
    model._mhbench_chunks(obs)

    assert shared.seeds_seen == [
        _derive_policy_noise_seed(23, "robot_a", 0, 0),
        _derive_policy_noise_seed(23, "robot_b", 0, 0),
        _derive_policy_noise_seed(23, "robot_a", 1, 0),
        _derive_policy_noise_seed(23, "robot_b", 1, 0),
    ]
    assert len(set(shared.seeds_seen)) == 4
