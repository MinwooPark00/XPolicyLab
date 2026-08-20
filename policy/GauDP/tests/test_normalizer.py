import pytest

torch = pytest.importorskip("torch")

from XPolicyLab.policy.GauDP.gaudp.normalizer import GauDPNormalizer


def test_state_and_action_statistics_are_independent():
    state = torch.stack((torch.zeros(42), torch.full((42,), 2.0)))
    action = torch.stack((torch.full((44,), 10.0), torch.full((44,), 14.0)))
    normalizer = GauDPNormalizer()
    normalizer.fit(state, action)
    assert normalizer.state_min.shape == (42,)
    assert normalizer.action_min.shape == (44,)
    assert torch.allclose(normalizer.normalize_state(state[0]), torch.full((42,), -1.0))
    assert torch.allclose(normalizer.normalize_action(action[0]), torch.full((44,), -1.0))
    assert torch.allclose(normalizer.unnormalize_action(normalizer.normalize_action(action)), action)
