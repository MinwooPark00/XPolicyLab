import pytest

torch = pytest.importorskip("torch")

from XPolicyLab.policy.GauDP.gaudp.normalizer import GauDPNormalizer
from XPolicyLab.policy.GauDP.gaudp.schema import ACTION_DIM, PROPRIO_DIM


def test_state_and_action_statistics_are_independent():
    state = torch.stack((torch.zeros(PROPRIO_DIM), torch.full((PROPRIO_DIM,), 2.0)))
    action = torch.stack((torch.full((ACTION_DIM,), 10.0), torch.full((ACTION_DIM,), 14.0)))
    normalizer = GauDPNormalizer()
    normalizer.fit(state, action)
    assert normalizer.state_min.shape == (PROPRIO_DIM,)
    assert normalizer.action_min.shape == (ACTION_DIM,)
    assert torch.allclose(normalizer.normalize_state(state[0]), torch.full((PROPRIO_DIM,), -1.0))
    assert torch.allclose(normalizer.normalize_action(action[0]), torch.full((ACTION_DIM,), -1.0))
    assert torch.allclose(normalizer.unnormalize_action(normalizer.normalize_action(action)), action)


def test_validation_range_diagnostics_do_not_refit_statistics():
    state = torch.stack((torch.zeros(PROPRIO_DIM), torch.ones(PROPRIO_DIM)))
    action = torch.stack((torch.zeros(ACTION_DIM), torch.ones(ACTION_DIM)))
    normalizer = GauDPNormalizer()
    normalizer.fit(state, action)
    before = normalizer.state_max.clone()
    metrics = normalizer.range_diagnostics(torch.full((1, PROPRIO_DIM), 2.0), torch.full((1, ACTION_DIM), 0.5))
    assert metrics["normalization/val_state_out_of_range_fraction"] == 1.0
    assert metrics["normalization/val_action_out_of_range_fraction"] == 0.0
    assert torch.equal(normalizer.state_max, before)


def test_normalizer_rejects_non_finite_statistics():
    state = torch.zeros(2, PROPRIO_DIM)
    state[0, 0] = float("nan")
    with pytest.raises(ValueError, match="state statistics contain"):
        GauDPNormalizer().fit(state, torch.zeros(2, ACTION_DIM))
