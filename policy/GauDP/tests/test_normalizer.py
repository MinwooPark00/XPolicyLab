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


def test_validation_range_diagnostics_do_not_refit_statistics():
    state = torch.stack((torch.zeros(42), torch.ones(42)))
    action = torch.stack((torch.zeros(44), torch.ones(44)))
    normalizer = GauDPNormalizer()
    normalizer.fit(state, action)
    before = normalizer.state_max.clone()
    metrics = normalizer.range_diagnostics(torch.full((1, 42), 2.0), torch.full((1, 44), 0.5))
    assert metrics["normalization/val_state_out_of_range_fraction"] == 1.0
    assert metrics["normalization/val_action_out_of_range_fraction"] == 0.0
    assert torch.equal(normalizer.state_max, before)


def test_normalizer_rejects_non_finite_statistics():
    state = torch.zeros(2, 42)
    state[0, 0] = float("nan")
    with pytest.raises(ValueError, match="state statistics contain"):
        GauDPNormalizer().fit(state, torch.zeros(2, 44))
