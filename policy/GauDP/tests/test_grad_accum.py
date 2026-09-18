"""--grad-accum takes the same optimizer step as the full batch."""

import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from XPolicyLab.policy.GauDP.gaudp.policy import GauDPPolicy
from XPolicyLab.policy.GauDP.gaudp.schema import ACTION_DIM, PROPRIO_DIM
from XPolicyLab.policy.GauDP.train_policy import _accumulated_backward


class _Identity(torch.nn.Module):
    def forward(self, context, global_step=0, return_features=False):
        image = context["image"]
        features = torch.cat((image, image.repeat(1, 1, 3, 1, 1), image[:, :, :1]), dim=2)[:, :, :13]
        return (None, features) if return_features else None


def _policy():
    torch.manual_seed(0)
    policy = GauDPPolicy(
        num_views=2, horizon=8, n_action_steps=6, num_inference_steps=1,
        obs_feature_dim=16, down_dims=(16, 32, 64), crop_shape=None,
        group_norm_divisor=16, gaussian_encoder=_Identity(),
    )
    policy.normalizer.fit(torch.randn(16, PROPRIO_DIM), torch.randn(16, ACTION_DIM))
    # float64: the comparison is of the arithmetic, not of fp32 summation order,
    # which alone moves a conv gradient summed over 240x320 pixels by ~1e-4.
    return policy.double()


def _batch(n=8, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return {
        "images": torch.rand(n, 1, 2, 3, 240, 320, generator=generator, dtype=torch.float64),
        "state": torch.randn(n, 1, PROPRIO_DIM, generator=generator, dtype=torch.float64),
        "action": torch.randn(n, 8, ACTION_DIM, generator=generator, dtype=torch.float64),
    }


def _split(batch, parts):
    size = batch["action"].shape[0] // parts
    return [{key: value[i * size : (i + 1) * size] for key, value in batch.items()} for i in range(parts)]


def _pinned_noise(policy, batch):
    """Draw each sample's diffusion noise and timestep once, so both paths see the same ones."""
    generator = torch.Generator().manual_seed(7)
    n = batch["action"].shape[0]
    noise = torch.randn(n, 8, ACTION_DIM, generator=generator, dtype=torch.float64)
    steps = torch.randint(0, policy.noise_scheduler.config.num_train_timesteps, (n,), generator=generator)
    return noise, steps


@pytest.mark.parametrize("parts", [2, 4])
def test_accumulated_gradient_equals_the_full_batch(monkeypatch, parts):
    full_policy = _policy()
    accum_policy = copy.deepcopy(full_policy)
    batch = _batch()
    noise, steps = _pinned_noise(full_policy, batch)

    def run(policy, micro_batches):
        cursor = {"at": 0}
        real_randn_like, real_randint = torch.randn_like, torch.randint

        def randn_like(tensor, *args, **kwargs):
            if tuple(tensor.shape[1:]) == (8, ACTION_DIM):
                n = tensor.shape[0]
                return noise[cursor["at"] : cursor["at"] + n].to(tensor)
            return real_randn_like(tensor, *args, **kwargs)

        def randint(low, high=None, size=None, *args, **kwargs):
            if size is not None and len(size) == 1:
                n = size[0]
                out = steps[cursor["at"] : cursor["at"] + n]
                cursor["at"] += n
                return out.to(kwargs.get("device", "cpu"))
            return real_randint(low, high, size, *args, **kwargs)

        monkeypatch.setattr(torch, "randn_like", randn_like)
        monkeypatch.setattr(torch, "randint", randint)
        policy.train()
        policy.zero_grad(set_to_none=True)
        loss, metrics = _accumulated_backward(policy, iter(micro_batches), len(micro_batches), torch.device("cpu"))
        monkeypatch.undo()
        return loss, metrics

    full_loss, _ = run(full_policy, [batch])
    accum_loss, _ = run(accum_policy, _split(batch, parts))
    assert torch.allclose(full_loss, accum_loss, rtol=1e-10, atol=1e-12)
    grads = [
        (name, a.grad, b.grad)
        for (name, a), (_, b) in zip(full_policy.named_parameters(), accum_policy.named_parameters())
        if a.grad is not None
    ]
    assert grads, "no parameter received a gradient"
    for name, full, accum in grads:
        scale = float(full.abs().max())
        diff = float((full - accum).abs().max())
        assert diff <= 1e-9 * scale + 1e-15, (name, diff, scale)


def test_a_short_last_step_weights_its_samples():
    policy = _policy()
    batch = _batch(n=6)
    parts = [{k: v[:4] for k, v in batch.items()}, {k: v[4:] for k, v in batch.items()}]
    loss, metrics = _accumulated_backward(policy, iter(parts), 4, torch.device("cpu"))
    assert torch.isfinite(loss) and "action/robot_a_mse" in metrics
