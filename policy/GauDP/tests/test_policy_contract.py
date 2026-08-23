import pytest
import numpy as np

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from torch import nn

from XPolicyLab.policy.GauDP.gaudp.policy import GauDPPolicy
from XPolicyLab.policy.GauDP.model import _require_finite


class TinyGaussian(nn.Module):
    def __init__(self):
        super().__init__()
        self.unused = nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward(self, context, global_step=0, return_features=False):
        self.calls += 1
        image = context["image"]
        features = torch.cat((image, image.repeat(1, 1, 3, 1, 1), image[:, :, :1]), dim=2)
        features = features[:, :, :13]
        return (None, features) if return_features else None


class TinyObservation(nn.Module):
    output_dim = 50
    feature_dim = 4

    def forward(self, images, state):
        pooled = images.mean(dim=(-1, -2)).flatten(1)
        padding = torch.zeros((images.shape[0], self.output_dim - pooled.shape[1] - 42), device=images.device)
        return torch.cat((pooled, padding, state), dim=-1)


def test_one_step_freezes_gaussian_and_returns_six_by_44():
    policy = GauDPPolicy(
        num_views=2,
        num_inference_steps=1,
        down_dims=(32, 64, 128),
        gaussian_encoder=TinyGaussian(),
        observation_encoder=TinyObservation(),
    )
    policy.normalizer.fit(torch.randn(5, 42), torch.randn(5, 44))
    batch = {
        "images": torch.rand(1, 8, 2, 3, 32, 32),
        "state": torch.randn(1, 8, 42),
        "action": torch.randn(1, 8, 44),
        "gaussian_features": torch.randn(1, 3, 2, 13, 32, 32),
    }
    loss, metrics = policy.compute_loss(batch, return_metrics=True)
    loss.backward()
    assert metrics["diffusion/noise_mse"] >= 0
    assert -1.0 <= metrics["diffusion/noise_cosine"] <= 1.0
    assert metrics["action/x0_clipped_mae"] >= 0
    assert metrics["action/robot_a_mse"] >= 0
    assert metrics["action/robot_b_mse"] >= 0
    assert policy.gaussian_encoder.calls == 0
    assert all(parameter.grad is None for parameter in policy.gaussian_encoder.parameters())
    policy.eval()
    output = policy.predict_action(batch["images"][:, :3], batch["state"][:, :3])
    assert policy.gaussian_encoder.calls == 1
    assert output.shape == (1, 6, 44)


def test_eval_rejects_non_finite_values():
    with pytest.raises(ValueError, match="predicted action contains 1"):
        _require_finite("predicted action", np.asarray([0.0, np.inf], dtype=np.float32))
