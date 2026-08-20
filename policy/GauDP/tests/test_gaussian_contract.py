import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn


SCRIPT = Path(__file__).resolve().parents[1] / "train_gaussian.py"
spec = importlib.util.spec_from_file_location("gaudp_train_gaussian", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TinyReconstructionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.5))

    def forward(self, context, global_step=0, return_features=False):
        return context["image"].mul(self.gain)


def test_rgb_and_masked_depth_reconstruction_one_step(monkeypatch):
    encoder = TinyReconstructionEncoder()

    def fake_render(gaussians, batch, image_shape):
        rgb = gaussians.add(1).div(2)
        depth = torch.ones_like(batch["depth"]) * encoder.gain
        return rgb, depth

    monkeypatch.setattr(module, "_render", fake_render)
    images = torch.rand(1, 2, 3, 16, 16)
    batch = {
        "images": images,
        "depth": torch.ones(1, 2, 16, 16),
        "intrinsics": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1),
        "extrinsics": torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1),
        "near": torch.full((1, 2), 0.1),
        "far": torch.full((1, 2), 10.0),
    }
    loss, metrics = module.reconstruction_loss(encoder, batch, global_step=0, depth_weight=0.1)
    loss.backward()
    assert encoder.gain.grad is not None
    assert metrics["rgb"] >= 0 and metrics["depth"] >= 0
