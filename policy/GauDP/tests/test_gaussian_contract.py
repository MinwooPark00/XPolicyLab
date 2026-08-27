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


class TinyNoPoSplatEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.gaussian_param_head = nn.Linear(2, 2)
        self.intrinsic_encoder = nn.Linear(2, 2)


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


def test_full_finetuning_matches_official_noposplat_optimizer_groups():
    encoder = TinyNoPoSplatEncoder()
    module._configure_finetuning(encoder, "full")
    optimizer = module._build_optimizer(
        encoder,
        "full",
        lr=1e-4,
        backbone_lr_multiplier=0.1,
        weight_decay=0.05,
    )

    assert [group["group_name"] for group in optimizer.param_groups] == ["head", "pretrained"]
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([1e-4, 1e-5])
    assert optimizer.defaults["betas"] == (0.9, 0.95)
    assert optimizer.defaults["weight_decay"] == 0.05
    head_ids = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    assert id(encoder.gaussian_param_head.weight) in head_ids
    assert id(encoder.intrinsic_encoder.weight) in head_ids
    assert id(encoder.backbone.weight) not in head_ids


def test_official_scheduler_warms_up_then_cosine_decays_head_lr():
    encoder = TinyNoPoSplatEncoder()
    optimizer = module._build_optimizer(
        encoder,
        "full",
        lr=1e-4,
        backbone_lr_multiplier=0.1,
        weight_decay=0.05,
    )
    scheduler = module._build_lr_scheduler(
        optimizer,
        warm_up_steps=2,
        max_steps=10,
        base_lr=1e-4,
        min_lr_ratio=0.1,
    )

    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([5e-5, 5e-6])
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([1e-4, 1e-5])
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] < 1e-4
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-5)


def test_head_only_optimizer_retains_legacy_defaults():
    encoder = TinyNoPoSplatEncoder()
    module._configure_finetuning(encoder, "heads")
    optimizer = module._build_optimizer(
        encoder,
        "heads",
        lr=1e-5,
        backbone_lr_multiplier=0.1,
        weight_decay=1e-6,
    )

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["group_name"] == "head"
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["weight_decay"] == 1e-6
    assert not encoder.backbone.weight.requires_grad
