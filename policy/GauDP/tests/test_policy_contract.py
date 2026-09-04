import pytest
import numpy as np

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from torch import nn

from XPolicyLab.policy.GauDP.gaudp.policy import (
    GauDPPolicy,
    MultiViewObservationEncoder,
    policy_checkpoint_payload,
)
from XPolicyLab.policy.GauDP.gaudp.schema import (
    ACTION_DIM,
    ACTION_SCHEMA,
    PROPRIO_DIM,
    STATE_SCHEMA,
)
from XPolicyLab.policy.GauDP.model import (
    POLICY_CHECKPOINT_FORMAT,
    _check_checkpoint_contract,
    _clip_enabled,
    _clip_fitted_state,
    _checkpoint_preference,
    _require_finite,
)


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
    output_dim = PROPRIO_DIM + 8
    feature_dim = 4

    def forward(self, images, state):
        pooled = images.mean(dim=(-1, -2)).flatten(1)
        padding = torch.zeros(
            (images.shape[0], self.output_dim - pooled.shape[1] - PROPRIO_DIM),
            device=images.device,
        )
        return torch.cat((pooled, padding, state), dim=-1)


def _policy() -> GauDPPolicy:
    return GauDPPolicy(
        num_views=2,
        num_inference_steps=1,
        down_dims=(32, 64, 128),
        gaussian_encoder=TinyGaussian(),
        observation_encoder=TinyObservation(),
    )


def test_one_step_freezes_gaussian_and_returns_six_by_70():
    policy = _policy()
    assert policy.n_obs_steps == 1
    policy.normalizer.fit(torch.randn(5, PROPRIO_DIM), torch.randn(5, ACTION_DIM))
    batch = {
        "images": torch.rand(1, 1, 2, 3, 32, 32),
        "state": torch.randn(1, 1, PROPRIO_DIM),
        "action": torch.randn(1, 8, ACTION_DIM),
        "gaussian_features": torch.randn(1, 1, 2, 13, 32, 32),
    }
    loss, metrics = policy.compute_loss(batch, return_metrics=True)
    loss.backward()
    assert metrics["diffusion/noise_mse"] >= 0
    assert -1.0 <= metrics["diffusion/noise_cosine"] <= 1.0
    assert metrics["action/x0_clipped_mae"] >= 0
    assert metrics["action/robot_a_mse"] >= 0
    assert metrics["action/robot_b_mse"] >= 0
    # The joint-space diagnostics replace the EEF/hand-compression ones.
    for group in ("arm", "hand", "waist", "height", "navigation"):
        assert metrics[f"action/{group}_mse"] >= 0
    assert policy.gaussian_encoder.calls == 0
    assert all(parameter.grad is None for parameter in policy.gaussian_encoder.parameters())
    policy.eval()
    output = policy.predict_action(batch["images"], batch["state"])
    assert policy.gaussian_encoder.calls == 1
    assert output.shape == (1, 6, ACTION_DIM)


def test_compute_loss_refuses_the_old_44d_action():
    policy = _policy()
    policy.normalizer.fit(torch.randn(5, PROPRIO_DIM), torch.randn(5, ACTION_DIM))
    batch = {
        "images": torch.rand(1, 1, 2, 3, 32, 32),
        "state": torch.randn(1, 1, PROPRIO_DIM),
        "action": torch.randn(1, 8, 44),
        "gaussian_features": torch.randn(1, 1, 2, 13, 32, 32),
    }
    with pytest.raises(ValueError, match=f"expected {ACTION_DIM}D action"):
        policy.compute_loss(batch)


def test_checkpoint_round_trips_the_joint_contract_and_rejects_v1(tmp_path):
    policy = _policy()
    payload = policy_checkpoint_payload(policy, epoch=0)
    assert payload["format"] == POLICY_CHECKPOINT_FORMAT
    assert payload["state_dim"] == PROPRIO_DIM and payload["action_dim"] == ACTION_DIM
    assert payload["state_schema"] == STATE_SCHEMA
    assert payload["action_schema"] == ACTION_SCHEMA
    # The frozen reconstruction encoder stays in its own checkpoint.
    assert not any(key.startswith("gaussian_encoder.") for key in payload["state_dict"])

    path = tmp_path / "best.ckpt"
    torch.save(payload, path)
    _check_checkpoint_contract(torch.load(path, map_location="cpu", weights_only=False), path)

    with pytest.raises(ValueError, match="retrained in joint space"):
        _check_checkpoint_contract({"format": "mhbench-gaudp-policy-v1"}, path)
    with pytest.raises(ValueError, match="86D / 70D"):
        _check_checkpoint_contract({**payload, "state_dim": 42, "action_dim": 44}, path)
    with pytest.raises(ValueError, match="centralized GR00T ordering"):
        _check_checkpoint_contract({**payload, "action_schema": ACTION_SCHEMA[::-1]}, path)


def test_eval_rejects_non_finite_values():
    with pytest.raises(ValueError, match="predicted action contains 1"):
        _require_finite("predicted action", np.asarray([0.0, np.inf], dtype=np.float32))


def test_eval_checkpoint_num_overrides_deploy_preference():
    assert _checkpoint_preference({"checkpoint": "last"}) == "last"
    assert _checkpoint_preference({"checkpoint": "last", "checkpoint_num": "best"}) == "best"
    with pytest.raises(ValueError, match="must be 'best' or 'last'"):
        _checkpoint_preference({"checkpoint_num": "100"})


def test_eval_clamps_joint_state_to_checkpoint_fitted_range():
    low = np.full(PROPRIO_DIM, -0.25, dtype=np.float32)
    high = np.full(PROPRIO_DIM, 0.5, dtype=np.float32)
    state = np.linspace(-1.0, 1.0, PROPRIO_DIM, dtype=np.float32)
    clipped = _clip_fitted_state(state, low, high)
    np.testing.assert_array_equal(clipped, np.clip(state, low, high))
    assert clipped.dtype == np.float32

    assert _clip_enabled("fitted", "obs_clip")
    assert not _clip_enabled(None, "obs_clip")
    with pytest.raises(ValueError, match="obs_clip must be"):
        _clip_enabled("training-range-ish", "obs_clip")


def _encoder(**kwargs) -> MultiViewObservationEncoder:
    return MultiViewObservationEncoder(num_views=2, feature_dim=8, **kwargs)


def test_crop_is_random_while_training_and_centred_at_eval():
    encoder = _encoder(crop_shape=(216, 288))
    images = torch.rand(1, 2, 3, 240, 320)
    state = torch.zeros(1, PROPRIO_DIM)

    encoder.train()
    torch.manual_seed(0)
    first = encoder(images, state)
    torch.manual_seed(1)
    second = encoder(images, state)
    assert not torch.allclose(first, second), "training crops must vary between calls"

    encoder.eval()
    with torch.no_grad():
        assert torch.allclose(encoder(images, state), encoder(images, state)), (
            "the served crop must be deterministic"
        )


def test_encoder_defaults_reproduce_the_pre_crop_network():
    """Old checkpoints carry no vision keys; the defaults must be what wrote them."""
    encoder = _encoder()
    assert encoder.crop is None
    assert encoder.image_norm == "symmetric"
    pixels = torch.rand(2, 3, 8, 8)
    assert torch.allclose(encoder._normalize(pixels), (pixels - 0.5) / 0.5)


def test_imagenet_norm_matches_torchvision():
    torchvision = pytest.importorskip("torchvision")
    encoder = _encoder(image_norm="imagenet")
    pixels = torch.rand(2, 3, 8, 8)
    expected = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )(pixels)
    assert torch.allclose(encoder._normalize(pixels), expected, atol=1e-6)


def test_group_norm_divisor_follows_upstream_when_set():
    from torch import nn as _nn

    upstream = _encoder(group_norm_divisor=16)
    legacy = _encoder()
    first = lambda enc: next(  # noqa: E731
        m for m in enc.backbone.modules() if isinstance(m, _nn.GroupNorm)
    )
    assert first(upstream).num_groups == 64 // 16
    assert first(legacy).num_groups == 32


def test_crop_shape_must_fit_inside_the_frame():
    with pytest.raises(ValueError):
        _encoder(crop_shape=(240, 320))


def test_vision_settings_round_trip_through_the_checkpoint():
    policy = GauDPPolicy(
        num_views=2,
        num_inference_steps=1,
        down_dims=(32, 64, 128),
        crop_shape=(216, 288),
        image_norm="imagenet",
        group_norm_divisor=16,
        gaussian_encoder=TinyGaussian(),
        observation_encoder=TinyObservation(),
    )
    config = policy_checkpoint_payload(policy)["config"]
    assert config["crop_shape"] == (216, 288)
    assert config["image_norm"] == "imagenet"
    assert config["group_norm_divisor"] == 16
