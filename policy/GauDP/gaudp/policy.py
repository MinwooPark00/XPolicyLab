"""Standalone centralized GauDP policy for MHBench."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .core.model.diffusion.conditional_unet1d import ConditionalUnet1D
from .core.model.gaussian_cnn import GaussianConvEncoder
from .core.model.vision.crop_randomizer import CropRandomizer
from .gaussian import build_gaussian_encoder, encode_gaussians, freeze_gaussian_encoder
from .normalizer import GauDPNormalizer
from .schema import ACTION_DIM, ACTION_SCHEMA, PROPRIO_DIM, ROBOT_ACTION_DIM, STATE_SCHEMA

IMAGE_SIZE = (240, 320)
"""The converted/served frame size. `dataset.py` and `model.py` resize to it."""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Every vision option below defaults to what the already-trained MHBench
# checkpoints were produced with, so a checkpoint whose recorded `config` predates
# them still deserializes into the network that wrote it. `train_policy.py`'s
# argparse defaults carry the *current* recipe; a new run records its choices in
# the checkpoint and `model.py` serves them back.
LEGACY_IMAGE_NORM = "symmetric"
LEGACY_GROUP_NORM_DIVISOR = None
LEGACY_CROP_SHAPE = None


def _replace_batch_norm(module: nn.Module, divisor: int | None = LEGACY_GROUP_NORM_DIVISOR) -> nn.Module:
    """BatchNorm2d -> GroupNorm, as every Diffusion-Policy vision stack does.

    `divisor` is upstream Policy-Lightning's `num_features // 16`
    (`MultiImageObsEncoder(use_group_norm=True)`, which DP's `robot_dp.yaml` also
    sets). `None` keeps this port's original `min(32, num_features)` -- the same
    parameter shapes but a much finer grouping: 32 groups of 2 channels at
    ResNet-18's first block where upstream uses 4 groups of 16.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            if divisor is None:
                groups = min(32, child.num_features)
            else:
                groups = max(1, child.num_features // divisor)
            while child.num_features % groups:
                groups -= 1
            setattr(module, name, nn.GroupNorm(groups, child.num_features))
        else:
            _replace_batch_norm(child, divisor)
    return module


def _normalized_crop_shape(crop_shape) -> tuple[int, int] | None:
    if crop_shape is None:
        return None
    height, width = (int(value) for value in crop_shape)
    if not 0 < height < IMAGE_SIZE[0] or not 0 < width < IMAGE_SIZE[1]:
        raise ValueError(
            f"crop_shape {(height, width)} must be strictly inside the {IMAGE_SIZE} frame"
        )
    return (height, width)


class MultiViewObservationEncoder(nn.Module):
    """Shared ResNet-18 over each fused view plus the pair's 86D joint state.

    The fused view is `GaussianConvEncoder`'s 3-channel output, so this stands in
    for upstream Policy-Lightning's `MultiImageObsEncoder`: the same
    resize -> random-crop -> normalize pipeline that encoder applies to each
    `head_cam_i` key before its ResNet.
    """

    def __init__(
        self,
        num_views: int,
        feature_dim: int = 512,
        *,
        crop_shape: tuple[int, int] | None = LEGACY_CROP_SHAPE,
        image_norm: str = LEGACY_IMAGE_NORM,
        group_norm_divisor: int | None = LEGACY_GROUP_NORM_DIVISOR,
    ) -> None:
        super().__init__()
        try:
            from torchvision.models import resnet18
        except ImportError as error:
            raise ImportError("torchvision is required for GauDP's observation encoder") from error
        if image_norm not in ("symmetric", "imagenet"):
            raise ValueError(f"image_norm must be 'symmetric' or 'imagenet', got {image_norm!r}")
        backbone = _replace_batch_norm(resnet18(weights=None), group_norm_divisor)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Identity() if feature_dim == in_features else nn.Linear(in_features, feature_dim)
        self.num_views = int(num_views)
        self.feature_dim = int(feature_dim)
        self.image_norm = str(image_norm)
        self.crop_shape = _normalized_crop_shape(crop_shape)
        self.crop = (
            None
            if self.crop_shape is None
            else CropRandomizer(
                input_shape=(3, *IMAGE_SIZE),
                crop_height=self.crop_shape[0],
                crop_width=self.crop_shape[1],
                num_crops=1,
                pos_enc=False,
            )
        )
        # Non-persistent: these are constants, and adding them to the state dict
        # would make every checkpoint written before this commit report a missing
        # key that `model.py` refuses to load through.
        self.register_buffer("_norm_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_norm_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @property
    def output_dim(self) -> int:
        return self.num_views * self.feature_dim + PROPRIO_DIM

    def _normalize(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.image_norm == "imagenet":
            return (pixels - self._norm_mean.to(pixels.dtype)) / self._norm_std.to(pixels.dtype)
        return (pixels - 0.5) / 0.5

    def forward(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[1] != self.num_views:
            raise ValueError(f"expected [B,{self.num_views},3,H,W], got {tuple(images.shape)}")
        batch, views = images.shape[:2]
        pixels = images.reshape(batch * views, *images.shape[2:])
        if self.crop is not None:
            if tuple(pixels.shape[-2:]) != IMAGE_SIZE:
                raise ValueError(
                    f"crop_shape {self.crop_shape} was fitted to {IMAGE_SIZE} frames, "
                    f"but this batch is {tuple(pixels.shape[-2:])}"
                )
            # CropRandomizer.forward branches on self.training: a random offset per
            # (sample, view) while training, the centre crop at eval -- so the
            # served frame is deterministic and `model.py`'s .eval() is what picks it.
            pixels = self.crop(pixels)
        pixels = self._normalize(pixels)
        visual = self.projection(self.backbone(pixels)).reshape(batch, views * self.feature_dim)
        return torch.cat((visual, state), dim=-1)


class GauDPPolicy(nn.Module):
    """Frozen NoPoSplat context encoder + trainable fusion/vision/DDPM policy."""

    def __init__(
        self,
        *,
        num_views: int = 2,
        horizon: int = 8,
        n_obs_steps: int = 1,
        n_action_steps: int = 6,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 100,
        obs_feature_dim: int = 512,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        crop_shape: tuple[int, int] | None = LEGACY_CROP_SHAPE,
        image_norm: str = LEGACY_IMAGE_NORM,
        group_norm_divisor: int | None = LEGACY_GROUP_NORM_DIVISOR,
        gaussian_encoder: nn.Module | None = None,
        observation_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        try:
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        except ImportError as error:
            raise ImportError(
                f"diffusers is required for GauDP; its import failed with: {error}. "
                "Run through train.sh or set PYTHONNOUSERSITE=1 so user-site packages "
                "cannot shadow the GauDP environment."
            ) from error
        self.num_views = int(num_views)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.num_inference_steps = int(num_inference_steps)
        self.down_dims = tuple(int(value) for value in down_dims)
        if not 1 <= self.n_obs_steps <= self.horizon:
            raise ValueError("n_obs_steps must be between 1 and horizon")
        if self.n_obs_steps - 1 + self.n_action_steps > self.horizon:
            raise ValueError("requested action chunk does not fit inside the diffusion horizon")

        self.gaussian_encoder = freeze_gaussian_encoder(
            build_gaussian_encoder(self.num_views) if gaussian_encoder is None else gaussian_encoder
        )
        self.gaussian_fusion = GaussianConvEncoder(in_channels=13, pre_fuse=True)
        self.crop_shape = _normalized_crop_shape(crop_shape)
        self.image_norm = str(image_norm)
        self.group_norm_divisor = None if group_norm_divisor is None else int(group_norm_divisor)
        self.obs_encoder = (
            MultiViewObservationEncoder(
                self.num_views,
                obs_feature_dim,
                crop_shape=self.crop_shape,
                image_norm=self.image_norm,
                group_norm_divisor=self.group_norm_divisor,
            )
            if observation_encoder is None
            else observation_encoder
        )
        self.normalizer = GauDPNormalizer()
        self.diffusion = ConditionalUnet1D(
            input_dim=ACTION_DIM,
            global_cond_dim=self.obs_encoder.output_dim * self.n_obs_steps,
            diffusion_step_embed_dim=128,
            down_dims=self.down_dims,
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=1e-4,
            beta_end=2e-2,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Parent .train() must never unfreeze/change the reconstruction encoder.
        freeze_gaussian_encoder(self.gaussian_encoder)
        return self

    def config(self) -> dict[str, Any]:
        return {
            "num_views": self.num_views,
            "horizon": self.horizon,
            "n_obs_steps": self.n_obs_steps,
            "n_action_steps": self.n_action_steps,
            "num_train_timesteps": self.noise_scheduler.config.num_train_timesteps,
            "num_inference_steps": self.num_inference_steps,
            "obs_feature_dim": self.obs_encoder.feature_dim,
            "down_dims": self.down_dims,
            # Vision settings travel with the checkpoint so evaluation rebuilds the
            # network that was trained. A checkpoint written before they existed
            # simply has none of these keys, and the constructor's legacy defaults
            # then reproduce it exactly.
            "crop_shape": self.crop_shape,
            "image_norm": self.image_norm,
            "group_norm_divisor": self.group_norm_divisor,
        }

    def _global_condition(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        gaussian_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        images = images[:, : self.n_obs_steps]
        state = self.normalizer.normalize_state(state[:, : self.n_obs_steps])
        batch, steps, views, channels, height, width = images.shape
        if gaussian_features is None:
            # Deployment receives new observations, so NoPoSplat remains an
            # online fixed feature extractor for inference.
            flattened = images.reshape(batch * steps, views, channels, height, width)
            with torch.no_grad():
                _, gaussian_features = encode_gaussians(
                    self.gaussian_encoder, flattened, return_features=True
                )
            gaussian_features = gaussian_features.reshape(batch, steps, views, 13, height, width)
        else:
            gaussian_features = gaussian_features[:, : self.n_obs_steps]
            expected = (batch, steps, views, 13, height, width)
            if tuple(gaussian_features.shape) != expected:
                raise ValueError(
                    f"expected cached Gaussian features {expected}, got {tuple(gaussian_features.shape)}"
                )
            gaussian_features = gaussian_features.to(dtype=images.dtype)
        fused = self.gaussian_fusion(gaussian_features, images)
        fused = fused.reshape(batch * steps, views, 3, height, width)
        encoded = self.obs_encoder(fused, state.reshape(batch * steps, PROPRIO_DIM))
        return encoded.reshape(batch, steps * self.obs_encoder.output_dim)

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_metrics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        images, state, action = batch["images"], batch["state"], batch["action"]
        if action.shape[-1] != ACTION_DIM:
            raise ValueError(f"expected {ACTION_DIM}D action, got {action.shape[-1]}")
        normalized_action = self.normalizer.normalize_action(action)
        noise = torch.randn_like(normalized_action)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (action.shape[0],),
            device=action.device,
        ).long()
        noisy = self.noise_scheduler.add_noise(normalized_action, noise, timesteps)
        prediction = self.diffusion(
            noisy,
            timesteps,
            global_cond=self._global_condition(images, state, batch.get("gaussian_features")),
        )
        loss = F.mse_loss(prediction, noise)
        if not return_metrics:
            return loss

        # DDPM predicts epsilon. Convert that prediction to a clean normalized
        # action estimate for diagnostics only; clipping follows the scheduler's
        # configured action range and avoids low-SNR timesteps dominating x0.
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=timesteps.device, dtype=normalized_action.dtype
        )
        alpha = alphas_cumprod[timesteps]
        alpha = alpha.reshape((-1,) + (1,) * (normalized_action.ndim - 1))
        predicted_x0 = (
            noisy - (1.0 - alpha).sqrt() * prediction
        ) / alpha.sqrt().clamp_min(1e-6)
        predicted_x0 = predicted_x0.clamp(-1.0, 1.0)
        squared_error = (predicted_x0 - normalized_action).square()
        absolute_error = (predicted_x0 - normalized_action).abs()

        def group_mse(indices: list[int]) -> float:
            return float(squared_error[..., indices].mean().detach())

        robot_a = list(range(0, ROBOT_ACTION_DIM))
        robot_b = list(range(ROBOT_ACTION_DIM, ACTION_DIM))
        def both(local: range | list[int]) -> list[int]:
            return [index + offset for offset in (0, ROBOT_ACTION_DIM) for index in local]

        arm = both(range(0, 14))
        hand = both(range(14, 28))
        waist = both(range(28, 31))
        height = both([31])
        navigation = both(range(32, 35))
        snr = alpha / (1.0 - alpha).clamp_min(1e-6)
        noise_cosine = F.cosine_similarity(
            prediction.flatten(1), noise.flatten(1), dim=1
        ).mean()
        metrics = {
            "diffusion/noise_mse": float(loss.detach()),
            "diffusion/noise_cosine": float(noise_cosine.detach()),
            "diffusion/pred_noise_rms": float(prediction.square().mean().sqrt().detach()),
            "diffusion/target_noise_rms": float(noise.square().mean().sqrt().detach()),
            "diffusion/timestep_mean": float(timesteps.float().mean().detach()),
            "diffusion/snr_mean": float(snr.mean().detach()),
            "action/x0_clipped_mse": float(squared_error.mean().detach()),
            "action/x0_clipped_mae": float(absolute_error.mean().detach()),
            "action/robot_a_mse": group_mse(robot_a),
            "action/robot_b_mse": group_mse(robot_b),
            "action/arm_mse": group_mse(arm),
            "action/hand_mse": group_mse(hand),
            "action/waist_mse": group_mse(waist),
            "action/height_mse": group_mse(height),
            "action/navigation_mse": group_mse(navigation),
        }
        return loss, metrics

    @torch.no_grad()
    def predict_action(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        global_condition = self._global_condition(images, state)
        trajectory = torch.randn(
            (batch, self.horizon, ACTION_DIM), device=images.device, dtype=images.dtype
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps, device=images.device)
        for timestep in self.noise_scheduler.timesteps:
            residual = self.diffusion(trajectory, timestep, global_cond=global_condition)
            trajectory = self.noise_scheduler.step(residual, timestep, trajectory).prev_sample
        action = self.normalizer.unnormalize_action(trajectory)
        start = self.n_obs_steps - 1
        return action[:, start : start + self.n_action_steps]


def policy_checkpoint_payload(policy: GauDPPolicy, **metadata: Any) -> dict[str, Any]:
    # Reconstruction weights stay in their Gaussian checkpoint (run-local or
    # official/external) and are not duplicated in every policy checkpoint.
    state = {
        key: value
        for key, value in policy.state_dict().items()
        if not key.startswith("gaussian_encoder.")
    }
    return {
        "format": "mhbench-gaudp-policy-v2",
        "state_dim": PROPRIO_DIM,
        "action_dim": ACTION_DIM,
        "state_schema": STATE_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "config": policy.config(),
        "state_dict": state,
        **metadata,
    }
