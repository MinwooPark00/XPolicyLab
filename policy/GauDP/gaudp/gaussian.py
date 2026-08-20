"""Construction and checkpoint utilities for the vendored NoPoSplat encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def build_gaussian_encoder(num_views: int = 2) -> nn.Module:
    """Build the official NoPoSplat ViT-L encoder from local vendored sources."""
    from .third_party.noposplat.model.encoder.backbone.backbone_croco import BackboneCrocoCfg
    from .third_party.noposplat.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
    from .third_party.noposplat.model.encoder.encoder_noposplat import (
        EncoderNoPoSplat,
        EncoderNoPoSplatCfg,
        OpacityMappingCfg,
    )
    from .third_party.noposplat.model.encoder.encoder_noposplat_multi import EncoderNoPoSplatMulti
    from .third_party.noposplat.model.encoder.visualization.encoder_visualizer_epipolar_cfg import (
        EncoderVisualizerEpipolarCfg,
    )

    if num_views < 2:
        raise ValueError("NoPoSplat requires at least two context views")
    multi = num_views > 2
    cfg = EncoderNoPoSplatCfg(
        name="noposplat_multi" if multi else "noposplat",
        d_feature=128,
        num_monocular_samples=32,
        backbone=BackboneCrocoCfg(
            name="croco_multi" if multi else "croco",
            model="ViTLarge_BaseDecoder",
            patch_embed_cls="PatchEmbedDust3R",
            asymmetry_decoder=True,
            # Policy inference deliberately has no camera calibration input.
            # Intrinsics/extrinsics are used by the renderer only in stage 1.
            intrinsics_embed_loc="none",
            intrinsics_embed_degree=0,
            intrinsics_embed_type="token",
        ),
        visualizer=EncoderVisualizerEpipolarCfg(num_samples=8, min_resolution=256, export_ply=False),
        gaussian_adapter=GaussianAdapterCfg(gaussian_scale_min=0.5, gaussian_scale_max=15.0, sh_degree=4),
        apply_bounds_shim=True,
        opacity_mapping=OpacityMappingCfg(initial=0.0, final=0.0, warm_up=1),
        gaussians_per_pixel=1,
        num_surfaces=1,
        gs_params_head_type="dpt_gs",
        pose_free=True,
    )
    return EncoderNoPoSplatMulti(cfg) if multi else EncoderNoPoSplat(cfg)


def _extract_encoder_state(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError("checkpoint must contain a dictionary")
    if "encoder_state" in payload:
        return payload["encoder_state"]
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError("checkpoint state_dict must be a dictionary")
    prefixes = ("module.encoder.", "gaussian_encoder.", "encoder.")
    extracted: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        stripped = key
        for prefix in prefixes:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                break
        extracted[stripped] = value
    return extracted


def load_gaussian_checkpoint(
    encoder: nn.Module,
    checkpoint: str | Path,
    *,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"NoPoSplat checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        # DUSt3R/MASt3R-style public initialization checkpoint supported by
        # NoPoSplat's own training entry point.
        from .third_party.noposplat.misc.weight_modify import checkpoint_filter_fn

        state = checkpoint_filter_fn(payload["model"], encoder)
    else:
        state = _extract_encoder_state(payload)
    if not state:
        raise ValueError(f"no encoder tensors were found in checkpoint: {path}")
    incompatible = encoder.load_state_dict(state, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def freeze_gaussian_encoder(encoder: nn.Module) -> nn.Module:
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def encode_gaussians(
    encoder: nn.Module,
    images: torch.Tensor,
    *,
    global_step: int = 0,
    return_features: bool = False,
):
    """Run NoPoSplat on RGB [0,1], applying its required [-1,1] transform."""
    if images.ndim != 5:
        raise ValueError(f"expected [B,V,3,H,W], got {tuple(images.shape)}")
    context = {"image": images.mul(2.0).sub(1.0)}
    return encoder(context, global_step=global_step, return_features=return_features)


def gaussian_checkpoint_metadata(num_views: int) -> dict[str, Any]:
    return {
        "format": "mhbench-gaudp-gaussian-v1",
        "num_views": int(num_views),
        "encoder": "noposplat_multi" if num_views > 2 else "noposplat",
        "feature_channels": 13,
        "rgb_range": [0.0, 1.0],
    }
