#!/usr/bin/env python3
"""Diagnose GauDP cache parity and reconstruction depth errors on keyframes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

_XPL_ROOT = Path(__file__).resolve().parents[2]
if str(_XPL_ROOT) not in sys.path:
    sys.path.insert(0, str(_XPL_ROOT))

from XPolicyLab.policy.GauDP.gaudp.dataset import GaussianFrameDataset
from XPolicyLab.policy.GauDP.gaudp.gaussian import (
    build_gaussian_encoder,
    encode_gaussians,
    load_gaussian_checkpoint,
)
from XPolicyLab.policy.GauDP.gaudp.recon_dump import KeyFrame, save_keyframe
import XPolicyLab.policy.GauDP.train_gaussian as gaussian_training


CHANNELS = (
    "mean_x", "mean_y", "mean_z",
    "cov_xx", "cov_xy", "cov_xz", "cov_yx", "cov_yy", "cov_yz",
    "cov_zx", "cov_zy", "cov_zz", "opacity",
)
DEPTH_BINS = (0.0, 0.5, 0.75, 1.0, 1.5, float("inf"))


class ErrorStats:
    def __init__(self) -> None:
        self.count = 0
        self.abs_sum = 0.0
        self.signed_sum = 0.0
        self.squared_sum = 0.0

    def add(self, error: torch.Tensor) -> None:
        values = error.detach().double()
        self.count += values.numel()
        self.abs_sum += values.abs().sum().item()
        self.signed_sum += values.sum().item()
        self.squared_sum += values.square().sum().item()

    def result(self) -> dict[str, float | int]:
        denominator = max(1, self.count)
        return {
            "pixels": self.count,
            "mae_m": self.abs_sum / denominator,
            "bias_m": self.signed_sum / denominator,
            "rmse_m": (self.squared_sum / denominator) ** 0.5,
        }


def select_keyframes(dataset: GaussianFrameDataset, fractions: tuple[float, ...]) -> list[KeyFrame]:
    selected: list[KeyFrame] = []
    offset = 0
    for episode_id, (start, end) in zip(dataset.episode_ids, dataset.episode_ranges):
        length = end - start
        for fraction in fractions:
            frame = min(int(fraction * length), length - 1)
            selected.append(KeyFrame(offset + frame, int(episode_id), frame, length))
        offset += length
    return selected


def edge_mask(depth: torch.Tensor, valid: torch.Tensor, threshold: float) -> torch.Tensor:
    edge = torch.zeros_like(valid)
    horizontal = valid[..., :, 1:] & valid[..., :, :-1]
    vertical = valid[..., 1:, :] & valid[..., :-1, :]
    horizontal &= (depth[..., :, 1:] - depth[..., :, :-1]).abs() >= threshold
    vertical &= (depth[..., 1:, :] - depth[..., :-1, :]).abs() >= threshold
    edge[..., :, 1:] |= horizontal
    edge[..., :, :-1] |= horizontal
    edge[..., 1:, :] |= vertical
    edge[..., :-1, :] |= vertical
    return edge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fractions", default="0,0.25,0.5,0.75")
    parser.add_argument("--edge-threshold", type=float, default=0.02)
    args = parser.parse_args()

    fractions = tuple(sorted({float(value) for value in args.fractions.split(",")}))
    if not fractions or any(not 0.0 <= value < 1.0 for value in fractions):
        parser.error("--fractions must contain comma-separated values in [0,1)")
    if args.edge_threshold <= 0:
        parser.error("--edge-threshold must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("diagnostic requires CUDA and the CUDA Gaussian rasterizer")

    device = torch.device("cuda")
    dataset = GaussianFrameDataset(args.data, train=False)
    keyframes = select_keyframes(dataset, fractions)
    encoder = build_gaussian_encoder(len(dataset.camera_order))
    missing, unexpected = load_gaussian_checkpoint(encoder, args.checkpoint, strict=False)
    encoder.to(device).eval().requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=True)

    depth_stats = {"all": ErrorStats(), "near_quartile": ErrorStats(), "edges": ErrorStats()}
    for lower, upper in zip(DEPTH_BINS[:-1], DEPTH_BINS[1:]):
        depth_stats[f"[{lower},{upper})"] = ErrorStats()
    feature_abs = torch.zeros(len(CHANNELS), dtype=torch.float64)
    feature_squared = torch.zeros(len(CHANNELS), dtype=torch.float64)
    feature_count = torch.zeros(len(CHANNELS), dtype=torch.int64)
    feature_max = torch.zeros(len(CHANNELS), dtype=torch.float64)
    quant_abs = torch.zeros(len(CHANNELS), dtype=torch.float64)
    exact = 0
    total = 0
    dot = online_norm = cache_norm = 0.0
    records: list[dict] = []

    with h5py.File(args.features, "r") as cache, torch.no_grad():
        recorded_checkpoint = Path(str(cache.attrs.get("gaussian_checkpoint", ""))).resolve()
        requested_checkpoint = args.checkpoint.expanduser().resolve()
        if recorded_checkpoint != requested_checkpoint:
            raise ValueError(f"cache checkpoint {recorded_checkpoint} != requested {requested_checkpoint}")
        if json.loads(cache.attrs["camera_order"]) != dataset.camera_order:
            raise ValueError("cache and dataset camera order differ")

        for keyframe in keyframes:
            sample = dataset[keyframe.position]
            batch = {name: value.unsqueeze(0).to(device) for name, value in sample.items()}
            images = batch["images"]
            gaussians, online = encode_gaussians(encoder, images, return_features=True)
            rendered_rgb, rendered_depth = gaussian_training._render(
                gaussians, batch, tuple(images.shape[-2:])
            )
            global_index = dataset.indices[keyframe.position]
            cached = torch.from_numpy(np.asarray(cache["gaussian_features"][global_index])).to(
                device=device, dtype=torch.float32
            ).unsqueeze(0)
            rounded = online.to(torch.float16).float()
            difference = cached - online
            quant_difference = rounded - online
            flat = difference.permute(2, 0, 1, 3, 4).reshape(len(CHANNELS), -1).double()
            quant_flat = quant_difference.permute(2, 0, 1, 3, 4).reshape(len(CHANNELS), -1).double()
            feature_abs += flat.abs().sum(dim=1).cpu()
            feature_squared += flat.square().sum(dim=1).cpu()
            feature_count += flat.shape[1]
            feature_max = torch.maximum(feature_max, flat.abs().amax(dim=1).cpu())
            quant_abs += quant_flat.abs().sum(dim=1).cpu()
            exact += int((cached == rounded).sum())
            total += cached.numel()
            dot += float((cached.double() * online.double()).sum())
            online_norm += float(online.double().square().sum())
            cache_norm += float(cached.double().square().sum())

            target = batch["depth"]
            valid = torch.isfinite(target) & (target > 0) & torch.isfinite(rendered_depth)
            error = rendered_depth - target
            depth_stats["all"].add(error[valid])
            threshold = torch.quantile(target[valid], 0.25)
            depth_stats["near_quartile"].add(error[valid & (target <= threshold)])
            edges = edge_mask(target, valid, args.edge_threshold)
            depth_stats["edges"].add(error[edges])
            for lower, upper in zip(DEPTH_BINS[:-1], DEPTH_BINS[1:]):
                mask = valid & (target >= lower) & (target < upper)
                depth_stats[f"[{lower},{upper})"].add(error[mask])

            record = {
                "episode_id": keyframe.episode_id,
                "frame": keyframe.frame,
                "global_index": int(global_index),
                "feature_mae": float(difference.abs().mean()),
                "feature_max_abs": float(difference.abs().max()),
                "quantization_mae": float(quant_difference.abs().mean()),
                "depth_mae_m": float(error[valid].abs().mean()),
                "near_quartile_mae_m": float(error[valid & (target <= threshold)].abs().mean()),
                "edge_mae_m": float(error[edges].abs().mean()) if edges.any() else None,
            }
            records.append(record)
            save_keyframe(
                args.output,
                keyframe,
                images=images[0].float().cpu().numpy(),
                depth=target[0].float().cpu().numpy(),
                rendered_rgb=rendered_rgb[0].float().cpu().numpy(),
                rendered_depth=rendered_depth[0].float().cpu().numpy(),
                near=float(batch["near"][0, 0]),
                far=float(batch["far"][0, 0]),
            )

    channel_metrics = {}
    for index, name in enumerate(CHANNELS):
        count = max(1, int(feature_count[index]))
        channel_metrics[name] = {
            "mae": float(feature_abs[index] / count),
            "rmse": float((feature_squared[index] / count).sqrt()),
            "max_abs": float(feature_max[index]),
            "float16_quantization_mae": float(quant_abs[index] / count),
        }
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "features": str(args.features.resolve()),
        "frames": len(keyframes),
        "missing_checkpoint_keys": missing,
        "unexpected_checkpoint_keys": unexpected,
        "feature_parity": {
            "exact_after_float16_fraction": exact / max(1, total),
            "cosine_similarity": dot / max((online_norm * cache_norm) ** 0.5, 1e-30),
            "channels": channel_metrics,
        },
        "depth": {name: stats.result() for name, stats in depth_stats.items()},
        "per_frame": records,
    }
    (args.output / "diagnostic.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output / "per_frame.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
