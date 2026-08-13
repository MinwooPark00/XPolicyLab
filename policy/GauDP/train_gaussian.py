#!/usr/bin/env python3
"""Fine-tune vendored NoPoSplat on MHBench RGB/depth/camera observations."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

_XPL_ROOT = Path(__file__).resolve().parents[2]
if str(_XPL_ROOT) not in sys.path:
    sys.path.insert(0, str(_XPL_ROOT))

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _render(gaussians, batch: dict[str, torch.Tensor], image_shape: tuple[int, int]):
    try:
        from XPolicyLab.policy.GauDP.gaudp.third_party.noposplat.model.decoder.cuda_splatting import render_cuda
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "NoPoSplat CUDA rasterizer is unavailable. Run install.sh and verify "
            "that diff-gaussian-rasterization-w-pose imports in this environment."
        ) from error
    from einops import rearrange, repeat

    extrinsics = batch["extrinsics"]
    intrinsics = batch["intrinsics"]
    near, far = batch["near"], batch["far"]
    b, v = extrinsics.shape[:2]
    background = torch.zeros((b * v, 3), device=extrinsics.device, dtype=extrinsics.dtype)
    color, depth = render_cuda(
        rearrange(extrinsics, "b v i j -> (b v) i j"),
        rearrange(intrinsics, "b v i j -> (b v) i j"),
        rearrange(near, "b v -> (b v)"),
        rearrange(far, "b v -> (b v)"),
        image_shape,
        background,
        repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
        repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
        repeat(gaussians.harmonics, "b g c d -> (b v) g c d", v=v),
        repeat(gaussians.opacities, "b g -> (b v) g", v=v),
        scale_invariant=False,
    )
    return (
        rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v),
        rearrange(depth, "(b v) h w -> b v h w", b=b, v=v),
    )


def reconstruction_loss(
    encoder,
    batch: dict[str, torch.Tensor],
    *,
    global_step: int,
    depth_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    import torch
    import torch.nn.functional as functional
    from XPolicyLab.policy.GauDP.gaudp.gaussian import encode_gaussians

    images = batch["images"]
    gaussians = encode_gaussians(encoder, images, global_step=global_step)
    rendered_rgb, rendered_depth = _render(gaussians, batch, tuple(images.shape[-2:]))
    rgb_loss = functional.mse_loss(rendered_rgb, images)
    target_depth = batch["depth"]
    valid = torch.isfinite(target_depth) & (target_depth > 0)
    depth_loss = (
        functional.l1_loss(rendered_depth[valid], target_depth[valid])
        if valid.any()
        else rgb_loss.new_zeros(())
    )
    loss = rgb_loss + depth_weight * depth_loss
    return loss, {
        "rgb": float(rgb_loss.detach()),
        "depth": float(depth_loss.detach()),
        "psnr": float(-10.0 * torch.log10(rgb_loss.detach().clamp_min(1e-10))),
    }


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _save(path: Path, encoder, optimizer, epoch: int, step: int, num_views: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **gaussian_checkpoint_metadata(num_views),
            "encoder_state": encoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": step,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--depth-weight", type=float, default=0.1)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("GAUDP_WANDB_MODE", "offline"),
        help="W&B mode; JSONL is always written regardless of this setting",
    )
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "MHBench-GauDP"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default="", help="comma-separated W&B tags")
    parser.add_argument("--debug", action="store_true", help="run one train and validation batch")
    args = parser.parse_args()

    global np, torch, DataLoader
    global GaussianFrameDataset, build_gaussian_encoder, encode_gaussians
    global gaussian_checkpoint_metadata, load_gaussian_checkpoint
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from XPolicyLab.policy.GauDP.gaudp.dataset import GaussianFrameDataset
    from XPolicyLab.policy.GauDP.gaudp.gaussian import (
        build_gaussian_encoder,
        encode_gaussians,
        gaussian_checkpoint_metadata,
        load_gaussian_checkpoint,
    )
    from XPolicyLab.policy.GauDP.gaudp.experiment_logger import ExperimentLogger, parse_wandb_tags

    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("NoPoSplat fine-tuning requires a CUDA device and CUDA rasterizer")
    train_data = GaussianFrameDataset(args.data, train=True)
    val_data = GaussianFrameDataset(args.data, train=False)
    num_views = len(train_data.camera_order)
    encoder = build_gaussian_encoder(num_views)
    missing, unexpected = load_gaussian_checkpoint(encoder, args.pretrained, strict=False)
    print(f"[GauDP] initialized NoPoSplat: missing={len(missing)}, unexpected={len(unexpected)}")
    encoder.to(device).train()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-6)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    best = math.inf
    global_step = 0
    epochs = 1 if args.debug else args.epochs
    run_name = args.wandb_run_name or f"{args.output.parent.name}-gaussian"
    with ExperimentLogger(
        args.output,
        config=vars(args),
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_run_name=run_name,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_tags=parse_wandb_tags(args.wandb_tags),
    ) as logger:
        for epoch in range(epochs):
            encoder.train()
            train_sums = {"loss": 0.0, "rgb_loss": 0.0, "depth_loss": 0.0, "psnr": 0.0}
            train_count = 0
            for batch_index, batch in enumerate(train_loader):
                batch = _to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss, batch_metrics = reconstruction_loss(
                    encoder, batch, global_step=global_step, depth_weight=args.depth_weight
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                optimizer.step()
                train_sums["loss"] += float(loss.detach())
                train_sums["rgb_loss"] += batch_metrics["rgb"]
                train_sums["depth_loss"] += batch_metrics["depth"]
                train_sums["psnr"] += batch_metrics["psnr"]
                train_count += 1
                global_step += 1
                if args.debug:
                    break

            encoder.eval()
            val_sums = {"loss": 0.0, "rgb_loss": 0.0, "depth_loss": 0.0, "psnr": 0.0}
            val_count = 0
            with torch.no_grad():
                for batch_index, batch in enumerate(val_loader):
                    loss, batch_metrics = reconstruction_loss(
                        encoder,
                        _to_device(batch, device),
                        global_step=global_step,
                        depth_weight=args.depth_weight,
                    )
                    val_sums["loss"] += float(loss)
                    val_sums["rgb_loss"] += batch_metrics["rgb"]
                    val_sums["depth_loss"] += batch_metrics["depth"]
                    val_sums["psnr"] += batch_metrics["psnr"]
                    val_count += 1
                    if args.debug:
                        break
            metrics = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                **{f"train/{key}": value / max(1, train_count) for key, value in train_sums.items()},
                **{f"val/{key}": value / max(1, val_count) for key, value in val_sums.items()},
            }
            _save(args.output / "last.ckpt", encoder, optimizer, epoch, global_step, num_views, metrics)
            if metrics["val/loss"] < best:
                best = metrics["val/loss"]
                _save(args.output / "best.ckpt", encoder, optimizer, epoch, global_step, num_views, metrics)
            logger.log(metrics, step=global_step)
            print(f"[GauDP][gaussian] step={global_step} {metrics}")


if __name__ == "__main__":
    main()
