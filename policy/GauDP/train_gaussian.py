#!/usr/bin/env python3
"""Fine-tune vendored NoPoSplat on MHBench RGB/depth/camera observations."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

_OFFICIAL_FULL_LR_MARKERS = ("gaussian_param_head", "intrinsic_encoder")
_OFFICIAL_ADAMW_BETAS = (0.9, 0.95)

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


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _should_log_batch(completed: int, total: int, interval: int) -> bool:
    return interval > 0 and (completed == 1 or completed == total or completed % interval == 0)


def _progress_metrics(completed: int, total: int, started_at: float, phase: str) -> dict[str, float]:
    elapsed = max(time.monotonic() - started_at, 1e-6)
    rate = completed / elapsed
    eta = (total - completed) / rate if rate > 0 else 0.0
    metrics = {
        f"progress/{phase}_batch": completed,
        f"progress/{phase}_batches": total,
        f"progress/{phase}_fraction": completed / max(1, total),
        f"performance/{phase}_batches_per_second": rate,
        f"performance/{phase}_eta_seconds": eta,
        f"performance/{phase}_elapsed_seconds": elapsed,
    }
    if torch.cuda.is_available():
        gib = 1024**3
        metrics.update(
            {
                "system/gpu_allocated_gib": torch.cuda.memory_allocated() / gib,
                "system/gpu_reserved_gib": torch.cuda.memory_reserved() / gib,
                "system/gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
            }
        )
    return metrics


def _print_batch_progress(stage: str, epoch: int, epochs: int, phase: str, metrics: dict) -> None:
    completed = int(metrics[f"progress/{phase}_batch"])
    total = int(metrics[f"progress/{phase}_batches"])
    percent = 100.0 * float(metrics[f"progress/{phase}_fraction"])
    rate = float(metrics[f"performance/{phase}_batches_per_second"])
    eta = _format_duration(float(metrics[f"performance/{phase}_eta_seconds"]))
    loss = float(metrics[f"{phase}/batch_loss"])
    details = ""
    if f"{phase}/batch_rgb_loss" in metrics:
        details = (
            f" rgb={metrics[f'{phase}/batch_rgb_loss']:.6f}"
            f" depth={metrics[f'{phase}/batch_depth_loss']:.6f}"
            f" psnr={metrics[f'{phase}/batch_psnr']:.2f}"
        )
    gpu = ""
    if "system/gpu_allocated_gib" in metrics:
        gpu = (
            f" gpu={metrics['system/gpu_allocated_gib']:.2f}GiB"
            f" reserved={metrics['system/gpu_reserved_gib']:.2f}GiB"
            f" peak={metrics['system/gpu_peak_allocated_gib']:.2f}GiB"
        )
    print(
        f"[GauDP][{stage}] epoch={epoch + 1}/{epochs} {phase}={completed}/{total} "
        f"({percent:.1f}%) loss={loss:.6f}{details} rate={rate:.3f} batch/s eta={eta}{gpu}",
        flush=True,
    )


def _configure_finetuning(encoder, mode: str):
    if mode == "full":
        encoder.requires_grad_(True)
    elif mode == "heads":
        encoder.requires_grad_(True)
        encoder.backbone.requires_grad_(False)
        encoder.backbone.eval()
    else:
        raise ValueError(f"unsupported fine-tuning mode: {mode}")

    parameters = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError(f"fine-tuning mode {mode!r} selected no trainable parameters")
    return parameters


def _build_optimizer(
    encoder,
    mode: str,
    *,
    lr: float,
    backbone_lr_multiplier: float,
    weight_decay: float,
):
    """Build the AdamW parameter groups used by official NoPoSplat training."""
    trainable = [(name, parameter) for name, parameter in encoder.named_parameters() if parameter.requires_grad]
    if mode == "full":
        full_lr = [
            parameter
            for name, parameter in trainable
            if any(marker in name for marker in _OFFICIAL_FULL_LR_MARKERS)
        ]
        pretrained = [
            parameter
            for name, parameter in trainable
            if not any(marker in name for marker in _OFFICIAL_FULL_LR_MARKERS)
        ]
        if not full_lr or not pretrained:
            raise RuntimeError(
                "official NoPoSplat full fine-tuning requires both Gaussian/intrinsic "
                "and pretrained parameter groups"
            )
        parameter_groups = [
            {"params": full_lr, "lr": lr, "group_name": "head"},
            {
                "params": pretrained,
                "lr": lr * backbone_lr_multiplier,
                "group_name": "pretrained",
            },
        ]
    else:
        parameter_groups = [
            {
                "params": [parameter for _, parameter in trainable],
                "lr": lr,
                "group_name": "head",
            }
        ]

    return torch.optim.AdamW(
        parameter_groups,
        lr=lr,
        weight_decay=weight_decay,
        betas=_OFFICIAL_ADAMW_BETAS if mode == "full" else (0.9, 0.999),
        # The foreach implementation creates sizeable temporary tensor lists
        # on the first step. The scalar loop is slower but has a lower peak,
        # which matters for workstation GPUs.
        foreach=False,
    )


def _build_lr_scheduler(
    optimizer,
    *,
    warm_up_steps: int,
    max_steps: int,
    base_lr: float,
    min_lr_ratio: float,
):
    """Match NoPoSplat's linear warm-up followed by step-wise cosine decay."""
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, max_steps),
        eta_min=base_lr * min_lr_ratio,
    )
    if warm_up_steps == 0:
        return cosine
    warm_up = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1 / warm_up_steps,
        end_factor=1,
        total_iters=warm_up_steps,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warm_up, cosine],
        milestones=[warm_up_steps],
    )


def _learning_rate_metrics(optimizer) -> dict[str, float]:
    metrics = {"lr": optimizer.param_groups[0]["lr"]}
    for index, group in enumerate(optimizer.param_groups):
        name = group.get("group_name", str(index))
        metrics[f"lr/{name}"] = group["lr"]
    return metrics


def _set_train_mode(encoder, mode: str) -> None:
    encoder.train()
    if mode == "heads":
        # A parent .train() call also toggles frozen children. Keep the frozen
        # backbone deterministic while the depth/Gaussian heads are trained.
        encoder.backbone.eval()


def _save(
    path: Path,
    encoder,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    num_views: int,
    finetune_mode: str,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **gaussian_checkpoint_metadata(num_views),
            "encoder_state": encoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": step,
            "finetune_mode": finetune_mode,
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Gaussian/intrinsic head LR; defaults to 1e-4 for full and 1e-5 for heads",
    )
    parser.add_argument(
        "--backbone-lr-multiplier",
        type=float,
        default=0.1,
        help="full-mode LR multiplier for pretrained NoPoSplat parameters",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="defaults to 0.05 for full and the legacy 1e-6 for heads",
    )
    parser.add_argument(
        "--warm-up-steps",
        type=int,
        default=None,
        help="full-mode linear LR warm-up steps; defaults to official 1x8 value 2000",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="cosine eta_min as a fraction of the Gaussian/intrinsic head LR",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=None,
        help="defaults to official 0.5 for full and the legacy 1.0 for heads",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="accumulate micro-batches before each optimizer/scheduler step",
    )
    parser.add_argument("--depth-weight", type=float, default=0.1)
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="print and send batch progress to W&B every N batches; 0 disables batch progress logs",
    )
    parser.add_argument(
        "--finetune-mode",
        choices=("full", "heads"),
        default="full",
        help="update the full NoPoSplat encoder, or freeze its ViT-L backbone and update only its depth/Gaussian heads",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("GAUDP_WANDB_MODE", "online"),
        help="W&B mode; JSONL is always written regardless of this setting",
    )
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "MHBench-GauDP"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default="", help="comma-separated W&B tags")
    parser.add_argument("--debug", action="store_true", help="run one train and validation batch")
    args = parser.parse_args()
    if args.lr is None:
        args.lr = 1e-4 if args.finetune_mode == "full" else 1e-5
    if args.weight_decay is None:
        args.weight_decay = 0.05 if args.finetune_mode == "full" else 1e-6
    if args.warm_up_steps is None:
        args.warm_up_steps = 2000 if args.finetune_mode == "full" else 0
    if args.gradient_clip is None:
        args.gradient_clip = 0.5 if args.finetune_mode == "full" else 1.0
    if args.log_every < 0:
        parser.error("--log-every must be non-negative")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.backbone_lr_multiplier < 0:
        parser.error("--backbone-lr-multiplier must be non-negative")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative")
    if args.warm_up_steps < 0:
        parser.error("--warm-up-steps must be non-negative")
    if not 0 <= args.min_lr_ratio <= 1:
        parser.error("--min-lr-ratio must be between 0 and 1")
    if args.gradient_clip <= 0:
        parser.error("--gradient-clip must be positive")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient-accumulation-steps must be positive")

    global DataLoader
    global GaussianFrameDataset, build_gaussian_encoder, encode_gaussians
    global gaussian_checkpoint_metadata, load_gaussian_checkpoint
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
    encoder.to(device)
    trainable_parameters = _configure_finetuning(encoder, args.finetune_mode)
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_count = sum(parameter.numel() for parameter in encoder.parameters())
    print(
        f"[GauDP] fine-tuning mode={args.finetune_mode}: "
        f"trainable={trainable_count / 1e6:.1f}M / total={total_count / 1e6:.1f}M parameters"
    )
    optimizer = _build_optimizer(
        encoder,
        args.finetune_mode,
        lr=args.lr,
        backbone_lr_multiplier=args.backbone_lr_multiplier,
        weight_decay=args.weight_decay,
    )
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    train_batches = min(1, len(train_loader)) if args.debug else len(train_loader)
    val_batches = min(1, len(val_loader)) if args.debug else len(val_loader)
    epochs = 1 if args.debug else args.epochs
    optimizer_steps_per_epoch = math.ceil(train_batches / args.gradient_accumulation_steps)
    max_optimizer_steps = max(1, epochs * optimizer_steps_per_epoch)
    if args.finetune_mode == "full":
        scheduler = _build_lr_scheduler(
            optimizer,
            warm_up_steps=args.warm_up_steps,
            max_steps=max_optimizer_steps,
            base_lr=args.lr,
            min_lr_ratio=args.min_lr_ratio,
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    print(
        f"[GauDP][gaussian] device={device} cameras={train_data.camera_order} "
        f"train_samples={len(train_data)} val_samples={len(val_data)} "
        f"train_batches={train_batches} val_batches={val_batches} "
        f"batch_size={args.batch_size} accumulation={args.gradient_accumulation_steps} "
        f"effective_batch_size={args.batch_size * args.gradient_accumulation_steps} "
        f"workers={args.num_workers} epochs={epochs} optimizer_steps={max_optimizer_steps} "
        f"warm_up_steps={args.warm_up_steps} log_every={args.log_every}",
        flush=True,
    )
    for group in optimizer.param_groups:
        print(
            f"[GauDP] optimizer group={group['group_name']} "
            f"initial_lr={group['initial_lr']:.2e} parameters={sum(p.numel() for p in group['params']) / 1e6:.1f}M",
            flush=True,
        )

    best = math.inf
    global_step = 0
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
            epoch_started = time.monotonic()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            print(f"[GauDP][gaussian] epoch={epoch + 1}/{epochs} train started", flush=True)
            _set_train_mode(encoder, args.finetune_mode)
            train_sums = {"loss": 0.0, "rgb_loss": 0.0, "depth_loss": 0.0, "psnr": 0.0}
            train_count = 0
            train_started = time.monotonic()
            for batch_index, batch in enumerate(train_loader):
                batch = _to_device(batch, device)
                accumulation_index = batch_index % args.gradient_accumulation_steps
                if accumulation_index == 0:
                    optimizer.zero_grad(set_to_none=True)
                accumulation_size = min(
                    args.gradient_accumulation_steps,
                    train_batches - batch_index + accumulation_index,
                )
                loss, batch_metrics = reconstruction_loss(
                    encoder, batch, global_step=global_step, depth_weight=args.depth_weight
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite reconstruction loss at epoch={epoch + 1}, batch={batch_index + 1}"
                    )
                (loss / accumulation_size).backward()
                should_step = (
                    accumulation_index + 1 == args.gradient_accumulation_steps
                    or batch_index + 1 == train_batches
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        args.gradient_clip,
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                train_sums["loss"] += float(loss.detach())
                train_sums["rgb_loss"] += batch_metrics["rgb"]
                train_sums["depth_loss"] += batch_metrics["depth"]
                train_sums["psnr"] += batch_metrics["psnr"]
                train_count += 1
                completed = batch_index + 1
                if _should_log_batch(completed, train_batches, args.log_every):
                    progress = {
                        "record_type": "batch",
                        "epoch": epoch,
                        "global_step": global_step,
                        "train/batch_loss": float(loss.detach()),
                        "train/batch_rgb_loss": batch_metrics["rgb"],
                        "train/batch_depth_loss": batch_metrics["depth"],
                        "train/batch_psnr": batch_metrics["psnr"],
                        **_learning_rate_metrics(optimizer),
                        **_progress_metrics(completed, train_batches, train_started, "train"),
                    }
                    logger.log(progress, step=global_step)
                    _print_batch_progress("gaussian", epoch, epochs, "train", progress)
                if args.debug:
                    break

            encoder.eval()
            print(f"[GauDP][gaussian] epoch={epoch + 1}/{epochs} validation started", flush=True)
            val_sums = {"loss": 0.0, "rgb_loss": 0.0, "depth_loss": 0.0, "psnr": 0.0}
            val_count = 0
            val_started = time.monotonic()
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
                    completed = batch_index + 1
                    if _should_log_batch(completed, val_batches, args.log_every):
                        progress = {
                            "val/batch_loss": float(loss),
                            "val/batch_rgb_loss": batch_metrics["rgb"],
                            "val/batch_depth_loss": batch_metrics["depth"],
                            "val/batch_psnr": batch_metrics["psnr"],
                            **_progress_metrics(completed, val_batches, val_started, "val"),
                        }
                        _print_batch_progress("gaussian", epoch, epochs, "val", progress)
                    if args.debug:
                        break
            metrics = {
                "record_type": "epoch",
                "epoch": epoch,
                **_learning_rate_metrics(optimizer),
                "performance/epoch_seconds": time.monotonic() - epoch_started,
                **{f"train/{key}": value / max(1, train_count) for key, value in train_sums.items()},
                **{f"val/{key}": value / max(1, val_count) for key, value in val_sums.items()},
            }
            print(f"[GauDP][gaussian] saving last checkpoint to {args.output / 'last.ckpt'}", flush=True)
            save_started = time.monotonic()
            _save(
                args.output / "last.ckpt",
                encoder,
                optimizer,
                scheduler,
                epoch,
                global_step,
                num_views,
                args.finetune_mode,
                metrics,
            )
            print(
                f"[GauDP][gaussian] saved last checkpoint in "
                f"{_format_duration(time.monotonic() - save_started)}",
                flush=True,
            )
            if metrics["val/loss"] < best:
                best = metrics["val/loss"]
                print(f"[GauDP][gaussian] new best val/loss={best:.6f}; saving best checkpoint", flush=True)
                best_save_started = time.monotonic()
                _save(
                    args.output / "best.ckpt",
                    encoder,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    num_views,
                    args.finetune_mode,
                    metrics,
                )
                print(
                    f"[GauDP][gaussian] saved best checkpoint in "
                    f"{_format_duration(time.monotonic() - best_save_started)}",
                    flush=True,
                )
            logger.log(metrics, step=global_step)
            print(f"[GauDP][gaussian] step={global_step} {metrics}", flush=True)


if __name__ == "__main__":
    main()
