#!/usr/bin/env python3
"""Train GauDP from offline Gaussian features; deployment keeps online encoding."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

_XPL_ROOT = Path(__file__).resolve().parents[2]
if str(_XPL_ROOT) not in sys.path:
    sys.path.insert(0, str(_XPL_ROOT))

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _parse_optional_int(value) -> int | None:
    if value is None or str(value).strip().lower() in ("none", "null", ""):
        return None
    return int(value)


def _parse_crop_shape(value) -> tuple[int, int] | None:
    """Accept 'H W', 'HxW', 'H,W' or 'none'."""
    if value is None or str(value).strip().lower() in ("none", "null", ""):
        return None
    parts = [part for part in str(value).replace("x", " ").replace(",", " ").split() if part]
    if len(parts) != 2:
        raise ValueError(f"expected two integers or 'none', got {value!r}")
    return (int(parts[0]), int(parts[1]))


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


def _print_batch_progress(epoch: int, epochs: int, phase: str, metrics: dict) -> None:
    completed = int(metrics[f"progress/{phase}_batch"])
    total = int(metrics[f"progress/{phase}_batches"])
    percent = 100.0 * float(metrics[f"progress/{phase}_fraction"])
    rate = float(metrics[f"performance/{phase}_batches_per_second"])
    eta = _format_duration(float(metrics[f"performance/{phase}_eta_seconds"]))
    loss = float(metrics[f"{phase}/batch_loss"])
    diagnostics = ""
    action_mae_key = f"{phase}/batch/action/x0_clipped_mae"
    cosine_key = f"{phase}/batch/diffusion/noise_cosine"
    grad_key = f"{phase}/batch/optimization/grad_norm"
    if action_mae_key in metrics:
        diagnostics += f" action_mae={metrics[action_mae_key]:.4f}"
    if cosine_key in metrics:
        diagnostics += f" noise_cos={metrics[cosine_key]:.3f}"
    if grad_key in metrics:
        diagnostics += f" grad_norm={metrics[grad_key]:.3f}"
    gpu = ""
    if "system/gpu_allocated_gib" in metrics:
        gpu = (
            f" gpu={metrics['system/gpu_allocated_gib']:.2f}GiB"
            f" reserved={metrics['system/gpu_reserved_gib']:.2f}GiB"
            f" peak={metrics['system/gpu_peak_allocated_gib']:.2f}GiB"
        )
    print(
        f"[GauDP][policy] epoch={epoch + 1}/{epochs} {phase}={completed}/{total} "
        f"({percent:.1f}%) loss={loss:.6f}{diagnostics} "
        f"rate={rate:.3f} batch/s eta={eta}{gpu}",
        flush=True,
    )


def _accumulate(sums: dict[str, float], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        sums[key] = sums.get(key, 0.0) + float(value)


def _save(
    path: Path, policy: GauDPPolicy, optimizer, scheduler, epoch, metrics,
    gaussian_checkpoint: Path, camera_order: list[str],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        policy_checkpoint_payload(
            policy,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            epoch=epoch,
            metrics=metrics,
            # Preserve checkpoints outside the run directory (for example the
            # official NoPoSplat checkpoint used with RGB-only LeRobot data).
            gaussian_checkpoint=str(gaussian_checkpoint.expanduser().resolve()),
            camera_order=list(camera_order),
        ),
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gaussian", type=Path, required=True)
    parser.add_argument("--gaussian-features", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--obs-steps", type=int, default=1)
    parser.add_argument("--action-steps", type=int, default=6)
    parser.add_argument("--inference-steps", type=int, default=100)
    # --- vision recipe -------------------------------------------------------
    # These defaults are the *current* recipe, not what the first MHBench GauDP
    # runs used; every value is recorded in the checkpoint so evaluation rebuilds
    # the same network. `--crop-shape none --image-norm symmetric
    # --group-norm-divisor none` reproduces those runs exactly.
    parser.add_argument(
        "--crop-shape",
        default="216 288",
        help="random crop while training / centre crop at eval, as 'H W'; "
             "'none' disables it. Upstream Policy-Lightning leaves crop_shape "
             "null, but MHBench's own DP baseline crops to 90%% of the frame and "
             "its config records why: without it validation loss bottoms early "
             "and then climbs. GauDP shows the same curve.",
    )
    parser.add_argument(
        "--image-norm",
        choices=("imagenet", "symmetric"),
        default="imagenet",
        help="normalization applied to the fused 3-channel view before the ResNet. "
             "'imagenet' is upstream's MultiImageObsEncoder(imagenet_norm=True); "
             "'symmetric' is (x-0.5)/0.5, what this port used before.",
    )
    parser.add_argument(
        "--group-norm-divisor",
        default="16",
        help="BatchNorm->GroupNorm grouping as num_features//DIVISOR (upstream "
             "uses 16); 'none' keeps this port's original min(32, num_features).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="print and send batch progress to W&B every N batches; 0 disables batch progress logs",
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
    if args.log_every < 0:
        parser.error("--log-every must be non-negative")
    try:
        args.crop_shape = _parse_crop_shape(args.crop_shape)
    except ValueError as error:
        parser.error(f"--crop-shape: {error}")
    try:
        args.group_norm_divisor = _parse_optional_int(args.group_norm_divisor)
    except ValueError as error:
        parser.error(f"--group-norm-divisor: {error}")

    global np, torch, DataLoader
    global GauDPSequenceDataset
    global GauDPPolicy, policy_checkpoint_payload
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from XPolicyLab.policy.GauDP.gaudp.dataset import GauDPSequenceDataset
    from XPolicyLab.policy.GauDP.gaudp.experiment_logger import ExperimentLogger, parse_wandb_tags
    from XPolicyLab.policy.GauDP.gaudp.policy import GauDPPolicy, policy_checkpoint_payload

    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("GauDP policy training requires CUDA for the frozen NoPoSplat encoder")
    train_data = GauDPSequenceDataset(
        args.data, True, args.horizon, args.obs_steps, args.gaussian_features
    )
    val_data = GauDPSequenceDataset(
        args.data, False, args.horizon, args.obs_steps, args.gaussian_features
    )
    cached_checkpoint = Path(train_data.gaussian_checkpoint).resolve()
    requested_checkpoint = args.gaussian.expanduser().resolve()
    if cached_checkpoint != requested_checkpoint:
        raise ValueError(
            f"offline features were extracted with {cached_checkpoint}, but policy training "
            f"requested {requested_checkpoint}; re-extract the cache or select the matching checkpoint"
        )
    policy = GauDPPolicy(
        num_views=len(train_data.camera_order),
        horizon=args.horizon,
        n_obs_steps=args.obs_steps,
        n_action_steps=args.action_steps,
        num_inference_steps=args.inference_steps,
        crop_shape=args.crop_shape,
        image_norm=args.image_norm,
        group_norm_divisor=args.group_norm_divisor,
        # Policy checkpoints deliberately exclude this module. Training uses
        # cached features, while model.py constructs and loads the real
        # NoPoSplat encoder for online benchmark inference.
        gaussian_encoder=nn.Identity(),
    )
    states, actions = train_data.normalization_arrays()
    policy.normalizer.fit(states, actions)
    val_states, val_actions = val_data.normalization_arrays()
    normalization_metrics = policy.normalizer.range_diagnostics(val_states, val_actions)
    policy.to(device)

    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, 1 if args.debug else args.epochs))
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    train_batches = min(1, len(train_loader)) if args.debug else len(train_loader)
    val_batches = min(1, len(val_loader)) if args.debug else len(val_loader)

    trainable_count = sum(parameter.numel() for parameter in trainable)
    total_count = sum(parameter.numel() for parameter in policy.parameters())
    print(
        f"[GauDP][policy] device={device} cameras={train_data.camera_order} "
        f"train_samples={len(train_data)} val_samples={len(val_data)} "
        f"train_batches={train_batches} val_batches={val_batches} "
        f"batch_size={args.batch_size} workers={args.num_workers} epochs={1 if args.debug else args.epochs} "
        f"trainable={trainable_count / 1e6:.1f}M / total={total_count / 1e6:.1f}M "
        f"gaussian_checkpoint={requested_checkpoint} gaussian_features={args.gaussian_features} "
        f"split={train_data.split_source} log_every={args.log_every} "
        f"crop_shape={args.crop_shape} image_norm={args.image_norm} "
        f"group_norm_divisor={args.group_norm_divisor} "
        f"val_state_oor={normalization_metrics['normalization/val_state_out_of_range_fraction']:.6f} "
        f"val_action_oor={normalization_metrics['normalization/val_action_out_of_range_fraction']:.6f}",
        flush=True,
    )

    best = math.inf
    epochs = 1 if args.debug else args.epochs
    global_step = 0
    run_name = args.wandb_run_name or f"{args.output.parent.name}-policy"
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
            print(f"[GauDP][policy] epoch={epoch + 1}/{epochs} train started", flush=True)
            policy.train()
            train_sums: dict[str, float] = {}
            train_count = 0
            train_started = time.monotonic()
            for batch_index, batch in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)
                loss, batch_metrics = policy.compute_loss(
                    _to_device(batch, device), return_metrics=True
                )
                loss.backward()
                if any(parameter.grad is not None for parameter in policy.gaussian_encoder.parameters()):
                    raise RuntimeError("frozen Gaussian encoder unexpectedly received gradients")
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
                batch_metrics["optimization/grad_norm"] = grad_norm
                batch_metrics["optimization/gradient_clipped"] = float(grad_norm > 1.0)
                optimizer.step()
                _accumulate(train_sums, batch_metrics)
                train_count += 1
                global_step += 1
                completed = batch_index + 1
                if _should_log_batch(completed, train_batches, args.log_every):
                    progress = {
                        "record_type": "batch",
                        "epoch": epoch,
                        "global_step": global_step,
                        "train/batch_loss": float(loss.detach()),
                        "train/batch/optimization/lr": optimizer.param_groups[0]["lr"],
                        **{f"train/batch/{key}": value for key, value in batch_metrics.items()},
                        **_progress_metrics(completed, train_batches, train_started, "train"),
                    }
                    logger.log(progress, step=global_step)
                    _print_batch_progress(epoch, epochs, "train", progress)
                if args.debug:
                    break
            scheduler.step()

            policy.eval()
            print(f"[GauDP][policy] epoch={epoch + 1}/{epochs} validation started", flush=True)
            val_sums: dict[str, float] = {}
            val_count = 0
            val_started = time.monotonic()
            with torch.no_grad():
                for batch_index, batch in enumerate(val_loader):
                    loss, batch_metrics = policy.compute_loss(
                        _to_device(batch, device), return_metrics=True
                    )
                    batch_loss = float(loss)
                    _accumulate(val_sums, batch_metrics)
                    val_count += 1
                    completed = batch_index + 1
                    if _should_log_batch(completed, val_batches, args.log_every):
                        progress = {
                            "val/batch_loss": batch_loss,
                            **{f"val/batch/{key}": value for key, value in batch_metrics.items()},
                            **_progress_metrics(completed, val_batches, val_started, "val"),
                        }
                        _print_batch_progress(epoch, epochs, "val", progress)
                    if args.debug:
                        break
            metrics = {
                "record_type": "epoch",
                "epoch": epoch,
                "lr": scheduler.get_last_lr()[0],
                "performance/epoch_seconds": time.monotonic() - epoch_started,
                **normalization_metrics,
                "train/loss": train_sums.get("diffusion/noise_mse", 0.0) / max(1, train_count),
                "val/loss": val_sums.get("diffusion/noise_mse", 0.0) / max(1, val_count),
                **{f"train/{key}": value / max(1, train_count) for key, value in train_sums.items()},
                **{f"val/{key}": value / max(1, val_count) for key, value in val_sums.items()},
            }
            print(f"[GauDP][policy] saving last checkpoint to {args.output / 'last.ckpt'}", flush=True)
            save_started = time.monotonic()
            _save(
                args.output / "last.ckpt", policy, optimizer, scheduler, epoch, metrics,
                args.gaussian, train_data.camera_order,
            )
            print(
                f"[GauDP][policy] saved last checkpoint in "
                f"{_format_duration(time.monotonic() - save_started)}",
                flush=True,
            )
            if metrics["val/loss"] < best:
                best = metrics["val/loss"]
                print(f"[GauDP][policy] new best val/loss={best:.6f}; saving best checkpoint", flush=True)
                best_save_started = time.monotonic()
                _save(
                    args.output / "best.ckpt", policy, optimizer, scheduler, epoch, metrics,
                    args.gaussian, train_data.camera_order,
                )
                print(
                    f"[GauDP][policy] saved best checkpoint in "
                    f"{_format_duration(time.monotonic() - best_save_started)}",
                    flush=True,
                )
            logger.log(metrics, step=global_step)
            print(f"[GauDP][policy] step={global_step} {metrics}", flush=True)


if __name__ == "__main__":
    main()
