#!/usr/bin/env python3
"""Evaluate a frozen NoPoSplat checkpoint on the GauDP validation split."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_XPL_ROOT = Path(__file__).resolve().parents[2]
if str(_XPL_ROOT) not in sys.path:
    sys.path.insert(0, str(_XPL_ROOT))

from XPolicyLab.policy.GauDP.gaudp.dataset import GaussianFrameDataset
from XPolicyLab.policy.GauDP.gaudp.experiment_logger import ExperimentLogger, parse_wandb_tags
from XPolicyLab.policy.GauDP.gaudp.gaussian import build_gaussian_encoder, load_gaussian_checkpoint
import XPolicyLab.policy.GauDP.train_gaussian as gaussian_training


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--depth-weight", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--debug", action="store_true", help="evaluate only one validation batch")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("GAUDP_WANDB_MODE", "online"),
    )
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "MHBench-GauDP"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default="")
    args = parser.parse_args()
    if args.log_every < 0:
        parser.error("--log-every must be non-negative")

    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("NoPoSplat evaluation requires a CUDA device and CUDA rasterizer")

    val_data = GaussianFrameDataset(args.data, train=False)
    encoder = build_gaussian_encoder(len(val_data.camera_order))
    missing, unexpected = load_gaussian_checkpoint(encoder, args.checkpoint, strict=False)
    encoder.to(device).eval()
    encoder.requires_grad_(False)
    loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    total_batches = min(1, len(loader)) if args.debug else len(loader)
    print(
        f"[GauDP][gaussian-eval] checkpoint={args.checkpoint.resolve()} device={device} "
        f"cameras={val_data.camera_order} val_samples={len(val_data)} "
        f"val_batches={total_batches} batch_size={args.batch_size} workers={args.num_workers} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )

    sums = {"loss": 0.0, "rgb_loss": 0.0, "depth_loss": 0.0, "psnr": 0.0}
    count = 0
    started = time.monotonic()
    run_name = args.wandb_run_name or f"{args.output.parent.parent.name}-gaussian-eval-{args.output.name}"
    config = {**vars(args), "checkpoint": args.checkpoint.resolve(), "camera_order": val_data.camera_order}
    with ExperimentLogger(
        args.output,
        config=config,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_run_name=run_name,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_tags=parse_wandb_tags(args.wandb_tags),
    ) as logger:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                loss, batch_metrics = gaussian_training.reconstruction_loss(
                    encoder,
                    gaussian_training._to_device(batch, device),
                    global_step=0,
                    depth_weight=args.depth_weight,
                )
                sums["loss"] += float(loss)
                sums["rgb_loss"] += batch_metrics["rgb"]
                sums["depth_loss"] += batch_metrics["depth"]
                sums["psnr"] += batch_metrics["psnr"]
                count += 1
                completed = batch_index + 1
                if gaussian_training._should_log_batch(completed, total_batches, args.log_every):
                    progress = {
                        "record_type": "batch",
                        "val/batch_loss": float(loss),
                        "val/batch_rgb_loss": batch_metrics["rgb"],
                        "val/batch_depth_loss": batch_metrics["depth"],
                        "val/batch_psnr": batch_metrics["psnr"],
                        **gaussian_training._progress_metrics(completed, total_batches, started, "val"),
                    }
                    logger.log(progress, step=completed)
                    gaussian_training._print_batch_progress("gaussian-eval", 0, 1, "val", progress)
                if args.debug:
                    break

        metrics = {
            "record_type": "evaluation",
            "checkpoint": str(args.checkpoint.resolve()),
            "performance/eval_seconds": time.monotonic() - started,
            **{f"val/{key}": value / max(1, count) for key, value in sums.items()},
        }
        logger.log(metrics, step=count + 1)
        print(f"[GauDP][gaussian-eval] batches={count} {metrics}", flush=True)


if __name__ == "__main__":
    main()
