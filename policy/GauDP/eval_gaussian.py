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
from torch.utils.data import DataLoader, Subset

_XPL_ROOT = Path(__file__).resolve().parents[2]
if str(_XPL_ROOT) not in sys.path:
    sys.path.insert(0, str(_XPL_ROOT))

from XPolicyLab.policy.GauDP.gaudp import recon_dump
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
    # --- reconstruction dump -------------------------------------------------
    parser.add_argument(
        "--dump-recon",
        action="store_true",
        help="write a GT/reconstruction grid for a few frames per validation episode",
    )
    parser.add_argument(
        "--dump-fractions",
        default=",".join(str(fraction) for fraction in recon_dump.DEFAULT_FRACTIONS),
        help="where in each episode to dump, as fractions in [0, 1). Episode starts "
             "alone would be misleading: every episode begins from the same reset "
             "pose, with no contact and no occlusion to reconstruct.",
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        help="destination for the PNGs and manifest.jsonl; defaults to <output>/recon",
    )
    parser.add_argument(
        "--dump-only",
        action="store_true",
        help="render only the selected keyframes instead of the whole validation "
             "split. Implies --dump-recon, and reports its metrics as keyframe/*",
    )
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
    if args.dump_only:
        args.dump_recon = True
    try:
        dump_fractions = recon_dump.parse_fractions(args.dump_fractions)
    except ValueError as error:
        parser.error(f"--dump-fractions: {error}")

    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("NoPoSplat evaluation requires a CUDA device and CUDA rasterizer")

    val_data = GaussianFrameDataset(args.data, train=False)
    keyframes = (
        recon_dump.select_keyframes(val_data.episode_ranges, dump_fractions, val_data.episode_ids)
        if args.dump_recon
        else []
    )
    dump_targets = {keyframe.position: keyframe for keyframe in keyframes}
    dump_dir = (args.dump_dir or args.output / "recon").expanduser()
    encoder = build_gaussian_encoder(len(val_data.camera_order))
    missing, unexpected = load_gaussian_checkpoint(encoder, args.checkpoint, strict=False)
    encoder.to(device).eval()
    encoder.requires_grad_(False)
    # --dump-only narrows the pass to the keyframes. Their scores are then not
    # the validation split's, so they are logged as `keyframe/*` rather than
    # silently replacing `val/*` with a 40-frame subset of it.
    order = sorted(dump_targets) if args.dump_only else list(range(len(val_data)))
    phase = "keyframe" if args.dump_only else "val"
    loader = DataLoader(
        Subset(val_data, order) if args.dump_only else val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    total_batches = min(1, len(loader)) if args.debug else len(loader)
    print(
        f"[GauDP][gaussian-eval] checkpoint={args.checkpoint.resolve()} device={device} "
        f"cameras={val_data.camera_order} val_samples={len(val_data)} "
        f"{phase}_batches={total_batches} batch_size={args.batch_size} workers={args.num_workers} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if args.dump_recon:
        print(
            f"[GauDP][gaussian-eval] dumping {len(keyframes)} keyframes from "
            f"{len(val_data.episode_ids)} validation episodes at fractions {dump_fractions} "
            f"into {dump_dir} (columns: {' | '.join(recon_dump.COLUMNS)}, rows: {val_data.camera_order})",
            flush=True,
        )

    sums = {"loss": 0.0, "rgb_loss": 0.0, "depth_loss": 0.0, "psnr": 0.0}
    manifest: list[dict] = []
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
                device_batch = gaussian_training._to_device(batch, device)
                loss, batch_metrics, rendered_rgb, rendered_depth = gaussian_training.reconstruction_outputs(
                    encoder,
                    device_batch,
                    global_step=0,
                    depth_weight=args.depth_weight,
                )
                if dump_targets:
                    # shuffle=False, so the batch's n-th sample is the n-th
                    # entry of `order` after the batches already consumed.
                    base = batch_index * args.batch_size
                    for offset in range(device_batch["images"].shape[0]):
                        keyframe = dump_targets.get(order[base + offset])
                        if keyframe is None:
                            continue
                        manifest.append(
                            recon_dump.save_keyframe(
                                dump_dir,
                                keyframe,
                                images=device_batch["images"][offset].float().cpu().numpy(),
                                depth=device_batch["depth"][offset].float().cpu().numpy(),
                                rendered_rgb=rendered_rgb[offset].float().cpu().numpy(),
                                rendered_depth=rendered_depth[offset].float().cpu().numpy(),
                                near=float(device_batch["near"][offset][0]),
                                far=float(device_batch["far"][offset][0]),
                            )
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
                        f"{phase}/batch_loss": float(loss),
                        f"{phase}/batch_rgb_loss": batch_metrics["rgb"],
                        f"{phase}/batch_depth_loss": batch_metrics["depth"],
                        f"{phase}/batch_psnr": batch_metrics["psnr"],
                        **gaussian_training._progress_metrics(completed, total_batches, started, phase),
                    }
                    logger.log(progress, step=completed)
                    gaussian_training._print_batch_progress("gaussian-eval", 0, 1, phase, progress)
                if args.debug:
                    break

        metrics = {
            "record_type": "evaluation",
            "checkpoint": str(args.checkpoint.resolve()),
            "performance/eval_seconds": time.monotonic() - started,
            **{f"{phase}/{key}": value / max(1, count) for key, value in sums.items()},
        }
        logger.log(metrics, step=count + 1)
        print(f"[GauDP][gaussian-eval] batches={count} {metrics}", flush=True)
        if manifest:
            manifest_path = recon_dump.write_manifest(dump_dir, manifest)
            worst = sorted(manifest, key=lambda record: record["psnr"])[:5]
            print(
                f"[GauDP][gaussian-eval] dumped {len(manifest)} keyframes to {dump_dir} "
                f"manifest={manifest_path}",
                flush=True,
            )
            print(
                "[GauDP][gaussian-eval] worst keyframes by PSNR: "
                + ", ".join(f"{record['file']}={record['psnr']:.2f}" for record in worst),
                flush=True,
            )


if __name__ == "__main__":
    main()
