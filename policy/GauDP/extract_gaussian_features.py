#!/usr/bin/env python3
"""Precompute frozen NoPoSplat features for GauDP policy training."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_XPL_ROOT = Path(__file__).resolve().parents[2]
if str(_XPL_ROOT) not in sys.path:
    sys.path.insert(0, str(_XPL_ROOT))

from XPolicyLab.policy.GauDP.gaudp.dataset import GaussianImageDataset, IMAGE_SIZE
from XPolicyLab.policy.GauDP.gaudp.gaussian import (
    build_gaussian_encoder,
    encode_gaussians,
    load_gaussian_checkpoint,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _validate_existing(
    output: Path,
    *,
    shape: tuple[int, ...],
    camera_order: list[str],
    checkpoint: Path,
) -> tuple[h5py.File, int]:
    destination = h5py.File(output, "r+")
    try:
        if "gaussian_features" not in destination:
            raise ValueError("missing gaussian_features dataset")
        if tuple(destination["gaussian_features"].shape) != shape:
            raise ValueError(
                f"shape mismatch: expected {shape}, got {tuple(destination['gaussian_features'].shape)}"
            )
        if json.loads(destination.attrs["camera_order"]) != camera_order:
            raise ValueError("camera order mismatch")
        recorded = Path(str(destination.attrs.get("gaussian_checkpoint", ""))).resolve()
        if recorded != checkpoint:
            raise ValueError(f"checkpoint mismatch: cache uses {recorded}, requested {checkpoint}")
        completed = int(destination.attrs.get("completed_frames", 0))
        if not 0 <= completed <= shape[0]:
            raise ValueError(f"invalid completed_frames={completed}")
        return destination, completed
    except Exception:
        destination.close()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--compression", choices=("none", "lzf", "gzip"), default="none")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing feature cache instead of resuming it",
    )
    parser.add_argument("--debug", action="store_true", help="extract one batch without marking the cache complete")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.log_every < 0:
        parser.error("--log-every must be non-negative")

    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("Gaussian feature extraction requires a CUDA device")

    data_path = args.data.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = GaussianImageDataset(data_path)
    shape = (len(dataset), len(dataset.camera_order), 13, *IMAGE_SIZE)
    numpy_dtype = np.float16 if args.dtype == "float16" else np.float32
    estimated_bytes = int(np.prod(shape)) * np.dtype(numpy_dtype).itemsize

    if output.exists() and args.overwrite:
        output.unlink()
    if output.exists():
        destination, completed = _validate_existing(
            output,
            shape=shape,
            camera_order=dataset.camera_order,
            checkpoint=checkpoint,
        )
        if bool(destination.attrs.get("complete", False)):
            destination.close()
            print(f"[GauDP][feature-extract] reusing complete cache: {output}", flush=True)
            return
        print(f"[GauDP][feature-extract] resuming at frame {completed}/{len(dataset)}", flush=True)
    else:
        free_bytes = shutil.disk_usage(output.parent).free
        if free_bytes < estimated_bytes:
            raise SystemExit(
                f"insufficient free space for uncompressed cache: need about "
                f"{estimated_bytes / 1024**3:.1f} GiB, have {free_bytes / 1024**3:.1f} GiB"
            )
        compression = None if args.compression == "none" else args.compression
        destination = h5py.File(output, "w", libver="latest")
        destination.create_dataset(
            "gaussian_features",
            shape=shape,
            dtype=numpy_dtype,
            chunks=(1, *shape[1:]),
            compression=compression,
            compression_opts=4 if compression == "gzip" else None,
        )
        destination.attrs["format"] = "mhbench-gaudp-features-v1"
        destination.attrs["source_data"] = str(data_path)
        destination.attrs["gaussian_checkpoint"] = str(checkpoint)
        destination.attrs["camera_order"] = json.dumps(dataset.camera_order)
        destination.attrs["completed_frames"] = 0
        destination.attrs["complete"] = False
        destination.flush()
        completed = 0

    encoder = build_gaussian_encoder(len(dataset.camera_order))
    missing, unexpected = load_gaussian_checkpoint(encoder, checkpoint, strict=False)
    encoder.to(device).eval().requires_grad_(False)
    remaining = Subset(dataset, range(completed, len(dataset)))
    loader = DataLoader(
        remaining,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    total_batches = len(loader)
    started = time.monotonic()
    print(
        f"[GauDP][feature-extract] checkpoint={checkpoint} output={output} "
        f"frames={len(dataset)} remaining={len(remaining)} cameras={dataset.camera_order} "
        f"dtype={args.dtype} estimated={estimated_bytes / 1024**3:.1f}GiB "
        f"batch_size={args.batch_size} workers={args.num_workers} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )

    try:
        with torch.no_grad():
            for batch_index, images in enumerate(loader):
                images = images.to(device, non_blocking=True)
                _, features = encode_gaussians(encoder, images, return_features=True)
                values = features.to(dtype=torch.float16 if args.dtype == "float16" else torch.float32)
                values = values.cpu().numpy()
                start = completed
                completed += values.shape[0]
                destination["gaussian_features"][start:completed] = values
                should_log = (
                    args.log_every > 0
                    and (batch_index == 0 or batch_index + 1 == total_batches or (batch_index + 1) % args.log_every == 0)
                )
                if should_log:
                    destination.attrs["completed_frames"] = completed
                    destination.flush()
                    elapsed = max(time.monotonic() - started, 1e-6)
                    rate = (completed - int(remaining.indices.start)) / elapsed
                    eta = (len(dataset) - completed) / max(rate, 1e-6)
                    print(
                        f"[GauDP][feature-extract] frames={completed}/{len(dataset)} "
                        f"({100.0 * completed / len(dataset):.1f}%) rate={rate:.2f} frame/s "
                        f"eta={_format_duration(eta)}",
                        flush=True,
                    )
                if args.debug:
                    break
        destination.attrs["completed_frames"] = completed
        if completed == len(dataset) and not args.debug:
            destination.attrs["complete"] = True
        destination.flush()
    finally:
        destination.close()

    if args.debug:
        print("[GauDP][feature-extract] debug extraction stopped; rerun without --debug to resume", flush=True)
    else:
        print(
            f"[GauDP][feature-extract] complete output={output} elapsed={_format_duration(time.monotonic() - started)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
