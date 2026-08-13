#!/usr/bin/env python3
"""Verify the GauDP policy environment after running install.sh."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-cuda-extensions",
        action="store_true",
        help="verify Python/data dependencies without cuRoPE or the renderer",
    )
    args = parser.parse_args()

    # Editable XPolicyLab should normally resolve this. Keeping the path based
    # on this file also makes diagnostics deterministic before editable metadata
    # is refreshed; it never points at any external policy implementation.
    xpl_root = Path(__file__).resolve().parents[2]
    if str(xpl_root) not in sys.path:
        sys.path.insert(0, str(xpl_root))

    import torch
    import torchvision

    modules = (
        "diffusers",
        "einops",
        "h5py",
        "jaxtyping",
        "beartype",
        "timm",
        "e3nn",
        "lpips",
        "cv2",
        "XPolicyLab.policy.GauDP.model",
    )
    for name in modules:
        importlib.import_module(name)

    if not args.skip_cuda_extensions:
        curope = importlib.import_module("curope")
        rasterizer = importlib.import_module("diff_gaussian_rasterization")
        importlib.import_module(
            "XPolicyLab.policy.GauDP.gaudp.third_party.noposplat.model.encoder.encoder_noposplat"
        )
        if not hasattr(curope, "rope_2d"):
            raise RuntimeError("curope extension imported but has no rope_2d kernel")
        if not hasattr(rasterizer, "GaussianRasterizer"):
            raise RuntimeError("Gaussian rasterizer extension is incomplete")

    print(f"[GauDP] torch={torch.__version__}, torch CUDA={torch.version.cuda}")
    print(f"[GauDP] torchvision={torchvision.__version__}")
    if not torch.cuda.is_available():
        print("[GauDP] WARNING: CUDA is not currently visible; training/inference was not GPU-tested.")
    print("[GauDP] dependency verification passed")


if __name__ == "__main__":
    main()
