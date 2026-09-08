#!/usr/bin/env python3
import argparse
import base64
import json
import time
import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, List
from urllib import error, request

import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.lerobot.lerobot_dataset import LeRobotDataset


def encode_image_b64(image: np.ndarray) -> str:
    image_uint8 = np.asarray(image, dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(image_uint8).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def post_json(url: str, payload: Dict) -> Dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"server returned HTTP {exc.code}: {detail}") from exc


def wait_for_server(url: str, retries: int, sleep_s: float) -> None:
    probe_payload = {
        "images_b64": [encode_image_b64(np.zeros((2, 2, 3), dtype=np.uint8))],
        "state": np.zeros((1, 36), dtype=np.float32).tolist(),
    }
    for attempt in range(retries):
        try:
            post_json(url, probe_payload)
            return
        except Exception:
            if attempt == retries - 1:
                raise
        time.sleep(sleep_s)


def action_labels(action_dim: int) -> List[str]:
    return [f"action_{idx:02d}" for idx in range(action_dim)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--server_url", default="http://127.0.0.1:8010/predict")
    parser.add_argument("--config_path", default="configs/hrdt_finetune_lerobot.yaml")
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--use_precomp_lang_embed", action="store_true", default=False)
    parser.add_argument("--rollout_stride", type=int, default=0)
    parser.add_argument("--wait_retries", type=int, default=30)
    parser.add_argument("--wait_seconds", type=float, default=2.0)
    args = parser.parse_args()

    if not args.server_url.endswith("/predict"):
        args.server_url = args.server_url.rstrip("/") + "/predict"

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset = LeRobotDataset(
        data_root=args.data_root,
        config=config,
        upsample_rate=1,
        val=False,
        use_precomp_lang_embed=args.use_precomp_lang_embed,
    )

    print(f"[test] Using server_url={args.server_url}")
    print(f"[test] Dataset size={len(dataset)}")
    wait_for_server(args.server_url, args.wait_retries, args.wait_seconds)

    total_abs = None
    total_sq = None
    total_count = 0
    labels = action_labels(int(config["common"]["action_dim"]))
    stride = args.rollout_stride if args.rollout_stride > 0 else 1

    num_samples = min(args.max_samples, len(dataset))
    for sample_idx in range(0, num_samples, stride):
        sample = dataset.get_item(sample_idx)
        if sample is None:
            continue

        images = sample["current_images"][0]
        payload = {
            "images_b64": [encode_image_b64(image) for image in images],
            "state": np.asarray(sample["states"], dtype=np.float32).tolist(),
        }
        if args.use_precomp_lang_embed and "lang_embeds" in sample:
            payload["lang_embeds"] = sample["lang_embeds"].tolist()

        response = post_json(args.server_url, payload)
        pred = np.asarray(response["action"], dtype=np.float32)
        gt = np.asarray(sample["actions"], dtype=np.float32)

        min_len = min(pred.shape[0], gt.shape[0])
        min_dim = min(pred.shape[1], gt.shape[1])
        pred = pred[:min_len, :min_dim]
        gt = gt[:min_len, :min_dim]
        err = pred - gt

        total_abs = np.abs(err).sum(axis=0) if total_abs is None else total_abs + np.abs(err).sum(axis=0)
        total_sq = (err ** 2).sum(axis=0) if total_sq is None else total_sq + (err ** 2).sum(axis=0)
        total_count += min_len

        print(f"[test] sample={sample_idx:03d} mean_l1={np.abs(err).mean():.6f} mse={np.mean(err ** 2):.6f}")

        if sample_idx == 0:
            step0_pred = pred[0]
            step0_gt = gt[0]
            step0_err = step0_pred - step0_gt
            for dim_idx, label in enumerate(labels[:min_dim]):
                print(
                    f"[test]   {label}: gt={step0_gt[dim_idx]:.6f} "
                    f"pred={step0_pred[dim_idx]:.6f} err={step0_err[dim_idx]:.6f}"
                )

    if total_abs is None or total_sq is None or total_count == 0:
        raise RuntimeError("No samples were evaluated.")

    avg_l1 = total_abs / total_count
    avg_mse = total_sq / total_count
    print("[test] avg_l1_per_dim=", avg_l1.tolist())
    print("[test] avg_l1_mean=", float(avg_l1.mean()))
    print("[test] avg_mse_per_dim=", avg_mse.tolist())
    print("[test] avg_mse_mean=", float(avg_mse.mean()))


if __name__ == "__main__":
    main()
