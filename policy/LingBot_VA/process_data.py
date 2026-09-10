#!/usr/bin/env python3
"""LingBot_VA data processing.

Turns a standard RoboDojo LeRobot v2.1 dataset (parquet + per-episode mp4) into a
LingBot-VA training dataset by running every step of the upstream pipeline
(`lingbot_va/README.md` -> Post-Training / Custom Dataset Preparation):

  1. Convert actions into the target layout: the 30-dim LingBot one for
     RoboDojo, or MHBench's own 35-dim action passed through unchanged
     (`--layout mhbench`, see baselines/scripts/README.md).
  2. Add `action_config` to `meta/episodes.jsonl` (one segment per episode).
  3. Extract Wan2.2 VAE video latents into `latents/` for every configured camera.
  4. Encode `empty_emb.pt` (empty-string text embedding) at the dataset root.

The output is a self-contained LeRobot v2.1 dataset the trainer can load directly.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

POLICY_DIR = Path(__file__).resolve().parent
LINGBOT_ROOT = POLICY_DIR / "lingbot_va"
sys.path.insert(0, str(LINGBOT_ROOT / "wan_va"))

from diffusers.pipelines.wan.pipeline_wan import prompt_clean  # noqa: E402
from modules.utils import (  # noqa: E402
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_vae,
)

# 30-dim LingBot action layout (see va_robotwin30_train_cfg / reference dataset).
ACTION_NAMES_30 = [
    "left_x", "left_y", "left_z", "left_rx", "left_ry", "left_rz", "left_w",
    "right_x", "right_y", "right_z", "right_rx", "right_ry", "right_rz", "right_w",
    "left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3",
    "left_joint_4", "left_joint_5", "left_joint_6",
    "right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3",
    "right_joint_4", "right_joint_5", "right_joint_6",
    "left_gripper", "right_gripper",
]

# Cameras encoded to latents (must match the training config's obs_cam_keys).
LATENT_CAM_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]

# MHBench's decentralized action, passed through as-is: 31 joint targets, the
# pelvis height and the base velocity. `prepare_lingbot_init_ckpt.py` widens the
# model's two action projections to match.
ACTION_DIM_MHBENCH = 35
MHBENCH_CAM_KEYS = ["observation.images.ego"]

TEXT_MAX_LEN = 512
VAE_TEMPORAL_RATE = 4  # Wan2.2 causal VAE temporal compression.


def map_action_14_to_30(arr14: np.ndarray) -> np.ndarray:
    """Map RoboDojo 14-dim [6 joints + gripper] x2 into the 30-dim LingBot layout.

    EEF poses (dims 0-13) and the 7th joint per arm are unavailable in joint-only
    data and are zero-padded, as sanctioned by the upstream README.
    """
    n = arr14.shape[0]
    out = np.zeros((n, 30), dtype=np.float32)
    out[:, 14:20] = arr14[:, 0:6]    # left_joint_0..5
    out[:, 21:27] = arr14[:, 7:13]   # right_joint_0..5
    out[:, 28] = arr14[:, 6]         # left_gripper
    out[:, 29] = arr14[:, 13]        # right_gripper
    return out


def read_video_frames(mp4_path: Path) -> np.ndarray:
    container = av.open(str(mp4_path))
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    return np.stack(frames)


def sample_frame_ids(num_frames: int, stride: int) -> list:
    """Sample every `stride`-th frame, then truncate so (n-1) % temporal_rate == 0."""
    ids = list(range(0, num_frames, stride))
    n = len(ids)
    n = ((n - 1) // VAE_TEMPORAL_RATE) * VAE_TEMPORAL_RATE + 1
    return ids[:n]


@torch.no_grad()
def encode_video_latent(frames_rgb: np.ndarray, vae, wrapper, device, dtype, size):
    """frames_rgb: (n, H, W, 3) uint8 -> normalized VAE latent, plus latent dims.

    `size` is (height, width); both must be multiples of 32 so the latent grid
    (VAE stride 16) is even and the model's 2x2 patching divides it.
    """
    x = torch.from_numpy(frames_rgb).float().permute(3, 0, 1, 2)  # (3, n, H, W)
    x = F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False)
    x = (x / 255.0 * 2.0 - 1.0).unsqueeze(0).to(device).to(dtype)  # (1, 3, n, H, W)

    wrapper.clear_cache()
    outs, t, first = [], 0, True
    while t < x.shape[2]:
        step = 1 if first else VAE_TEMPORAL_RATE
        chunk = x[:, :, t:t + step]
        if chunk.shape[2] == 0:
            break
        outs.append(wrapper.encode_chunk(chunk))
        t += step
        first = False
    enc = torch.cat(outs, dim=2)
    mu, _ = torch.chunk(enc, 2, dim=1)  # (1, C, F, H, W)

    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(mu)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(mu)
    mu_norm = (mu.float() - mean) * (1.0 / std)

    _, c, f, h, w = mu_norm.shape
    flat = mu_norm[0].permute(1, 2, 3, 0).reshape(f * h * w, c).to(torch.bfloat16).cpu()
    return flat, f, h, w


@torch.no_grad()
def encode_text(prompt: str, tokenizer, text_encoder):
    p = [prompt_clean(prompt)]
    ti = tokenizer(
        p, padding="max_length", max_length=TEXT_MAX_LEN, truncation=True,
        add_special_tokens=True, return_attention_mask=True, return_tensors="pt",
    )
    ids, mask = ti.input_ids, ti.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    dev = next(text_encoder.parameters()).device
    emb = text_encoder(ids.to(dev), mask.to(dev)).last_hidden_state.to(torch.bfloat16)
    emb = [u[:v] for u, v in zip(emb, seq_lens)]
    emb = torch.stack(
        [torch.cat([u, u.new_zeros(TEXT_MAX_LEN - u.size(0), u.size(1))]) for u in emb],
        dim=0,
    )
    return emb[0].cpu()


def compute_column_stats(arr: np.ndarray) -> dict:
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def quantile_norm_stat(arr: np.ndarray, floor: float = 1e-3) -> dict:
    """Per-dimension q01/q99, the normalisation the model's action head expects.

    A dimension that never moves would give q99 == q01 and a divide-by-nothing;
    it is widened to `floor` so the normalised value stays 0 rather than -1.
    """
    q01 = np.quantile(arr, 0.01, axis=0)
    q99 = np.quantile(arr, 0.99, axis=0)
    flat = (q99 - q01) < floor
    mid = (q99 + q01) / 2.0
    q01 = np.where(flat, mid - floor / 2, q01)
    q99 = np.where(flat, mid + floor / 2, q99)
    return {"q01": q01.astype(float).tolist(), "q99": q99.astype(float).tolist(),
            "flat_dims": np.flatnonzero(flat).tolist()}


def source_video_path(src: Path, info: dict, cam: str, ep_idx: int, ep_chunk: int) -> Path:
    rel = info["video_path"].format(
        episode_chunk=ep_chunk, video_key=cam, episode_index=ep_idx,
    )
    return src / rel


def main():
    ap = argparse.ArgumentParser(description="LingBot_VA data processing (latent extraction).")
    ap.add_argument("--source-dataset", required=True, help="Source LeRobot v2.1 dataset dir.")
    ap.add_argument("--output-dataset", required=True, help="Output LingBot-VA dataset dir.")
    ap.add_argument("--base-model", required=True, help="lingbot-va-base weights dir (vae/tokenizer/text_encoder).")
    ap.add_argument("--num-episodes", type=int, default=0, help="Cap episodes (0 = all).")
    ap.add_argument("--target-fps", type=int, default=10, help="Target sampling fps for latents.")
    ap.add_argument("--image-size", default="256x256",
                    help="VAE input resolution, HxW or a single square edge. "
                         "Both sides must be multiples of 32.")
    ap.add_argument("--layout", choices=("robodojo", "mhbench"), default="robodojo",
                    help="robodojo: map 14D joints into the 30D LingBot layout. "
                         "mhbench: pass MHBench's 35D action through unchanged.")
    ap.add_argument("--cam-keys", default="",
                    help="Comma-separated video keys to encode; defaults to the layout's.")
    ap.add_argument("--device", default="cuda", help="Torch device.")
    args = ap.parse_args()

    if "x" in str(args.image_size):
        size_hw = tuple(int(v) for v in str(args.image_size).split("x"))
    else:
        size_hw = (int(args.image_size), int(args.image_size))
    if any(v % 32 for v in size_hw):
        raise SystemExit(f"--image-size {size_hw} is not a multiple of 32 on both sides")
    cam_keys = ([k for k in args.cam_keys.split(",") if k] or
                (MHBENCH_CAM_KEYS if args.layout == "mhbench" else LATENT_CAM_KEYS))
    action_dim = ACTION_DIM_MHBENCH if args.layout == "mhbench" else 30

    src = Path(args.source_dataset).resolve()
    out = Path(args.output_dataset).resolve()
    base = Path(args.base_model).resolve()
    device = args.device
    dtype = torch.bfloat16

    info = json.loads((src / "meta" / "info.json").read_text())
    # The output is always v2.1, whatever the source is: LatentLeRobotDataset
    # pins revision "v2.1" and reads per-episode statistics from
    # meta/episodes_stats.jsonl, which v2.0 does not have (it carries one
    # meta/stats.json for the whole dataset). MHBench's own export is v2.0, so
    # the per-episode entries below are computed here rather than copied.
    src_version = info.get("codebase_version")
    if src_version not in ("v2.0", "v2.1"):
        print(f"[process_data] WARNING: source codebase {src_version} is neither v2.0 nor "
              "v2.1; the LeRobot layout this reads may not match.", flush=True)
    ori_fps = int(info["fps"])
    chunks_size = int(info.get("chunks_size", 1000))
    stride = max(1, round(ori_fps / args.target_fps))

    episodes = [json.loads(l) for l in (src / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    ep_stats_file = src / "meta" / "episodes_stats.jsonl"
    ep_stats = {}
    if ep_stats_file.exists():
        ep_stats = {json.loads(l)["episode_index"]: json.loads(l)
                    for l in ep_stats_file.read_text().splitlines() if l.strip()}
    if args.num_episodes > 0:
        episodes = episodes[:args.num_episodes]

    print(f"[process_data] source={src}", flush=True)
    print(f"[process_data] output={out}", flush=True)
    print(f"[process_data] episodes={len(episodes)} ori_fps={ori_fps} stride={stride} "
          f"target_fps={args.target_fps} size={size_hw[0]}x{size_hw[1]}", flush=True)
    print(f"[process_data] layout={args.layout} action_dim={action_dim} "
          f"cams={cam_keys}", flush=True)

    (out / "meta").mkdir(parents=True, exist_ok=True)

    print("[process_data] loading Wan2.2 VAE + text encoder ...", flush=True)
    vae = load_vae(str(base / "vae"), dtype, device)
    wrapper = WanVAEStreamingWrapper(vae)
    tokenizer = load_tokenizer(str(base / "tokenizer"))
    text_encoder = load_text_encoder(str(base / "text_encoder"), dtype, device)

    new_episodes, new_ep_stats = [], []
    total_frames = 0
    all_actions = []

    for ep in episodes:
        ep_idx = ep["episode_index"]
        ep_chunk = ep_idx // chunks_size
        length = ep["length"]
        task = ep["tasks"][0] if ep.get("tasks") else ""
        chunk_dir = f"chunk-{ep_chunk:03d}"

        # --- Step 1: rewrite parquet action/state to 30-dim -------------------
        src_pq = src / "data" / chunk_dir / f"episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(src_pq)
        act_src = np.stack(df["action"].to_numpy()).astype(np.float32)
        st_src = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
        if args.layout == "mhbench":
            if act_src.shape[1] != action_dim:
                raise SystemExit(
                    f"episode {ep_idx}: action is {act_src.shape[1]}D, expected {action_dim}")
            act30, st30 = act_src, st_src
        else:
            act30 = map_action_14_to_30(act_src)
            st30 = map_action_14_to_30(st_src)
            df["action"] = list(act30)
            df["observation.state"] = list(st30)
        all_actions.append(act30)
        out_pq = out / "data" / chunk_dir / f"episode_{ep_idx:06d}.parquet"
        out_pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_pq, index=False)
        total_frames += len(df)

        # --- Step 2: action_config (single segment per episode) ---------------
        ep_out = dict(ep)
        ep_out["action_config"] = [{
            "start_frame": 0, "end_frame": length, "action_text": task,
        }]
        new_episodes.append(ep_out)

        # updated per-episode stats for the rewritten columns
        st_entry = ep_stats.get(ep_idx, {"episode_index": ep_idx, "stats": {}})
        stats = dict(st_entry["stats"])
        stats["action"] = compute_column_stats(act30)
        stats["observation.state"] = compute_column_stats(st30)
        # A v2.0 source brings no per-episode entry at all, so fill in every
        # numeric column the parquet carries.
        for column in df.columns:
            if column in stats or column.startswith("observation.images"):
                continue
            values = np.asarray(df[column].to_list())
            if values.ndim == 1:
                values = values[:, None]
            if not np.issubdtype(values.dtype, np.number):
                continue
            stats[column] = compute_column_stats(values.astype(np.float64))
        new_ep_stats.append({"episode_index": ep_idx, "stats": stats})

        # --- Step 3: text embedding + video latents per camera ----------------
        text_emb = encode_text(task, tokenizer, text_encoder)
        for cam in cam_keys:
            mp4 = source_video_path(src, info, cam, ep_idx, ep_chunk)
            if not mp4.exists():
                raise FileNotFoundError(f"missing source video: {mp4}")
            frames = read_video_frames(mp4)
            n_avail = min(len(frames), length)
            frame_ids = sample_frame_ids(n_avail, stride)
            clip = frames[frame_ids]
            latent, f_lat, h_lat, w_lat = encode_video_latent(
                clip, vae, wrapper, device, dtype, size_hw)

            rec = {
                "latent": latent,
                "latent_num_frames": f_lat,
                "latent_height": h_lat,
                "latent_width": w_lat,
                "video_num_frames": len(frame_ids),
                "video_height": size_hw[0],
                "video_width": size_hw[1],
                "text_emb": text_emb,
                "text": task,
                "frame_ids": frame_ids,
                "start_frame": 0,
                "end_frame": length,
                "fps": args.target_fps,
                "ori_fps": ori_fps,
            }
            lat_dir = out / "latents" / chunk_dir / cam
            lat_dir.mkdir(parents=True, exist_ok=True)
            torch.save(rec, lat_dir / f"episode_{ep_idx:06d}_0_{length}.pth")

        print(f"[process_data]   episode {ep_idx}: length={length} "
              f"latent_frames={f_lat} cams={len(cam_keys)}", flush=True)

    # --- Step 4: empty_emb.pt -------------------------------------------------
    torch.save(encode_text("", tokenizer, text_encoder), out / "empty_emb.pt")

    # --- Step 5: the action normalisation the training config reads ------------
    norm_stat = quantile_norm_stat(np.concatenate(all_actions, axis=0))
    (out / "meta" / "lingbot_norm_stat.json").write_text(json.dumps(norm_stat, indent=2))
    if norm_stat["flat_dims"]:
        print(f"[process_data] action dims that never move: {norm_stat['flat_dims']}",
              flush=True)

    # --- meta files -----------------------------------------------------------
    (out / "meta" / "episodes.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in new_episodes) + "\n")
    (out / "meta" / "episodes_stats.jsonl").write_text(
        "\n".join(json.dumps(e) for e in new_ep_stats) + "\n")
    # tasks.jsonl: copy source
    (out / "meta" / "tasks.jsonl").write_text((src / "meta" / "tasks.jsonl").read_text())

    # info.json: 30-dim action/state, trimmed episode/frame counts
    out_info = dict(info)
    out_info["codebase_version"] = "v2.1"
    for key in ("action", "observation.state"):
        feat = dict(info["features"][key])
        feat["shape"] = [action_dim]
        if args.layout != "mhbench":
            feat["names"] = list(ACTION_NAMES_30)
        out_info["features"][key] = feat
    out_info["total_episodes"] = len(new_episodes)
    out_info["total_frames"] = total_frames
    out_info["total_chunks"] = (len(new_episodes) - 1) // chunks_size + 1
    out_info["total_videos"] = len(new_episodes) * sum(
        1 for k, v in info["features"].items() if v.get("dtype") == "video")
    out_info["splits"] = {"train": f"0:{len(new_episodes)}"}
    (out / "meta" / "info.json").write_text(json.dumps(out_info, indent=4))

    # symlink source videos so the dataset is a valid LeRobot layout
    for ep in new_episodes:
        ep_idx = ep["episode_index"]
        ep_chunk = ep_idx // chunks_size
        for cam, feat in info["features"].items():
            if feat.get("dtype") != "video":
                continue
            src_v = source_video_path(src, info, cam, ep_idx, ep_chunk)
            if not src_v.exists():
                continue
            dst_v = out / info["video_path"].format(
                episode_chunk=ep_chunk, video_key=cam, episode_index=ep_idx)
            dst_v.parent.mkdir(parents=True, exist_ok=True)
            if dst_v.is_symlink() or dst_v.exists():
                dst_v.unlink()
            os.symlink(src_v, dst_v)

    print(f"[process_data] done. dataset ready at: {out}", flush=True)
    print(f"[process_data] set LINGBOT_VA_DATASET_PATH={out}", flush=True)


if __name__ == "__main__":
    main()
