import argparse
import base64
import glob
import io
import json
import os
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from urllib import request

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset_mod
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as NativeLeRobotDataset


def _patch_native_lerobot_dataset() -> None:
    if getattr(NativeLeRobotDataset, "_egovla_hf_column_patch", False):
        return

    def __init__(
        self,
        repo_id,
        root=None,
        episodes=None,
        image_transforms=None,
        delta_timestamps=None,
        tolerance_s=1e-4,
        revision=None,
        force_cache_sync=False,
        download_videos=True,
        video_backend=None,
    ):
        self.repo_id = repo_id
        self.root = Path(root) if root else lerobot_dataset_mod.HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else lerobot_dataset_mod.CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else lerobot_dataset_mod.get_safe_default_codec()
        self.delta_indices = None

        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        self.meta = lerobot_dataset_mod.LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        if self.episodes is not None and self.meta._version >= lerobot_dataset_mod.packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = lerobot_dataset_mod.aggregate_stats(episodes_stats)

        try:
            if force_cache_sync:
                raise FileNotFoundError
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = lerobot_dataset_mod.get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = lerobot_dataset_mod.get_episode_data_index(self.meta.episodes, self.episodes)

        timestamps = np.asarray(self.hf_dataset["timestamp"], dtype=np.float32)
        episode_indices = np.asarray(self.hf_dataset["episode_index"], dtype=np.int64)
        ep_data_index_np = {
            k: t.numpy() if hasattr(t, "numpy") else np.asarray(t) for k, t in self.episode_data_index.items()
        }
        lerobot_dataset_mod.check_timestamps_sync(
            timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s
        )

        if self.delta_timestamps is not None:
            lerobot_dataset_mod.check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = lerobot_dataset_mod.get_delta_indices(self.delta_timestamps, self.fps)

    NativeLeRobotDataset.__init__ = __init__
    NativeLeRobotDataset._egovla_hf_column_patch = True


_patch_native_lerobot_dataset()


def load_task_descriptions(task_path: str) -> Dict[int, str]:
    tasks_path = os.path.join(task_path, "meta", "tasks.jsonl")
    desc = {}
    if not os.path.isfile(tasks_path):
        return desc
    with open(tasks_path, "r") as f:
        for line in f:
            item = json.loads(line)
            desc[item["task_index"]] = item.get("description", item.get("task", ""))
    return desc


def load_modality_slices(task_path: str):
    modality_path = os.path.join(task_path, "meta", "modality.json")
    if not os.path.isfile(modality_path):
        return None, None
    with open(modality_path, "r") as f:
        modality = json.load(f)
    return (
        get_modality_slices(modality, "state", expected_key="states"),
        get_modality_slices(modality, "action", expected_key="action"),
    )


def get_modality_slices(modality: Dict, section: str, expected_key: str | None = None):
    if not modality:
        return None
    entries = modality.get(section, {})
    if not entries:
        return None
    if expected_key is not None:
        original_keys = {meta.get("original_key") for meta in entries.values()}
        if original_keys != {expected_key}:
            return None
    slices = []
    for _, meta in entries.items():
        start = int(meta["start"])
        end = int(meta["end"])
        slices.append((start, end))
    return slices


def reorder_by_slices(vec: np.ndarray, slices):
    if slices is None:
        return vec
    parts = [vec[start:end] for start, end in slices]
    return np.concatenate(parts, axis=0) if parts else vec


def encode_image_b64(image: np.ndarray) -> str:
    img = Image.fromarray(image)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def iter_episodes(
    data_root: str,
    task_dir_filter: str | None = None,
    episode_idx_filter: int | None = None,
):
    task_dirs = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )
    for task_dir in task_dirs:
        if task_dir_filter and task_dir != task_dir_filter:
            continue
        task_path = os.path.join(data_root, task_dir)
        info_path = os.path.join(task_path, "meta", "info.json")
        if not os.path.isfile(info_path):
            continue
        with open(info_path, "r") as f:
            info = json.load(f)
        native_dataset = NativeLeRobotDataset(
            repo_id=task_dir,
            root=task_path,
            download_videos=False,
            video_backend=os.environ.get("LEROBOT_VIDEO_BACKEND"),
        )
        task_desc = load_task_descriptions(task_path)
        state_slices, action_slices = load_modality_slices(task_path)
        data_glob = os.path.join(task_path, "data", "chunk-*", "episode_*.parquet")
        for parquet_path in sorted(glob.glob(data_glob)):
            episode_index = int(re.search(r"episode_(\d+)\.parquet$", parquet_path).group(1))
            chunk_match = re.search(r"chunk-(\d+)", parquet_path)
            episode_chunk = int(chunk_match.group(1)) if chunk_match else 0
            if episode_idx_filter is not None and episode_index != episode_idx_filter:
                continue
            video_rel = info["video_path"].format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
            )
            video_path = os.path.join(task_path, video_rel)
            yield parquet_path, video_path, task_desc, native_dataset, episode_index, state_slices, action_slices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/hfm/data/real_teleop_g1/lerobot")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8000/predict")
    parser.add_argument("--future_index", type=int, default=0)
    parser.add_argument("--predict_future_step", type=int, default=30)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--use_norm_stats", action="store_true", help="Report normalized errors using norm_stats.json")
    parser.add_argument("--task_dir", type=str, default=None, help="Restrict to a single task directory")
    parser.add_argument("--rollout_stride", type=int, default=0, help="If >0, evaluate every N steps and compare to action at t+N")
    parser.add_argument("--episode_idx", type=int, default=None, help="Evaluate a specific episode index")
    parser.add_argument("--frame_idx", type=int, default=None, help="Evaluate a specific frame index within the episode")
    parser.add_argument("--episode_mean", action="store_true", help="Report mean MSE for each selected episode")
    args = parser.parse_args()
    if not args.server_url.endswith("/predict"):
        args.server_url = args.server_url.rstrip("/") + "/predict"
    print(f"[test] Using server_url={args.server_url}")

    total = 0
    mse_sum = 0.0
    mse_norm_sum = 0.0
    printed_first = False

    for parquet_path, video_path, task_desc, native_dataset, episode_index, state_slices, action_slices in iter_episodes(
        args.data_root, args.task_dir, args.episode_idx
    ):
        table = pq.read_table(
            parquet_path,
            columns=["states", "action", "frame_index", "task_index", "next.done"],
        )
        states = table.column("states").to_pylist()
        actions = table.column("action").to_pylist()
        frame_index = table.column("frame_index").to_pylist()
        task_index = table.column("task_index").to_pylist()
        done_flags = table.column("next.done").to_pylist()

        ep_start = native_dataset.episode_data_index["from"][episode_index].item()
        ep_end = native_dataset.episode_data_index["to"][episode_index].item()
        norm_mean = None
        norm_std = None
        if args.use_norm_stats:
            task_root = os.path.dirname(os.path.dirname(os.path.dirname(parquet_path)))
            norm_path = os.path.join(task_root, "norm_stats.json")
            if os.path.isfile(norm_path):
                with open(norm_path, "r") as f:
                    norm_stats = json.load(f).get("norm_stats", {})
                actions_stats = norm_stats.get("actions", {})
                norm_mean = np.array(actions_stats.get("mean", []), dtype=np.float32)
                norm_std = np.array(actions_stats.get("std", []), dtype=np.float32)
                if norm_mean.size and norm_std.size:
                    # Match the same action ordering used for gt/pred.
                    norm_mean = reorder_by_slices(norm_mean, action_slices)
                    norm_std = reorder_by_slices(norm_std, action_slices)
        step_indices = range(len(states))
        if args.rollout_stride and args.rollout_stride > 0:
            step_indices = range(0, len(states), args.rollout_stride)
        if args.frame_idx is not None:
            if args.frame_idx in frame_index:
                step_indices = [frame_index.index(args.frame_idx)]
            else:
                raise ValueError(f"frame_idx {args.frame_idx} not found in episode {episode_index}")
        ep_mse_sum = 0.0
        ep_count = 0
        for i in step_indices:
            if total >= args.max_samples:
                break
            fut_start = i + args.future_index
            if fut_start >= len(actions):
                fut_start = len(actions) - 1
            fut_end = min(fut_start + args.predict_future_step, len(actions))
            if done_flags[i]:
                continue

            frame_idx = frame_index[i]
            global_idx = min(ep_start + frame_idx, ep_end - 1)
            item = native_dataset[global_idx]
            frame = item["observation.images.egocentric"]
            if hasattr(frame, "detach"):
                frame = frame.detach().cpu().numpy()
                if frame.ndim == 3 and frame.shape[0] in (1, 3):
                    frame = np.transpose(frame, (1, 2, 0))
                if frame.dtype != np.uint8:
                    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)

            task_desc_str = task_desc.get(int(task_index[i]), "finish the task")
            payload = {
                "state": reorder_by_slices(np.asarray(states[i], dtype=np.float32), state_slices).tolist(),
                "image_b64": encode_image_b64(frame),
                "task_desc": task_desc_str,
            }

            req = request.Request(
                args.server_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with request.urlopen(req) as resp:
                    resp_payload = json.loads(resp.read())
            except request.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"Server {args.server_url} returned {exc.code}: {body}") from exc
            pred = np.array(resp_payload["action"], dtype=np.float32)
            if pred.ndim == 1:
                pred = pred.reshape(1, -1)
            gt_chunk = []
            for t in range(fut_start, fut_end):
                gt_chunk.append(reorder_by_slices(np.array(actions[t], dtype=np.float32), action_slices))
            gt = np.stack(gt_chunk, axis=0)
            if args.episode_idx is not None and args.frame_idx is not None:
                print("[test] pred:", pred.tolist())
                print("[test] gt:", gt.tolist())
            if not printed_first:
                per_dim_err = (pred - gt).reshape(-1).tolist()
                print("[test] per-dimension error (pred-gt):", per_dim_err)
                if norm_mean is not None and norm_std is not None and len(norm_mean) == len(gt[0]):
                    pred_norm = (pred - norm_mean) / (norm_std + 1e-6)
                    gt_norm = (gt - norm_mean) / (norm_std + 1e-6)
                    per_dim_err_norm = (pred_norm - gt_norm).reshape(-1).tolist()
                    print("[test] per-dimension error normalized:", per_dim_err_norm)
                printed_first = True
            mse = float(np.mean((pred - gt) ** 2))
            mse_sum += mse
            ep_mse_sum += mse
            ep_count += 1
            if norm_mean is not None and norm_std is not None and len(norm_mean) == len(gt[0]):
                pred_norm = (pred - norm_mean) / (norm_std + 1e-6)
                gt_norm = (gt - norm_mean) / (norm_std + 1e-6)
                mse_norm = float(np.mean((pred_norm - gt_norm) ** 2))
                mse_norm_sum += mse_norm
            total += 1

        if total >= args.max_samples:
            break
        if args.episode_mean and ep_count > 0:
            print(f"[test] episode {episode_index} mean_mse={ep_mse_sum / ep_count:.6f} samples={ep_count}")

    if total == 0:
        print("No samples evaluated.")
        return
    msg = f"samples={total} mse={mse_sum / total:.6f}"
    if mse_norm_sum > 0:
        msg += f" mse_norm={mse_norm_sum / total:.6f}"
    print(msg)


if __name__ == "__main__":
    main()
