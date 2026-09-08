"""
Convert one or more Psi0 LeRobot v2.1 datasets to a single DreamZero LeRobot v2.0 dataset.

Psi0 (G1 robot) stores data in LeRobot v2.1 format with:
  - Parquet columns: `states` (32D), `action` (36D)
  - Video: videos/chunk-NNN/egocentric/episode_NNN.mp4 (30fps or 50fps)
  - Language: in meta/episodes.jsonl `instruction` field

DreamZero expects:
  - Parquet columns: `observation.state` (flat), `action` (flat),
    `annotation.language.language_instruction` (int task_index)
  - Video: videos/chunk-NNN/observation.images.egocentric/episode_NNN.mp4
  - Language: task_index int in parquet → meta/tasks.jsonl for string lookup
  - Metadata: info.json (v2.0), modality.json, stats.json, episodes.jsonl, tasks.jsonl

When multiple --input-paths are given the datasets are merged into one output:
  - Episodes are renumbered globally (source 0: 0..N0-1, source 1: N0..N0+N1-1, …)
  - Frame indices are offset accordingly so the global `index` column is monotone
  - Task strings are deduplicated; a single tasks.jsonl covers all sources

Usage:
  # Single task
  python baselines/dreamzero/convert_psi0_to_dreamzero.py \
      --input-paths /path/to/psi0/Task_A \
      --output-path /path/to/dreamzero/Task_A

  # Multiple tasks merged into one dataset
  python baselines/dreamzero/convert_psi0_to_dreamzero.py \
      --input-paths /path/to/psi0/Task_A /path/to/psi0/Task_B /path/to/psi0/Task_C \
      --output-path /path/to/dreamzero/merged
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# G1 robot dimensions (defaults match SIMPLE sim; override via CLI for real-robot data)
G1_STATE_DIM = 32
G1_ACTION_DIM = 36
EMBODIMENT_TAG = "simple_g1"
OUTPUT_CHUNKS_SIZE = 1000  # chunk size used for the merged output


# ──────────────────────────────────────────────────────────────────────────────
# Per-source loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_psi0_info(input_path: Path) -> dict:
    """Load and validate Psi0 meta/info.json. Accepts 30fps (real) or 50fps (sim)."""
    info_path = input_path / "meta" / "info.json"
    if not info_path.exists():
        log.error("meta/info.json not found at %s", info_path)
        sys.exit(1)
    with open(info_path) as f:
        info = json.load(f)
    assert info.get("codebase_version") in ("v2.1", "v2.0"), \
        f"Expected LeRobot v2.0 or v2.1, got {info.get('codebase_version')}"
    fps = info.get("fps")
    assert fps in (30, 50), f"Expected fps in (30, 50), got {fps}"
    return info


def load_psi0_episodes(input_path: Path) -> list[dict]:
    """Load episodes.jsonl and extract instructions."""
    episodes_path = input_path / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        log.error("meta/episodes.jsonl not found at %s", episodes_path)
        sys.exit(1)
    episodes = []
    with open(episodes_path) as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    return episodes


def _instruction_text(ep: dict) -> str:
    """Extract the human-readable task instruction from a Psi0 episode entry.

    Handles two formats:
      - real Psi0: {"instruction": "grasp the cup"}
      - SIMPLE sim: {"instruction": {"task_index": 0, "task": "grasp the cup"}}
    """
    instr = ep.get("instruction", "")
    if isinstance(instr, dict):
        instr = instr.get("task", "") or instr.get("instruction", "")
    instr = (instr or "").strip()
    return instr or "not provided"


# ──────────────────────────────────────────────────────────────────────────────
# Global task mapping (built across all sources)
# ──────────────────────────────────────────────────────────────────────────────

def build_global_task_mapping(
    all_episodes: list[list[dict]],
) -> tuple[list[dict], dict[str, int]]:
    """Build a deduplicated tasks.jsonl covering all source datasets.

    Returns (tasks_list, instruction_to_task_index).
    Tasks are assigned indices in first-seen order across sources.
    """
    instruction_to_idx: dict[str, int] = {}
    for episodes in all_episodes:
        for ep in episodes:
            instr = _instruction_text(ep)
            if instr not in instruction_to_idx:
                instruction_to_idx[instr] = len(instruction_to_idx)
    tasks = [
        {"task_index": idx, "task": text}
        for text, idx in sorted(instruction_to_idx.items(), key=lambda x: x[1])
    ]
    return tasks, instruction_to_idx


# ──────────────────────────────────────────────────────────────────────────────
# Merged conversion: parquet + video
# ──────────────────────────────────────────────────────────────────────────────

def convert_and_merge_parquets(
    all_input_paths: list[Path],
    all_infos: list[dict],
    all_episodes: list[list[dict]],
    output_path: Path,
    global_task_mapping: dict[str, int],
    state_dim: int = G1_STATE_DIM,
    action_dim: int = G1_ACTION_DIM,
) -> tuple[int, int]:
    """Convert and merge parquets from all sources into the output directory.

    Episode indices and frame indices are assigned globally so the merged
    dataset has contiguous, non-overlapping episode_index and index columns.

    Returns (total_episodes, total_frames).
    """
    ep_offset = 0    # global episode index of the first episode in current source
    frame_offset = 0 # global frame index of the first frame in current source

    for src_path, info, episodes in zip(all_input_paths, all_infos, all_episodes):
        local_chunks_size = info.get("chunks_size", 1000)
        data_pattern = info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
        total_local_eps = info["total_episodes"]

        for local_ep_idx in tqdm(
            range(total_local_eps), desc=f"Parquets  {src_path.name}"
        ):
            local_chunk = local_ep_idx // local_chunks_size
            src = src_path / data_pattern.format(
                episode_chunk=local_chunk, episode_index=local_ep_idx
            )
            if not src.exists():
                log.warning("Missing parquet: %s", src)
                continue

            df = pd.read_parquet(src)
            local_len = len(df)

            # ── column renames ────────────────────────────────────────────────
            if "states" in df.columns:
                df = df.rename(columns={"states": "observation.state"})

            # ── validate dimensions ───────────────────────────────────────────
            sample_state = np.array(df["observation.state"].iloc[0])
            sample_action = np.array(df["action"].iloc[0])
            assert sample_state.shape == (state_dim,), \
                f"{src_path.name} ep {local_ep_idx}: state {sample_state.shape} != ({state_dim},)"
            assert sample_action.shape == (action_dim,), \
                f"{src_path.name} ep {local_ep_idx}: action {sample_action.shape} != ({action_dim},)"

            # ── global index columns ──────────────────────────────────────────
            global_ep_idx = ep_offset + local_ep_idx
            df["episode_index"] = global_ep_idx
            df["index"] = range(frame_offset, frame_offset + local_len)

            # ── language annotation ───────────────────────────────────────────
            ep_meta = episodes[local_ep_idx]
            instr = _instruction_text(ep_meta)
            task_idx = global_task_mapping[instr]
            df["annotation.language.language_instruction"] = task_idx
            df["task_index"] = task_idx

            # ── standard v2.0 columns ─────────────────────────────────────────
            if "next.done" not in df.columns:
                df["next.done"] = False
                df.iloc[-1, df.columns.get_loc("next.done")] = True

            # ── enforce dtypes ────────────────────────────────────────────────
            df["observation.state"] = df["observation.state"].apply(
                lambda x: np.array(x, dtype=np.float32)
            )
            df["action"] = df["action"].apply(
                lambda x: np.array(x, dtype=np.float32)
            )

            # ── write to output using global episode index ────────────────────
            global_chunk = global_ep_idx // OUTPUT_CHUNKS_SIZE
            dst = output_path / f"data/chunk-{global_chunk:03d}/episode_{global_ep_idx:06d}.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst, index=False)

            frame_offset += local_len

        ep_offset += total_local_eps

    return ep_offset, frame_offset


def copy_and_merge_videos(
    all_input_paths: list[Path],
    all_infos: list[dict],
    output_path: Path,
) -> None:
    """Copy videos from all sources into the merged output.

    Source: videos/chunk-NNN/egocentric/episode_NNN.mp4
    Output: videos/chunk-GGG/observation.images.egocentric/episode_GGG.mp4
    where GGG is the global episode index.
    """
    ep_offset = 0
    for src_path, info in zip(all_input_paths, all_infos):
        local_chunks_size = info.get("chunks_size", 1000)
        total_local_eps = info["total_episodes"]

        for local_ep_idx in tqdm(
            range(total_local_eps), desc=f"Videos    {src_path.name}"
        ):
            local_chunk = local_ep_idx // local_chunks_size
            src_video = (
                src_path
                / f"videos/chunk-{local_chunk:03d}/egocentric/episode_{local_ep_idx:06d}.mp4"
            )
            if not src_video.exists():
                log.warning("Missing video: %s", src_video)
                continue

            global_ep_idx = ep_offset + local_ep_idx
            global_chunk = global_ep_idx // OUTPUT_CHUNKS_SIZE
            dst_dir = (
                output_path
                / f"videos/chunk-{global_chunk:03d}/observation.images.egocentric"
            )
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_video = dst_dir / f"episode_{global_ep_idx:06d}.mp4"

            if not dst_video.exists():
                shutil.copy2(src_video, dst_video)

        ep_offset += total_local_eps

    log.info("Videos copied to merged DreamZero directory structure")


# ──────────────────────────────────────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────────────────────────────────────

def compute_stats(output_path: Path, total_episodes: int) -> dict:
    """Compute q99 normalization statistics over the full merged dataset."""
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []

    for global_ep_idx in tqdm(range(total_episodes), desc="Computing stats"):
        global_chunk = global_ep_idx // OUTPUT_CHUNKS_SIZE
        pq_path = (
            output_path
            / f"data/chunk-{global_chunk:03d}/episode_{global_ep_idx:06d}.parquet"
        )
        if not pq_path.exists():
            continue
        df = pd.read_parquet(pq_path)
        all_states.append(np.stack(df["observation.state"].values))
        all_actions.append(np.stack(df["action"].values))

    states  = np.concatenate(all_states,  axis=0).astype(np.float64)
    actions = np.concatenate(all_actions, axis=0).astype(np.float64)

    def per_col_stats(data: np.ndarray) -> dict:
        return {
            "mean": np.mean(data,              axis=0).tolist(),
            "std":  np.std(data,               axis=0).tolist(),
            "min":  np.min(data,               axis=0).tolist(),
            "max":  np.max(data,               axis=0).tolist(),
            "q01":  np.quantile(data, 0.01,    axis=0).tolist(),
            "q99":  np.quantile(data, 0.99,    axis=0).tolist(),
        }

    return {
        "observation.state": per_col_stats(states),
        "action":            per_col_stats(actions),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Metadata writers
# ──────────────────────────────────────────────────────────────────────────────

def write_info_json(
    output_path: Path,
    fps: int,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    video_shape: list,
    video_codec: str,
    state_dim: int = G1_STATE_DIM,
    action_dim: int = G1_ACTION_DIM,
) -> None:
    """Write DreamZero-compatible info.json (v2.0 format)."""
    total_chunks = (total_episodes + OUTPUT_CHUNKS_SIZE - 1) // OUTPUT_CHUNKS_SIZE
    new_info = {
        "codebase_version": "v2.0",
        "robot_type": EMBODIMENT_TAG,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": 1,
        "total_chunks": total_chunks,
        "chunks_size": OUTPUT_CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.egocentric": {
                "dtype": "video",
                "shape": video_shape,
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": video_codec,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": ["joint_position"],
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": ["joint_position"],
            },
            "timestamp":   {"dtype": "float32", "shape": [1]},
            "task_index":  {"dtype": "int64",   "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index":       {"dtype": "int64",   "shape": [1]},
            "next.done":   {"dtype": "bool",    "shape": [1]},
            "annotation.language.language_instruction": {"dtype": "int64", "shape": [1]},
        },
    }
    with open(output_path / "meta" / "info.json", "w") as f:
        json.dump(new_info, f, indent=4)
    log.info("Wrote info.json (v2.0): %d episodes, %d frames, %d tasks",
             total_episodes, total_frames, total_tasks)


def write_modality_json(
    output_path: Path,
    state_dim: int = G1_STATE_DIM,
    action_dim: int = G1_ACTION_DIM,
) -> None:
    modality = {
        "state": {"joint_position": {"start": 0, "end": state_dim}},
        "action": {"joint_position": {"start": 0, "end": action_dim}},
        "video": {"egocentric": {"original_key": "observation.images.egocentric"}},
        "annotation": {"language.language_instruction": {"original_key": "task_index"}},
    }
    with open(output_path / "meta" / "modality.json", "w") as f:
        json.dump(modality, f, indent=4)
    log.info("Wrote modality.json")


def write_embodiment_json(output_path: Path) -> None:
    with open(output_path / "meta" / "embodiment.json", "w") as f:
        json.dump({"robot_type": EMBODIMENT_TAG, "embodiment_tag": EMBODIMENT_TAG}, f, indent=4)
    log.info("Wrote embodiment.json (tag=%s)", EMBODIMENT_TAG)


def write_tasks_jsonl(output_path: Path, tasks: list[dict]) -> None:
    with open(output_path / "meta" / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    log.info("Wrote tasks.jsonl (%d tasks)", len(tasks))


def write_episodes_jsonl(
    output_path: Path,
    all_episodes: list[list[dict]],
    all_infos: list[dict],
    global_task_mapping: dict[str, int],
) -> None:
    """Write a merged episodes.jsonl with globally renumbered episode indices."""
    ep_offset = 0
    entries = []
    for episodes, info in zip(all_episodes, all_infos):
        for ep in episodes:
            instr = _instruction_text(ep)
            entries.append({
                "episode_index": ep_offset + ep["episode_index"],
                "tasks": [instr],
                "length": ep["length"],
            })
        ep_offset += info["total_episodes"]

    with open(output_path / "meta" / "episodes.jsonl", "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    log.info("Wrote episodes.jsonl (%d episodes)", len(entries))


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert one or more Psi0 G1 datasets to a merged DreamZero dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-paths", nargs="+", required=True,
        help="One or more Psi0 LeRobot dataset directories to merge.",
    )
    parser.add_argument(
        "--output-path", type=str, required=True,
        help="Output path for the merged DreamZero dataset.",
    )
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output directory.")
    parser.add_argument("--state-dim", type=int, default=G1_STATE_DIM,
                        help=f"State dim in `states` column (default {G1_STATE_DIM}).")
    parser.add_argument("--action-dim", type=int, default=G1_ACTION_DIM,
                        help=f"Action dim in `action` column (default {G1_ACTION_DIM}).")
    args = parser.parse_args()

    state_dim  = args.state_dim
    action_dim = args.action_dim

    input_paths = [Path(p).resolve() for p in args.input_paths]
    output_path = Path(args.output_path).resolve()

    for p in input_paths:
        if not p.exists():
            log.error("Input path does not exist: %s", p)
            sys.exit(1)

    if output_path.exists() and not args.force:
        log.error("Output path exists: %s  (use --force to overwrite)", output_path)
        sys.exit(1)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "meta").mkdir(parents=True, exist_ok=True)

    # ── 1. Load metadata from every source ───────────────────────────────────
    all_infos:    list[dict]        = []
    all_episodes: list[list[dict]] = []
    for p in input_paths:
        info     = load_psi0_info(p)
        episodes = load_psi0_episodes(p)
        all_infos.append(info)
        all_episodes.append(episodes)
        log.info("Source %s: %d episodes, %d frames, %dfps",
                 p.name, info["total_episodes"], info["total_frames"], info["fps"])

    # Validate fps consistency
    fpses = {info["fps"] for info in all_infos}
    if len(fpses) > 1:
        log.warning("Sources have different fps values: %s — using first (%d)", fpses, all_infos[0]["fps"])
    merged_fps = all_infos[0]["fps"]

    # ── 2. Build global task mapping ──────────────────────────────────────────
    tasks, global_task_mapping = build_global_task_mapping(all_episodes)
    log.info("Unique tasks across all sources: %d", len(tasks))

    # ── 3. Convert + merge parquets ───────────────────────────────────────────
    total_episodes, total_frames = convert_and_merge_parquets(
        input_paths, all_infos, all_episodes, output_path,
        global_task_mapping, state_dim=state_dim, action_dim=action_dim,
    )

    # ── 4. Link/copy videos ───────────────────────────────────────────────────
    copy_and_merge_videos(input_paths, all_infos, output_path)

    # ── 5. Compute normalization stats ────────────────────────────────────────
    log.info("Computing q99 normalization stats over %d episodes…", total_episodes)
    stats = compute_stats(output_path, total_episodes)
    with open(output_path / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    log.info("Wrote stats.json")

    # ── 6. Write metadata files ───────────────────────────────────────────────
    # Derive video shape/codec from first source that has the key
    src_video_meta = all_infos[0].get("features", {}).get("observation.images.egocentric", {})
    video_shape = src_video_meta.get("shape", [480, 640, 3])
    video_codec = src_video_meta.get("video_info", {}).get("video.codec", "h264")

    write_info_json(
        output_path,
        fps=merged_fps,
        total_episodes=total_episodes,
        total_frames=total_frames,
        total_tasks=len(tasks),
        video_shape=video_shape,
        video_codec=video_codec,
        state_dim=state_dim,
        action_dim=action_dim,
    )
    write_modality_json(output_path, state_dim=state_dim, action_dim=action_dim)
    write_embodiment_json(output_path)
    write_tasks_jsonl(output_path, tasks)
    write_episodes_jsonl(output_path, all_episodes, all_infos, global_task_mapping)

    # ── 7. Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Psi0 → DreamZero conversion complete!")
    print(f"  Sources   : {len(input_paths)}")
    for p, info in zip(input_paths, all_infos):
        print(f"    {p.name}  ({info['total_episodes']} eps)")
    print(f"  Output    : {output_path}")
    print(f"  Embodiment: {EMBODIMENT_TAG}")
    print(f"  State     : observation.state ({state_dim}D)")
    print(f"  Action    : action ({action_dim}D)")
    print(f"  Video     : observation.images.egocentric ({merged_fps}fps)")
    print(f"  Tasks     : {len(tasks)}")
    print(f"  Episodes  : {total_episodes}")
    print(f"  Frames    : {total_frames}")
    print("=" * 60)


if __name__ == "__main__":
    main()
