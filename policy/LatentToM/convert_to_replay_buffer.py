"""MHBench's LeRobot v2.1 export -> LatentToM's replay_buffer.zarr + videos/.

Reads ``datasets/<task>/lerobot/`` (``scripts/export_lerobot.py``'s output in
the main MHBench repo)

Proprio (43D/robot) and action (35D/robot) are joint-space, matching the
benchmark's own GR00T contract (``configs/gr00t/mhbench_keys.py``) Action is
exactly ``mhbench_keys.action_keys(robot)``'s key list; proprio is the plain
43D joint block (``mhbench_keys.JOINT_GROUPS``)

    state  43 = 7 URDF joint groups: legs(12) + waist(3) + arms(14) + hands(14)
    action 35 = 28 joint targets (arm+hand groups, no legs)
                + base_height(1) + navigate_command(3)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import cv2
import numpy as np
import pyarrow.parquet as pq
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.real_world.video_recorder import VideoRecorder

ROBOT_FOR_ARM = {1: "robot_a", 2: "robot_b"}
CAMERA_FOR_ARM = {1: "ego_a", 2: "ego_b"}  # arm{N}'s private camera; camera_3 (shared) is always "scene"

STATE_KEY_NAMES = ("left_leg", "right_leg", "waist", "left_arm", "left_hand", "right_arm", "right_hand")

ACTION_KEY_NAMES = (
    "left_arm", "right_arm", "left_hand", "right_hand", "waist",
    "base_height_command", "navigate_command",
)

DEFAULT_SOURCE = {"state": "observation.state", "action": "action"}


def load_json(path: pathlib.Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text())


def load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class LerobotDataset:
    """One ``datasets/<task>/lerobot/`` export: meta plus per-episode paths."""

    def __init__(self, root: pathlib.Path):
        self.root = root
        meta = root / "meta"
        if not meta.is_dir():
            raise SystemExit(f"{root} has no meta/ -- not a LeRobot export (run scripts/export_lerobot.py first)")
        self.info = load_json(meta / "info.json")
        self.modality = load_json(meta / "modality.json")
        self.episodes = load_jsonl(meta / "episodes.jsonl")
        provenance = load_json(meta / "mhbench_provenance.json")
        self.success = {e["episode_index"]: bool(e.get("success", True)) for e in provenance.get("episodes", [])}

    def data_path(self, episode_index: int) -> pathlib.Path:
        chunk = episode_index // self.info["chunks_size"]
        return self.root / self.info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)

    def video_path(self, episode_index: int, camera: str) -> pathlib.Path:
        chunk = episode_index // self.info["chunks_size"]
        video_key = f"observation.images.{camera}"
        return self.root / self.info["video_path"].format(
            episode_chunk=chunk, video_key=video_key, episode_index=episode_index
        )


def val_episode_indices(dataset: LerobotDataset) -> set[int]:
    """Episode indices the export declares as validation.

    ``meta/info.json``'s ``splits`` are half-open index ranges --
    ``{"train": "0:50", "val": "50:65"}``. Empty when the export declares none.
    """
    splits = dataset.info.get("splits") or {}
    spec = splits.get("val") or splits.get("validation")
    if not spec:
        return set()
    start, _, end = str(spec).partition(":")
    return set(range(int(start), int(end or dataset.info["total_episodes"])))


def list_episodes(dataset: LerobotDataset, include_failed: bool, max_demos) -> list[int]:
    indices = [e["episode_index"] for e in dataset.episodes]
    kept = [i for i in indices if include_failed or dataset.success.get(i, True)]
    dropped = len(indices) - len(kept)
    if dropped:
        print(f"[LatentToM] skipping {dropped} non-success episode(s) (--include-failed to keep them)")
    if max_demos is not None:
        kept = kept[:max_demos]
    if not kept:
        raise SystemExit("no episodes to convert")
    print(f"[LatentToM] converting {len(kept)} episode(s)")
    return kept


def read_columns(table: pq.Table, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Only the vector columns this converter actually needs, as (T, D) float64."""
    out = {}
    for name in names:
        if name in table.column_names:
            out[name] = np.array(table.column(name).to_pylist(), dtype=np.float64)
    return out


def resolve(modality_section: dict, key: str, section: str) -> tuple[str, int, int]:
    if key not in modality_section:
        raise SystemExit(f"modality.json[{section!r}] has no {key!r} -- export/mhbench_keys.py mismatch?")
    entry = modality_section[key]
    return entry.get("original_key", DEFAULT_SOURCE[section]), entry["start"], entry["end"]


def robot_vector(
    columns: dict[str, np.ndarray], modality: dict, section: str, robot: str, names: tuple[str, ...]
) -> np.ndarray:
    parts = []
    for name in names:
        key = f"{robot}_{name}"
        source, start, end = resolve(modality[section], key, section)
        if source not in columns:
            raise SystemExit(f"'{key}' points at column {source!r}, not read for this episode")
        parts.append(columns[source][:, start:end])
    return np.concatenate(parts, axis=-1).astype(np.float32)


def read_video(path: pathlib.Path) -> np.ndarray:
    """(T, H, W, 3) RGB uint8.

    ``cv2.VideoCapture.read()`` hands back BGR -- OpenCV's own decode order,
    not the image-bit codepath ``AGENTS.md`` reserves for
    ``utils/process_data.py`` -- so the ``COLOR_BGR2RGB`` right after it is
    the documented exception, not a bug.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"could not open {path}")
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise SystemExit(f"{path} has no frames")
    return np.stack(frames, axis=0)


def convert(source_path: str, out_dir: pathlib.Path, include_failed: bool, max_demos) -> None:
    dataset = LerobotDataset(pathlib.Path(source_path).expanduser())
    episode_indices = list_episodes(dataset, include_failed, max_demos)
    fps = int(round(dataset.info["fps"]))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    video_dir = out_dir / "videos"

    buffer = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(out_dir / "replay_buffer.zarr")))
    recorders = {cam_idx: VideoRecorder.create_h264(fps=fps) for cam_idx in (1, 3, 4)}

    needed_columns = ("observation.state", "action", "teleop.navigate_command", "teleop.base_height_command")

    for ep_idx, episode_index in enumerate(episode_indices):
        table = pq.read_table(dataset.data_path(episode_index))
        columns = read_columns(table, needed_columns)
        episode_length = table.num_rows

        proprio = {
            arm: robot_vector(columns, dataset.modality, "state", robot, STATE_KEY_NAMES)
            for arm, robot in ROBOT_FOR_ARM.items()
        }
        actions = {
            arm: robot_vector(columns, dataset.modality, "action", robot, ACTION_KEY_NAMES)
            for arm, robot in ROBOT_FOR_ARM.items()
        }

        # camera_1 = arm1's private view, camera_3 = shared scene, camera_4 = arm2's private view.
        camera_frames = {
            1: read_video(dataset.video_path(episode_index, CAMERA_FOR_ARM[1])),
            3: read_video(dataset.video_path(episode_index, "scene")),
            4: read_video(dataset.video_path(episode_index, CAMERA_FOR_ARM[2])),
        }
        for cam_idx, frames in camera_frames.items():
            if frames.shape[0] != episode_length:
                raise SystemExit(
                    f"episode {episode_index}: camera {cam_idx} has {frames.shape[0]} frames, "
                    f"expected {episode_length} (from the parquet row count)"
                )

        ep_video_dir = video_dir / str(ep_idx)
        ep_video_dir.mkdir(parents=True)
        for cam_idx, frames in camera_frames.items():
            recorder = recorders[cam_idx]
            recorder.start(str(ep_video_dir / f"{cam_idx}.mp4"))
            for t in range(episode_length):
                recorder.write_frame(np.ascontiguousarray(frames[t]))
            recorder.stop()

        buffer.add_episode(
            {
                # Per-episode clock, spaced by 1/fps -- matches the video
                # encoding fps exactly, so real_data_to_replay_buffer's
                # frame/step alignment is 1:1 (no repeats/skips).
                "timestamp": np.arange(episode_length, dtype=np.float64) / fps,
                "arm1_proprio": proprio[1],
                "arm2_proprio": proprio[2],
                "arm1_action": actions[1],
                "arm2_action": actions[2],
            }
        )
        print(f"[LatentToM] {ep_idx + 1}/{len(episode_indices)} episode_{episode_index}: {episode_length} steps")

    # Indexed the way the buffer is (converted order), not the way the export
    # is -- a dropped episode shifts every index after it.
    val_indices = val_episode_indices(dataset)
    if val_indices:
        val_mask = np.array([i in val_indices for i in episode_indices], dtype=bool)
        buffer.update_meta({"val_mask": val_mask})
        print(f"[LatentToM] val split from meta/info.json: "
              f"{int(val_mask.sum())} val / {int((~val_mask).sum())} train episodes")
    else:
        print("[LatentToM] no val split declared in meta/info.json -- "
              "training will fall back to task.dataset.val_ratio")

    print(f"[LatentToM] wrote {out_dir} ({len(episode_indices)} episodes, fps={fps})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_path", help="A datasets/<task>/lerobot export root (has meta/ and data/ inside).")
    parser.add_argument("out_dir", help="Directory to write replay_buffer.zarr/videos/ into.")
    parser.add_argument("--include-failed", action="store_true", help="Convert every episode, not just ones flagged success.")
    parser.add_argument("--max-demos", type=int, default=None, help="Convert only the first N episodes (for a quick check).")
    args = parser.parse_args()
    convert(args.source_path, pathlib.Path(args.out_dir), args.include_failed, args.max_demos)


if __name__ == "__main__":
    main()
