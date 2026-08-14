import argparse
import pathlib
import re
import shutil

import h5py
import numpy as np
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.rotation_conversion import repack_action
from diffusion_policy.real_world.video_recorder import VideoRecorder

ROBOT_FOR_ARM = {1: "robot_a", 2: "robot_b"}
CAMERA_FOR_ARM = {1: "ego_a", 2: "ego_b"}  # arm{N}'s private camera; camera_3 (shared) is always "scene"

# Proprio (25D): root pos(3) + root rot(4) + left eef pos(3) + left eef rot(4)
# + right eef pos(3) + right eef rot(4) + grasp(4: l_trigger, l_squeeze,
# r_trigger, r_squeeze).

PROPRIO_OBS_FIELDS = [
    "root_pos",
    "root_rot",
    "left_eef_pos",
    "left_eef_rot",
    "right_eef_pos",
    "right_eef_rot",
    "left_grasp",
    "right_grasp",
]

DEFAULT_FPS = 10  # fallback only -- overridden by the source data's measured sync/sim_time rate.


# --------------------------------------------------------------------------
# Hand action compression: 14 DoF -> 4 independent signals per robot.
# Vendored verbatim from scripts/data_convertion.py (see there for the full
# derivation) -- kept here rather than imported so this file has zero
# dependency on the main repo's scripts/ directory.
#
# Column layout inside one robot's 32D slice's [14:28) hand block, from the
# TensorReorderer output_order in external/IsaacLab's
# locomanipulation_g1_env_cfg.py:_build_g1_locomanipulation_pipeline:
#   0 l_index_proximal   1 l_middle_proximal   2 l_thumb_rotation
#   3 r_index_proximal   4 r_middle_proximal   5 r_thumb_rotation
#   6 l_index_distal     7 l_middle_distal     8 l_thumb_proximal
#   9 r_index_distal    10 r_middle_distal    11 r_thumb_proximal
#  12 l_thumb_distal    13 r_thumb_distal
# --------------------------------------------------------------------------

HAND_BLOCK_OFFSET = 14
HAND_SIGNAL_COLS = (0, 1, 3, 4)  # l_index_proximal, l_middle_proximal, r_index_proximal, r_middle_proximal


def compress_hand_action(action: np.ndarray) -> np.ndarray:
    """Replace each robot's 14D hand block with its 4 independent signals.

    Works on any action array laid out as one or more concatenated
    32D-per-robot slices (32D for a single decentralized robot, 64D for the
    centralized pair) -- each 32D chunk becomes 22D.
    """
    if action.shape[1] % 32 != 0:
        raise SystemExit(
            "--compress-hands expects a 32D-per-robot action layout (the raw 64D "
            f"'actions', not --action-key processed_actions) -- got {action.shape[1]}D"
        )
    hand_cols = [HAND_BLOCK_OFFSET + c for c in HAND_SIGNAL_COLS]
    chunks = []
    for start in range(0, action.shape[1], 32):
        robot = action[:, start : start + 32]
        chunks.append(np.concatenate([robot[:, :14], robot[:, hand_cols], robot[:, 28:32]], axis=1))
    return np.concatenate(chunks, axis=1)


def slice_action_for_robot(action: np.ndarray, robot: str) -> np.ndarray:
    """robot_a = first half, robot_b = second half -- 64D raw actions split
    32/32; any other even width (e.g. 86D processed_actions) splits evenly
    the same way."""
    dim = action.shape[1]
    half = 32 if dim == 64 else dim // 2
    if dim != 64 and dim % 2 != 0:
        raise SystemExit(f"can't split a {dim}D action evenly between robot_a/robot_b")
    start = 0 if robot == "robot_a" else half
    return action[:, start : start + half]


# --------------------------------------------------------------------------
# Self-contained HDF5 reading -- a minimal stand-in for scripts/_dataset.py's
# DemoSource, vendored here for the same reason as compress_hand_action
# above: zero dependency on the main repo's scripts/ directory.
# --------------------------------------------------------------------------

_SHARD_NAME = re.compile(r"^.+\.demo_(?P<index>\d+)\.hdf5$")
_DEMO_NAME = re.compile(r"^demo_(\d+)$")


class DemoSource:
    """A single ``.hdf5``, a directory of ``<stem>.demo_N.hdf5`` shards, or
    the dataset root above that shard directory -- yields demos in index
    order either way. Context manager; ``demos[name]`` is the ``h5py.Group``."""

    def __init__(self, dataset_path):
        self.path = pathlib.Path(dataset_path).expanduser()
        self._streams: list = []
        self._groups: dict = {}
        self.names: list = []

    def _shard_files(self) -> list[pathlib.Path]:
        path = self.path
        directory = None
        if path.is_dir():
            if any(_SHARD_NAME.match(child.name) for child in path.iterdir()):
                directory = path
            else:
                nested = path / "data"
                if nested.is_dir() and any(_SHARD_NAME.match(child.name) for child in nested.iterdir()):
                    directory = nested
        if directory is None:
            return []
        shards = [child for child in directory.iterdir() if _SHARD_NAME.match(child.name)]
        return sorted(shards, key=lambda p: int(_SHARD_NAME.match(p.name).group("index")))

    def __enter__(self) -> "DemoSource":
        shards = self._shard_files()
        if shards:
            names: list[str] = []
            for shard in shards:
                stream = h5py.File(shard, "r")
                self._streams.append(stream)
                data = stream["data"]
                for name in data:
                    if _DEMO_NAME.match(name):
                        self._groups[name] = data[name]
                        names.append(name)
            self.names = sorted(names, key=lambda n: int(n.split("_")[1]))
            return self

        if not self.path.is_file():
            raise SystemExit(f"{self.path} is neither a dataset file nor a directory of shards")
        stream = h5py.File(self.path, "r")
        self._streams.append(stream)
        data = stream.get("data")
        if data is None:
            raise SystemExit(f"{self.path} has no 'data' group")
        self._groups = {name: data[name] for name in data if _DEMO_NAME.match(name)}
        self.names = sorted(self._groups, key=lambda n: int(n.split("_")[1]))
        return self

    def __exit__(self, *exc) -> None:
        for stream in self._streams:
            stream.close()
        self._streams.clear()
        self._groups.clear()

    def __getitem__(self, name: str):
        return self._groups[name]


def list_demos(demos: DemoSource, include_failed: bool, max_demos) -> list[str]:
    names = demos.names
    kept = [n for n in names if include_failed or bool(demos[n].attrs.get("success", True))]
    dropped = len(names) - len(kept)
    if dropped:
        print(f"[LatentToM] skipping {dropped} non-success episode(s) (--include-failed to keep them)")
    if max_demos is not None:
        kept = kept[:max_demos]
    if not kept:
        raise SystemExit("no episodes to convert")
    print(f"[LatentToM] converting {len(kept)} episode(s)")
    return kept


def read_actions(demo: h5py.Group, key: str = "actions") -> np.ndarray:
    if key not in demo:
        raise SystemExit(f"'{demo.name}' has no '{key}' dataset")
    return np.asarray(demo[key], dtype=np.float32)


def read_rgb(demo: h5py.Group, cam: str) -> np.ndarray:
    return np.asarray(demo[f"obs/images/{cam}/rgb"])  # (T, H, W, 3) uint8


def read_sim_time(demo: h5py.Group) -> np.ndarray | None:
    if "sync/sim_time" not in demo:
        return None
    return np.asarray(demo["sync/sim_time"], dtype=np.float64).squeeze(-1)


def estimate_fps(demo_names: list[str], demos: DemoSource, fallback: float) -> float:
    for name in demo_names:
        t = read_sim_time(demos[name])
        if t is not None and len(t) > 1:
            dt = np.median(np.diff(t))
            if dt > 0:
                return float(1.0 / dt)
    return fallback


def _proprio_from_obs(demo: h5py.Group, robot: str, episode_length: int) -> np.ndarray:
    parts = []
    for field in PROPRIO_OBS_FIELDS:
        key = f"obs/{robot}_{field}"
        if key not in demo:
            raise SystemExit(f"'{demo.name}' has no '{key}' -- can't build proprio without it")
        arr = np.asarray(demo[key], dtype=np.float32)
        if arr.shape[0] != episode_length:
            raise SystemExit(f"'{key}' has {arr.shape[0]} rows, expected {episode_length} (from 'actions')")
        parts.append(arr)
    return np.concatenate(parts, axis=-1).astype(np.float32)


def convert(
    source_path: str,
    out_dir: pathlib.Path,
    rotation_rep: str,
    action_key: str,
    compress_hands: bool,
    include_failed: bool,
    max_demos,
    fps_fallback: int,
) -> None:
    with DemoSource(source_path) as demos:
        demo_names = list_demos(demos, include_failed, max_demos)
        first_demo = demos[demo_names[0]]

        if read_sim_time(first_demo) is None:
            fps = fps_fallback
            print(f"[LatentToM] no sync/sim_time in source; using fallback fps={fps}")
        else:
            fps = int(round(estimate_fps(demo_names, demos, fps_fallback)))
            print(f"[LatentToM] measured {fps} Hz from sync/sim_time")

        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        video_dir = out_dir / "videos"

        buffer = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(out_dir / "replay_buffer.zarr")))
        recorders = {cam_idx: VideoRecorder.create_h264(fps=fps) for cam_idx in (1, 3, 4)}

        for ep_idx, name in enumerate(demo_names):
            demo = demos[name]
            action = read_actions(demo, action_key)
            episode_length = action.shape[0]

            proprio = {
                arm: _proprio_from_obs(demo, robot, episode_length) for arm, robot in ROBOT_FOR_ARM.items()
            }
            actions = {}
            for arm, robot in ROBOT_FOR_ARM.items():
                robot_action = slice_action_for_robot(action, robot)
                if compress_hands:
                    robot_action = compress_hand_action(robot_action)
                actions[arm] = repack_action(robot_action.astype(np.float32), rotation_rep)

            # camera_1 = arm1's private view, camera_3 = shared scene, camera_4 = arm2's private view.
            camera_frames = {
                1: read_rgb(demo, CAMERA_FOR_ARM[1]),
                3: read_rgb(demo, "scene"),
                4: read_rgb(demo, CAMERA_FOR_ARM[2]),
            }

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
            print(f"[LatentToM] {ep_idx + 1}/{len(demo_names)} {name}: {episode_length} steps")

    print(
        f"[LatentToM] wrote {out_dir} ({len(demo_names)} episodes, "
        f"rotation_rep={rotation_rep}, fps={fps})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_path", help="Raw MHBench dataset: a .hdf5, a shard directory, or the dataset root.")
    parser.add_argument("out_dir", help="Directory to write replay_buffer.zarr/videos/ into.")
    parser.add_argument("--rotation-rep", choices=["quat", "rot6d"], default="quat")
    parser.add_argument(
        "--action-key", default="actions", choices=["actions", "processed_actions"],
        help="Which top-level action dataset to read (raw 64D teleop, or the 86D retargeted one).",
    )
    parser.add_argument(
        "--compress-hands", action=argparse.BooleanOptionalAction, default=True,
        help="32D/robot -> 22D/robot hand compression (default on); only valid with "
        "--action-key actions (the default), not processed_actions.",
    )
    parser.add_argument("--include-failed", action="store_true", help="Convert every episode, not just ones flagged success.")
    parser.add_argument("--max-demos", type=int, default=None, help="Convert only the first N episodes (for a quick check).")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Fallback fps if the source has no sync/sim_time.")
    args = parser.parse_args()
    convert(
        args.source_path,
        pathlib.Path(args.out_dir),
        args.rotation_rep,
        args.action_key,
        args.compress_hands,
        args.include_failed,
        args.max_demos,
        args.fps,
    )


if __name__ == "__main__":
    main()
