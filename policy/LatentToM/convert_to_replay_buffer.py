import argparse
import pathlib
import shutil
import sys

import numpy as np
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.rotation_conversion import repack_action
from diffusion_policy.real_world.video_recorder import VideoRecorder

ROBOT_FOR_ARM = {1: "robot_a", 2: "robot_b"}
CAMERA_FOR_ARM = {1: "ego_a", 2: "ego_b"}  # arm{N}'s private camera; camera_3 (shared) is always "scene"

# Proprio (21D): pelvis pose(7) + left eef + pose(7) + right eef pose(7)
PROPRIO_FIELDS = ["root_pose", "left_eef_pos", "left_eef_rot", "right_eef_pos", "right_eef_rot"]

DEFAULT_FPS = 10  # fallback only -- overridden by the source data's measured sync/sim_time rate.


def _import_data_convertion():
    mhbench_root = pathlib.Path(__file__).resolve().parents[4]
    scripts_dir = mhbench_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import data_convertion as dc  # noqa: E402  (MHBench repo root, not a package)

    return dc


def _proprio_from_state(state: np.ndarray, state_fields, robot: str) -> np.ndarray:
    field_ranges = {name: (start, end) for name, start, end, _ in state_fields}
    parts = []
    for field in PROPRIO_FIELDS:
        name = f"{robot}_{field}"
        if name not in field_ranges:
            raise KeyError(
                f"'{name}' not in this episode's state fields -- data_convertion.py's "
                "CANDIDATE_STATE_FIELDS needs obs/robot_{a,b}_{left,right}_eef_{pos,rot} "
                "recorded before this converter can extract full 21D proprio."
            )
        start, end = field_ranges[name]
        parts.append(state[:, start:end])
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
    dc = _import_data_convertion()

    with dc.DemoSource(source_path) as demos:
        demo_names = dc.list_demos(demos, include_failed, max_demos)
        first_demo = demos[demo_names[0]]
        schema = dc.build_state_schema(first_demo)

        if dc.read_sim_time(first_demo) is None:
            fps = fps_fallback
            print(f"[LatentToM] no sync/sim_time in source; using fallback fps={fps}")
        else:
            fps = int(round(dc.estimate_fps(demo_names, demos, fps_fallback)))
            print(f"[LatentToM] measured {fps} Hz from sync/sim_time")

        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        video_dir = out_dir / "videos"

        buffer = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(out_dir / "replay_buffer.zarr")))
        recorders = {cam_idx: VideoRecorder.create_h264(fps=fps) for cam_idx in (1, 3, 4)}

        for ep_idx, name in enumerate(demo_names):
            demo = demos[name]
            action = dc.read_actions(demo, action_key)
            state, state_fields = dc.read_state(demo, schema, action.shape[0])
            episode_length = action.shape[0]

            proprio = {arm: _proprio_from_state(state, state_fields, robot) for arm, robot in ROBOT_FOR_ARM.items()}
            actions = {}
            for arm, robot in ROBOT_FOR_ARM.items():
                robot_action = dc.slice_action_for_robot(action, robot)
                if compress_hands:
                    robot_action = dc.compress_hand_action(robot_action)
                actions[arm] = repack_action(robot_action.astype(np.float32), rotation_rep)

            # camera_1 = arm1's private view, camera_3 = shared scene, camera_4 = arm2's private view.
            camera_frames = {
                1: np.asarray(dc.read_rgb(demo, CAMERA_FOR_ARM[1])),
                3: np.asarray(dc.read_rgb(demo, "scene")),
                4: np.asarray(dc.read_rgb(demo, CAMERA_FOR_ARM[2])),
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
        help="Same choice as data_convertion.py's --action-key.",
    )
    parser.add_argument(
        "--compress-hands", action=argparse.BooleanOptionalAction, default=True,
        help="Same as data_convertion.py's --compress-hands (default on) -- 32D/robot -> 22D/robot; "
        "only valid with --action-key actions (the default), not processed_actions.",
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
