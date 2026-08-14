#!/usr/bin/env python3
"""Convert HDF5 or LeRobot-v2.1 MHBench data into GauDP's training HDF5."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

_MHBENCH_ROOT = Path(__file__).resolve().parents[4]
_XPL_ROOT = Path(__file__).resolve().parents[2]
_MHBENCH_SCRIPTS = _MHBENCH_ROOT / "scripts"
if str(_XPL_ROOT) not in sys.path:
    sys.path.insert(0, str(_XPL_ROOT))
if str(_MHBENCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_MHBENCH_SCRIPTS))

from _dataset import DemoSource  # noqa: E402
from XPolicyLab.policy.GauDP.gaudp.schema import (  # noqa: E402
    ACTION_DIM,
    PROPRIO_DIM,
    ROBOT_NAMES,
    compress_joint_action,
    proprio_from_demo,
)


def _append(group: h5py.File, key: str, value: np.ndarray, compression: str | None = None) -> None:
    value = np.asarray(value)
    if key not in group:
        chunks = (1,) + value.shape[1:] if value.ndim > 1 else True
        group.create_dataset(
            key,
            data=value,
            maxshape=(None,) + value.shape[1:],
            chunks=chunks,
            compression=compression,
        )
        return
    dataset = group[key]
    old = dataset.shape[0]
    dataset.resize(old + value.shape[0], axis=0)
    dataset[old:] = value


def _camera_array(demo, camera: str, field: str, length: int) -> np.ndarray:
    path = f"images/{camera}/{field}" if field in {"rgb", "depth"} else f"camera_info/{camera}/{field}"
    if path not in demo:
        raise KeyError(f"{demo.name!r} is missing required camera field {path!r}")
    value = np.asarray(demo[path])
    if value.shape[0] != length:
        raise ValueError(f"{path} has {value.shape[0]} frames; expected {length}")
    return value


def _convert_hdf5(
    source: Path,
    target: h5py.File,
    max_demos: int | None,
    use_scene: bool,
    include_failed: bool,
) -> tuple[int, list[int], list[str]]:
    cameras = ["ego_a", "ego_b"] + (["scene"] if use_scene else [])
    with DemoSource(source) as demos:
        names = [name for name in demos.names if include_failed or bool(demos[name].attrs.get("success", True))]
        if max_demos is not None:
            names = names[:max_demos]
        if not names:
            raise SystemExit("no successful demonstrations found")

        episode_ends = []
        total = 0
        for episode_index, name in enumerate(names):
            demo = demos[name]
            if "actions" not in demo:
                raise KeyError(f"{demo.name!r} has no raw 'actions' dataset")
            raw_action = np.asarray(demo["actions"], dtype=np.float32)
            length = raw_action.shape[0]
            action = compress_joint_action(raw_action)
            per_robot_state = [proprio_from_demo(demo, robot, length) for robot in ROBOT_NAMES]
            state = np.concatenate(per_robot_state, axis=-1)
            if state.shape[1] != PROPRIO_DIM or action.shape[1] != ACTION_DIM:
                raise AssertionError(f"schema mismatch: state={state.shape}, action={action.shape}")

            _append(target, "state", state)
            _append(target, "action", action)
            for robot_index, robot in enumerate(ROBOT_NAMES):
                _append(target, f"state_{robot_index}", per_robot_state[robot_index])
                _append(target, f"action_{robot_index}", action[:, robot_index * 22 : (robot_index + 1) * 22])

            for camera_index, camera in enumerate(cameras):
                rgb = _camera_array(demo, camera, "rgb", length).astype(np.uint8)
                depth = _camera_array(demo, camera, "depth", length).astype(np.float16)
                intrinsics = _camera_array(demo, camera, "intrinsics", length).astype(np.float32)
                pose = _camera_array(demo, camera, "pose", length).astype(np.float32)
                _append(target, f"rgb_{camera_index}", rgb, compression="lzf")
                _append(target, f"depth_{camera_index}", depth, compression="lzf")
                _append(target, f"intrinsics_{camera_index}", intrinsics)
                _append(target, f"pose_{camera_index}", pose)

            total += length
            episode_ends.append(total)
            print(f"[GauDP] {episode_index + 1}/{len(names)} {name}: {length} frames")

    target.attrs["source_format"] = "mhbench-hdf5"
    target.attrs["gaussian_supervision"] = True
    target.attrs["camera_pose_convention"] = "opengl"
    return total, episode_ends, cameras


_EPISODE_FILE = re.compile(r"episode_(\d+)\.parquet$")


def _feature_names(info: dict, key: str) -> list[str]:
    try:
        names = info["features"][key]["names"]
    except KeyError as error:
        raise KeyError(f"LeRobot metadata is missing feature {key!r}") from error
    if not names:
        raise ValueError(f"LeRobot feature {key!r} has no dimension names in meta/info.json")
    return [str(name) for name in names]


def _named_columns(values: np.ndarray, names: list[str], wanted: list[str], key: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError(f"{key} has shape {values.shape}, but metadata declares {len(names)} dimensions")
    positions = {name: index for index, name in enumerate(names)}
    missing = [name for name in wanted if name not in positions]
    if missing:
        raise KeyError(f"{key} is missing required named dimensions: {missing}")
    return values[:, [positions[name] for name in wanted]]


def _robot_pose_names(robot: str, prefix: str) -> list[str]:
    return [
        f"{robot}/{prefix}_pos_x",
        f"{robot}/{prefix}_pos_y",
        f"{robot}/{prefix}_pos_z",
        f"{robot}/{prefix}_quat_x",
        f"{robot}/{prefix}_quat_y",
        f"{robot}/{prefix}_quat_z",
        f"{robot}/{prefix}_quat_w",
    ]


def _camera_pose_names(camera: str) -> list[str]:
    return [
        f"{camera}/pos_x",
        f"{camera}/pos_y",
        f"{camera}/pos_z",
        f"{camera}/quat_x",
        f"{camera}/quat_y",
        f"{camera}/quat_z",
        f"{camera}/quat_w",
    ]


def _lerobot_state_action(columns: dict[str, np.ndarray], info: dict) -> tuple[np.ndarray, np.ndarray]:
    """Map the named LeRobot v2.1 fields to GauDP's 42D/44D contract."""
    required = (
        "observation.robots_state",
        "observation.eef_state",
        "action.eef",
        "action",
        "teleop.navigate_command",
        "teleop.base_height_command",
    )
    missing = [key for key in required if key not in columns]
    if missing:
        raise KeyError(f"LeRobot episode is missing required columns: {missing}")

    root_names = _feature_names(info, "observation.robots_state")
    eef_state_names = _feature_names(info, "observation.eef_state")
    eef_action_names = _feature_names(info, "action.eef")
    joint_action_names = _feature_names(info, "action")
    navigation_names = _feature_names(info, "teleop.navigate_command")
    height_names = _feature_names(info, "teleop.base_height_command")

    states, actions = [], []
    for robot in ROBOT_NAMES:
        root = _named_columns(
            columns["observation.robots_state"],
            root_names,
            _robot_pose_names(robot, "root"),
            "observation.robots_state",
        )
        wrists = []
        eef_actions = []
        for side in ("left", "right"):
            prefix = f"{side}_wrist"
            wanted = _robot_pose_names(robot, prefix)
            wrists.append(
                _named_columns(
                    columns["observation.eef_state"], eef_state_names, wanted, "observation.eef_state"
                )
            )
            eef_actions.append(
                _named_columns(columns["action.eef"], eef_action_names, wanted, "action.eef")
            )
        states.append(np.concatenate((root, *wrists), axis=-1))

        hands = _named_columns(
            columns["action"],
            joint_action_names,
            [
                f"{robot}/left_hand_index_0_joint",
                f"{robot}/left_hand_middle_0_joint",
                f"{robot}/right_hand_index_0_joint",
                f"{robot}/right_hand_middle_0_joint",
            ],
            "action",
        )
        navigation = _named_columns(
            columns["teleop.navigate_command"],
            navigation_names,
            [f"{robot}/lin_vel_x", f"{robot}/lin_vel_y", f"{robot}/ang_vel_z"],
            "teleop.navigate_command",
        )
        height = _named_columns(
            columns["teleop.base_height_command"],
            height_names,
            [f"{robot}/base_height"],
            "teleop.base_height_command",
        )
        actions.append(np.concatenate((*eef_actions, hands, navigation, height), axis=-1))

    state = np.concatenate(states, axis=-1).astype(np.float32)
    action = np.concatenate(actions, axis=-1).astype(np.float32)
    if state.shape[1] != PROPRIO_DIM or action.shape[1] != ACTION_DIM:
        raise AssertionError(f"LeRobot schema mismatch: state={state.shape}, action={action.shape}")
    return state, action


def _read_video(path: Path, length: int) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:
        raise SystemExit("LeRobot MP4 conversion requires opencv-python-headless; rerun install.sh") from error
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"could not open LeRobot video: {path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # VideoCapture returns BGR. GauDP and XPolicyLab use RGB end to end.
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(frames) != length:
        raise ValueError(f"{path} has {len(frames)} decoded frames; expected {length}")
    return np.stack(frames).astype(np.uint8)


def _camera_intrinsics(info: dict, camera: str) -> np.ndarray | None:
    for key in (f"observation.depth.{camera}", f"observation.images.{camera}"):
        metadata = info.get("features", {}).get(key, {}).get("info", {})
        value = metadata.get("camera.intrinsics")
        if value is not None:
            matrix = np.asarray(value, dtype=np.float32)
            if matrix.shape != (3, 3):
                raise ValueError(f"{key} camera.intrinsics must be 3x3, got {matrix.shape}")
            return matrix
    return None


def _decode_depth_video(path: Path, length: int, metadata: dict) -> np.ndarray:
    encoding = str(metadata.get("depth.encoding", ""))
    if encoding != "uint16_hi_lo_rgb":
        raise ValueError(f"unsupported depth encoding {encoding!r} for {path}")
    units_per_metre = float(metadata.get("depth.units_per_metre", 0))
    if units_per_metre <= 0:
        raise ValueError(f"invalid depth.units_per_metre={units_per_metre} for {path}")
    invalid_value = int(metadata.get("depth.invalid_value", 0))
    encoded = _read_video(path, length).astype(np.uint16)
    packed = (encoded[..., 0] << 8) | encoded[..., 1]
    depth = packed.astype(np.float32) / units_per_metre
    depth[packed == invalid_value] = np.nan
    return depth


def _episode_index(path: Path) -> int:
    match = _EPISODE_FILE.search(path.name)
    if match is None:
        raise ValueError(f"unexpected LeRobot episode filename: {path.name}")
    return int(match.group(1))


def _convert_lerobot(
    source: Path,
    target: h5py.File,
    max_demos: int | None,
    use_scene: bool,
) -> tuple[int, list[int], list[str]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit("LeRobot parquet conversion requires pyarrow; rerun install.sh") from error

    info_path = source / "meta" / "info.json"
    with info_path.open(encoding="utf-8") as stream:
        info = json.load(stream)
    cameras = ["ego_a", "ego_b"] + (["scene"] if use_scene else [])
    for camera in cameras:
        key = f"observation.images.{camera}"
        if key not in info.get("features", {}):
            raise KeyError(f"LeRobot metadata is missing required video feature {key!r}")

    files = sorted(source.glob("data/chunk-*/episode_*.parquet"), key=_episode_index)
    if max_demos is not None:
        files = files[:max_demos]
    if not files:
        raise SystemExit(f"no LeRobot episode parquet files found below {source / 'data'}")

    total = 0
    episode_ends = []
    data_template = str(info.get("data_path", ""))
    video_template = str(info.get("video_path", ""))
    camera_pose_names = (
        _feature_names(info, "observation.camera_pose")
        if "observation.camera_pose" in info.get("features", {})
        else []
    )
    intrinsics = {camera: _camera_intrinsics(info, camera) for camera in cameras}
    ego_depth_keys = [f"observation.depth.{camera}" for camera in ("ego_a", "ego_b")]
    gaussian_supervision = bool(camera_pose_names) and all(
        matrix is not None for matrix in intrinsics.values()
    ) and all(key in info.get("features", {}) for key in ego_depth_keys)
    for number, path in enumerate(files, 1):
        episode = _episode_index(path)
        table = parquet.read_table(path)
        columns = {key: np.asarray(table[key].to_pylist()) for key in table.column_names}
        state, action = _lerobot_state_action(columns, info)
        length = state.shape[0]
        if action.shape[0] != length:
            raise ValueError(f"{path} state/action length mismatch: {length} vs {action.shape[0]}")

        _append(target, "state", state)
        _append(target, "action", action)
        for robot_index in range(len(ROBOT_NAMES)):
            _append(target, f"state_{robot_index}", state[:, robot_index * 21 : (robot_index + 1) * 21])
            _append(target, f"action_{robot_index}", action[:, robot_index * 22 : (robot_index + 1) * 22])

        chunk = episode // int(info.get("chunks_size", 1000))
        for camera_index, camera in enumerate(cameras):
            video_path = (
                source
                / "videos"
                / f"chunk-{chunk:03d}"
                / f"observation.images.{camera}"
                / f"episode_{episode:06d}.mp4"
            )
            _append(target, f"rgb_{camera_index}", _read_video(video_path, length), compression="lzf")
            if gaussian_supervision:
                pose = _named_columns(
                    columns["observation.camera_pose"],
                    camera_pose_names,
                    _camera_pose_names(camera),
                    "observation.camera_pose",
                )
                intrinsic = np.repeat(intrinsics[camera][None], length, axis=0)
                depth_key = f"observation.depth.{camera}"
                if depth_key in info["features"]:
                    depth_path = (
                        source
                        / "videos"
                        / f"chunk-{chunk:03d}"
                        / depth_key
                        / f"episode_{episode:06d}.mp4"
                    )
                    depth = _decode_depth_video(
                        depth_path, length, info["features"][depth_key].get("info", {})
                    )
                else:
                    # The current release has no scene depth. RGB reconstruction
                    # still supervises that view; NaNs exclude it from depth L1.
                    height, width = info["features"][f"observation.images.{camera}"]["shape"][:2]
                    depth = np.full((length, height, width), np.nan, dtype=np.float32)
                _append(target, f"depth_{camera_index}", depth.astype(np.float16), compression="lzf")
                _append(target, f"intrinsics_{camera_index}", intrinsic)
                _append(target, f"pose_{camera_index}", pose)

        total += length
        episode_ends.append(total)
        print(f"[GauDP] {number}/{len(files)} episode_{episode:06d}: {length} frames")

    target.attrs["source_format"] = "lerobot-v2.1"
    target.attrs["source_data_path"] = data_template
    target.attrs["source_video_path"] = video_template
    target.attrs["gaussian_supervision"] = gaussian_supervision
    target.attrs["depth_encoding"] = "uint16_hi_lo_rgb" if gaussian_supervision else ""
    target.attrs["camera_pose_convention"] = "isaac_x_forward_y_left_z_up"
    return total, episode_ends, cameras


def convert(source: Path, output: Path, max_demos: int | None, use_scene: bool, include_failed: bool) -> None:
    source = source.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with h5py.File(temporary, "w") as target:
            if (source / "meta" / "info.json").is_file() and (source / "data").is_dir():
                total, episode_ends, cameras = _convert_lerobot(source, target, max_demos, use_scene)
            else:
                total, episode_ends, cameras = _convert_hdf5(source, target, max_demos, use_scene, include_failed)

            target.create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64))
            target.attrs["camera_order"] = json.dumps(cameras)
            target.attrs["use_scene"] = bool(use_scene)
            target.attrs["state_dim"] = PROPRIO_DIM
            target.attrs["action_dim"] = ACTION_DIM
            target.attrs["quaternion_order"] = "xyzw"
            target.attrs["source"] = str(source.resolve())
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"[GauDP] wrote {output} ({total} frames, {len(episode_ends)} episodes, cameras={cameras})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-demos", type=int)
    parser.add_argument("--use-scene", action="store_true")
    parser.add_argument("--include-failed", action="store_true")
    args = parser.parse_args()
    convert(args.source, args.output, args.max_demos, args.use_scene, args.include_failed)


if __name__ == "__main__":
    main()
