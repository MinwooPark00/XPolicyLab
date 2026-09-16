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
    ACTION_GROUPS,
    ACTION_SCHEMA,
    CAMERA_SLOT,
    EGO_VIEWS,
    ROBOT_ACTION_DIM,
    ROBOT_NAMES,
    ROBOT_PROPRIO_DIM,
    SCENE_VIEW,
    STATE_GROUPS,
    STATE_SLICES,
    STATE_SCHEMA,
    action_dim,
    ego_views,
    proprio_dim,
    robot_count_from_state_dim,
    robot_names,
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


def _direct_modality(robots: tuple[str, ...] = ROBOT_NAMES) -> dict:
    """GR00T modality slices for the canonical exported float columns."""
    state, action = {}, {}
    for robot_index, robot in enumerate(robots):
        base = robot_index * ROBOT_PROPRIO_DIM
        for group, _ in STATE_GROUPS:
            sl = STATE_SLICES[group]
            state[f"{robot}_{group}"] = {"start": base + sl.start, "end": base + sl.stop}
        for group, _ in ACTION_GROUPS:
            key = f"{robot}_{group}"
            if group == "base_height_command":
                action[key] = {
                    "start": robot_index,
                    "end": robot_index + 1,
                    "original_key": "teleop.base_height_command",
                }
            elif group == "navigate_command":
                action[key] = {
                    "start": robot_index * 3,
                    "end": robot_index * 3 + 3,
                    "original_key": "teleop.navigate_command",
                }
            else:
                sl = STATE_SLICES[group]
                action[key] = {"start": base + sl.start, "end": base + sl.stop}
    return {"state": state, "action": action}


def _modality_slice(
    columns: dict[str, np.ndarray], entry: dict, default_column: str, name: str
) -> np.ndarray:
    column = str(entry.get("original_key", default_column))
    if column not in columns:
        raise KeyError(f"{name} requires missing source column {column!r}")
    start, end = int(entry["start"]), int(entry["end"])
    values = np.asarray(columns[column], dtype=np.float32)
    if values.ndim != 2 or not 0 <= start < end <= values.shape[1]:
        raise ValueError(
            f"invalid modality slice {name}: {column}[{start}:{end}] for shape {values.shape}"
        )
    return values[:, start:end]


def _joint_state_action_from_columns(
    columns: dict[str, np.ndarray], modality: dict, robots: tuple[str, ...] = ROBOT_NAMES
) -> tuple[np.ndarray, np.ndarray]:
    """Select exactly the centralized GR00T state/action modality keys."""
    states, actions = [], []
    for robot in robots:
        state_parts = []
        action_parts = []
        for group, width in STATE_GROUPS:
            key = f"{robot}_{group}"
            try:
                entry = modality["state"][key]
            except KeyError as error:
                raise KeyError(f"meta/modality.json is missing state key {key!r}") from error
            value = _modality_slice(columns, entry, "observation.state", key)
            if value.shape[1] != width:
                raise ValueError(f"{key} is {value.shape[1]}D, expected {width}D")
            state_parts.append(value)
        for group, width in ACTION_GROUPS:
            key = f"{robot}_{group}"
            try:
                entry = modality["action"][key]
            except KeyError as error:
                raise KeyError(f"meta/modality.json is missing action key {key!r}") from error
            value = _modality_slice(columns, entry, "action", key)
            if value.shape[1] != width:
                raise ValueError(f"{key} is {value.shape[1]}D, expected {width}D")
            action_parts.append(value)
        states.append(np.concatenate(state_parts, axis=-1))
        actions.append(np.concatenate(action_parts, axis=-1))
    state = np.concatenate(states, axis=-1).astype(np.float32)
    action = np.concatenate(actions, axis=-1).astype(np.float32)
    if state.shape[1] != len(robots) * ROBOT_PROPRIO_DIM or action.shape[1] != len(robots) * ROBOT_ACTION_DIM:
        raise AssertionError(f"GR00T schema mismatch: state={state.shape}, action={action.shape}")
    return state, action


def _robot_count_from_info(info: dict) -> int:
    """How many robots a LeRobot export carries, read off its declared widths.

    `observation.state` is 43D per robot; the teleop command columns are one and
    three wide per robot. All three must agree -- there is no knob to restate it.
    """
    features = info.get("features", {})

    def width(key: str) -> int:
        try:
            return int(features[key]["shape"][0])
        except (KeyError, IndexError, TypeError) as error:
            raise KeyError(f"LeRobot metadata is missing the width of {key!r}") from error

    count = robot_count_from_state_dim(width("observation.state"))
    for key, per_robot in (("teleop.base_height_command", 1), ("teleop.navigate_command", 3)):
        if width(key) != per_robot * count:
            raise ValueError(
                f"{key} is {width(key)}D but observation.state says {count} robots "
                f"({per_robot * count}D expected)"
            )
    return count


def resolve_cameras(
    features: dict, robot_count: int, use_scene: bool, override: str | None = None
) -> list[str]:
    """The views to convert: every robot's own ego camera, plus the scene if asked.

    One ego view per robot is the dataset contract, so the default is read off
    the robot count and checked against the export. `override` (--cameras,
    GAUDP_CAMERAS) may pick a different list, but never a width: state and
    action always come from the export.
    """
    declared = [view for view in EGO_VIEWS if f"observation.images.{view}" in features]
    expected = list(ego_views(robot_count))
    if declared != expected:
        raise ValueError(
            f"export declares ego views {declared} but has {robot_count} robots, whose own head "
            f"cameras are {expected}; one ego view per robot is the dataset contract"
        )
    default = expected + ([SCENE_VIEW] if use_scene else [])
    if override is None or not str(override).strip():
        return default
    cameras = [token for token in re.split(r"[,:+\s]+", str(override)) if token]
    unknown = [camera for camera in cameras if camera not in CAMERA_SLOT]
    if unknown:
        raise ValueError(f"unknown views {unknown}; known views are {sorted(CAMERA_SLOT)}")
    if len(set(cameras)) != len(cameras):
        raise ValueError(f"duplicate views in {cameras}")
    if len(cameras) < 2:
        raise ValueError(f"NoPoSplat needs at least two context views, got {cameras}")
    egos = [camera for camera in cameras if camera in EGO_VIEWS]
    if egos != sorted(egos, key=EGO_VIEWS.index):
        raise ValueError(f"ego views must stay in {list(EGO_VIEWS)} order, got {egos}")
    if use_scene and SCENE_VIEW not in cameras:
        raise ValueError("--use-scene contradicts --cameras; list 'scene' or drop --use-scene")
    missing = [camera for camera in cameras if f"observation.images.{camera}" not in features]
    if missing:
        raise ValueError(f"the export has no video for {missing}")
    return cameras


def _convert_hdf5(
    source: Path,
    target: h5py.File,
    max_demos: int | None,
    use_scene: bool,
    include_failed: bool,
) -> tuple[int, list[int], list[str], int]:
    # Two robots only: the columns come from scripts/export_lerobot.py, which in
    # this checkout lays out a pair. Three-robot data converts from its LeRobot
    # export.
    try:
        from export_lerobot import ACTION_TERM_LAYOUT, episode_columns, precheck
    except ImportError as error:
        raise SystemExit(
            "raw HDF5 joint conversion requires scripts/export_lerobot.py and its dependencies; "
            "export the dataset to LeRobot v2.1 first"
        ) from error
    cameras = list(ego_views(len(ROBOT_NAMES))) + ([SCENE_VIEW] if use_scene else [])
    with DemoSource(source) as demos:
        facts = precheck(demos, selection=None, layout=ACTION_TERM_LAYOUT)
        names = [name for name in demos.names if include_failed or bool(demos[name].attrs.get("success", True))]
        if max_demos is not None:
            names = names[:max_demos]
        if not names:
            raise SystemExit("no successful demonstrations found")

        episode_ends = []
        total = 0
        for episode_index, name in enumerate(names):
            demo = demos[name]
            columns = episode_columns(demo, facts)
            state, action = _joint_state_action_from_columns(columns, _direct_modality())
            length = state.shape[0]
            per_robot_state = [
                state[:, i * ROBOT_PROPRIO_DIM : (i + 1) * ROBOT_PROPRIO_DIM]
                for i in range(len(ROBOT_NAMES))
            ]
            if state.shape[1] != proprio_dim(len(ROBOT_NAMES)) or action.shape[1] != action_dim(len(ROBOT_NAMES)):
                raise AssertionError(f"schema mismatch: state={state.shape}, action={action.shape}")

            _append(target, "state", state)
            _append(target, "action", action)
            for robot_index, robot in enumerate(ROBOT_NAMES):
                _append(target, f"state_{robot_index}", per_robot_state[robot_index])
                _append(
                    target,
                    f"action_{robot_index}",
                    action[:, robot_index * ROBOT_ACTION_DIM : (robot_index + 1) * ROBOT_ACTION_DIM],
                )

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
    return total, episode_ends, cameras, len(ROBOT_NAMES)


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


def _lerobot_state_action(
    columns: dict[str, np.ndarray],
    info: dict,
    modality: dict | None = None,
    robots: tuple[str, ...] = ROBOT_NAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Map LeRobot v2.1 fields to the centralized joint contract (43D/35D per robot)."""
    required = (
        "observation.state",
        "action",
        "teleop.navigate_command",
        "teleop.base_height_command",
    )
    missing = [key for key in required if key not in columns]
    if missing:
        raise KeyError(f"LeRobot episode is missing required columns: {missing}")
    # ``info`` remains an argument because callers already load it and its
    # declared widths are a useful early corruption check. The authoritative
    # group ordering/slices live in meta/modality.json.
    for key in required:
        names = _feature_names(info, key)
        if len(names) != np.asarray(columns[key]).shape[1]:
            raise ValueError(
                f"{key} metadata declares {len(names)} dimensions but data has "
                f"{np.asarray(columns[key]).shape[1]}"
            )
    return _joint_state_action_from_columns(columns, modality or _direct_modality(robots), robots)


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


def _split_range(spec: str, total_episodes: int) -> set[int]:
    """Expand a LeRobot half-open episode range such as ``"0:50"``."""
    start_text, separator, end_text = str(spec).partition(":")
    if not separator:
        raise ValueError(f"invalid LeRobot split range {spec!r}; expected 'start:end'")
    start = int(start_text or 0)
    end = int(end_text or total_episodes)
    if not 0 <= start <= end <= total_episodes:
        raise ValueError(
            f"LeRobot split range {spec!r} is outside 0:{total_episodes}"
        )
    return set(range(start, end))


def _official_split_masks(info: dict, episode_ids: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    """Return train/val masks aligned to converted episodes, if both survive."""
    splits = info.get("splits") or {}
    train_spec = splits.get("train")
    val_spec = splits.get("val") or splits.get("validation")
    if not train_spec or not val_spec:
        return None
    total = int(info.get("total_episodes", max(episode_ids, default=-1) + 1))
    train_ids = _split_range(train_spec, total)
    val_ids = _split_range(val_spec, total)
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f"LeRobot train/val splits overlap at episodes {sorted(overlap)[:10]}")
    train_mask = np.asarray([episode in train_ids for episode in episode_ids], dtype=bool)
    val_mask = np.asarray([episode in val_ids for episode in episode_ids], dtype=bool)
    if not train_mask.any() or not val_mask.any():
        return None
    if np.any(~(train_mask | val_mask)):
        missing = [episode for episode, known in zip(episode_ids, train_mask | val_mask) if not known]
        raise ValueError(f"converted episodes are outside the declared train/val splits: {missing[:10]}")
    return train_mask, val_mask


def _convert_lerobot(
    source: Path,
    target: h5py.File,
    max_demos: int | None,
    use_scene: bool,
    cameras_override: str | None = None,
) -> tuple[int, list[int], list[str], int]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit("LeRobot parquet conversion requires pyarrow; rerun install.sh") from error

    info_path = source / "meta" / "info.json"
    with info_path.open(encoding="utf-8") as stream:
        info = json.load(stream)
    modality_path = source / "meta" / "modality.json"
    if not modality_path.is_file():
        raise FileNotFoundError(
            f"LeRobot dataset is missing {modality_path}; GauDP uses the same named slices as GR00T"
        )
    with modality_path.open(encoding="utf-8") as stream:
        modality = json.load(stream)
    robot_count = _robot_count_from_info(info)
    robots = robot_names(robot_count)
    for camera in list(ego_views(robot_count)) + ([SCENE_VIEW] if use_scene else []):
        key = f"observation.images.{camera}"
        if key not in info.get("features", {}):
            raise KeyError(f"LeRobot metadata is missing required video feature {key!r}")
    cameras = resolve_cameras(info.get("features", {}), robot_count, use_scene, cameras_override)
    default = list(ego_views(robot_count)) + ([SCENE_VIEW] if use_scene else [])
    print(
        f"[GauDP] robots   {robot_count} ({', '.join(robots)})  "
        f"state={proprio_dim(robot_count)}D action={action_dim(robot_count)}D"
    )
    print(f"[GauDP] cameras  {cameras}" + ("" if cameras == default else f"  (default: {default})"))

    files = sorted(source.glob("data/chunk-*/episode_*.parquet"), key=_episode_index)
    if max_demos is not None:
        files = files[:max_demos]
    if not files:
        raise SystemExit(f"no LeRobot episode parquet files found below {source / 'data'}")

    episode_ids = [_episode_index(path) for path in files]
    official_masks = _official_split_masks(info, episode_ids)
    target.create_dataset("episode_ids", data=np.asarray(episode_ids, dtype=np.int64))
    if official_masks is not None:
        train_mask, val_mask = official_masks
        target.create_dataset("train_mask", data=train_mask)
        target.create_dataset("val_mask", data=val_mask)
        target.attrs["split_source"] = "meta/info.json"
        print(
            f"[GauDP] split from meta/info.json: {int(train_mask.sum())} train / "
            f"{int(val_mask.sum())} val episodes"
        )
    elif info.get("splits"):
        target.attrs["split_source"] = "95:5-fallback"
        print(
            "[GauDP] declared split is incomplete after episode selection; "
            "falling back to a 95:5 episode split",
        )

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
    ego_depth_keys = [f"observation.depth.{camera}" for camera in cameras if camera in EGO_VIEWS]
    for key in ego_depth_keys:
        print(f"[GauDP] depth    {key}: {'present' if key in info.get('features', {}) else 'MISSING'}")
    gaussian_supervision = bool(camera_pose_names) and all(
        matrix is not None for matrix in intrinsics.values()
    ) and all(key in info.get("features", {}) for key in ego_depth_keys)
    for number, path in enumerate(files, 1):
        episode = _episode_index(path)
        table = parquet.read_table(path)
        columns = {key: np.asarray(table[key].to_pylist()) for key in table.column_names}
        state, action = _lerobot_state_action(columns, info, modality, robots)
        length = state.shape[0]
        if action.shape[0] != length:
            raise ValueError(f"{path} state/action length mismatch: {length} vs {action.shape[0]}")

        _append(target, "state", state)
        _append(target, "action", action)
        for robot_index in range(robot_count):
            _append(
                target,
                f"state_{robot_index}",
                state[:, robot_index * ROBOT_PROPRIO_DIM : (robot_index + 1) * ROBOT_PROPRIO_DIM],
            )
            _append(
                target,
                f"action_{robot_index}",
                action[:, robot_index * ROBOT_ACTION_DIM : (robot_index + 1) * ROBOT_ACTION_DIM],
            )

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
    return total, episode_ends, cameras, robot_count


def convert(
    source: Path,
    output: Path,
    max_demos: int | None,
    use_scene: bool,
    include_failed: bool,
    cameras: str | None = None,
) -> None:
    source = source.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with h5py.File(temporary, "w") as target:
            if (source / "meta" / "info.json").is_file() and (source / "data").is_dir():
                total, episode_ends, cameras, robot_count = _convert_lerobot(
                    source, target, max_demos, use_scene, cameras
                )
            else:
                if cameras:
                    raise SystemExit("--cameras applies to LeRobot exports; the raw HDF5 path keeps ego_a/ego_b")
                total, episode_ends, cameras, robot_count = _convert_hdf5(
                    source, target, max_demos, use_scene, include_failed
                )

            target.create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64))
            target.attrs["camera_order"] = json.dumps(cameras)
            target.attrs["use_scene"] = SCENE_VIEW in cameras
            target.attrs["state_dim"] = proprio_dim(robot_count)
            target.attrs["action_dim"] = action_dim(robot_count)
            target.attrs["robot_count"] = robot_count
            target.attrs["robot_names"] = json.dumps(list(robot_names(robot_count)))
            target.attrs["schema_version"] = "mhbench-gaudp-joint-v2"
            target.attrs["action_type"] = "joint"
            target.attrs["state_schema"] = json.dumps(STATE_SCHEMA)
            target.attrs["action_schema"] = json.dumps(ACTION_SCHEMA)
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
    parser.add_argument(
        "--cameras",
        default=None,
        help="views to convert, e.g. 'ego_a,ego_b'; default is every robot's ego view "
        "(plus scene with --use-scene)",
    )
    args = parser.parse_args()
    convert(args.source, args.output, args.max_demos, args.use_scene, args.include_failed, args.cameras)


if __name__ == "__main__":
    main()
