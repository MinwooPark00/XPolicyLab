#!/usr/bin/env python3
"""Convert raw MHBench demonstrations into a self-contained GauDP HDF5 file."""

from __future__ import annotations

import argparse
import json
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


def convert(source: Path, output: Path, max_demos: int | None, use_scene: bool, include_failed: bool) -> None:
    cameras = ["ego_a", "ego_b"] + (["scene"] if use_scene else [])
    if output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    with DemoSource(source) as demos, h5py.File(output, "w") as target:
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

        target.create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64))
        target.attrs["camera_order"] = json.dumps(cameras)
        target.attrs["use_scene"] = bool(use_scene)
        target.attrs["state_dim"] = PROPRIO_DIM
        target.attrs["action_dim"] = ACTION_DIM
        target.attrs["quaternion_order"] = "xyzw"
        target.attrs["source"] = str(source.resolve())
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
