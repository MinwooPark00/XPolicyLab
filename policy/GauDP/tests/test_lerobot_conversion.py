import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from XPolicyLab.policy.GauDP.gaudp.dataset import (
    GaussianFrameDataset,
    _LazyH5Dataset,
    _OPENCV_TO_ISAAC_CAMERA,
)
from XPolicyLab.policy.GauDP.gaudp.schema import (
    ACTION_DIM,
    ACTION_SCHEMA,
    PROPRIO_DIM,
    STATE_SCHEMA,
)
from XPolicyLab.policy.GauDP.process_data import (
    _camera_intrinsics,
    _direct_modality,
    _lerobot_state_action,
    _official_split_masks,
)

# The contract GauDP restates lives in MHBench's GR00T config, which is pure
# data with no imports -- load it by path so this test compares against the
# source of truth rather than a second copy of the numbers.
_KEYS_PY = Path(__file__).resolve().parents[5] / "configs" / "gr00t" / "mhbench_keys.py"
if not _KEYS_PY.is_file():  # a standalone GauDP checkout, without MHBench around it
    pytest.skip(f"{_KEYS_PY} is not available", allow_module_level=True)
_spec = importlib.util.spec_from_file_location("mhbench_keys", _KEYS_PY)
mhbench_keys = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mhbench_keys)

ROBOTS = mhbench_keys.ROBOTS
JOINTS_PER_ROBOT = mhbench_keys.JOINTS_PER_ROBOT


def _joint_bounds() -> dict[str, tuple[int, int]]:
    """`urdf_order`'s group bounds, from the widths the config declares."""
    bounds, start = {}, 0
    for group in mhbench_keys.JOINT_GROUPS:
        width = mhbench_keys.JOINT_GROUP_WIDTHS[group]
        bounds[group] = (start, start + width)
        start += width
    assert start == JOINTS_PER_ROBOT
    return bounds


def _modality() -> dict:
    """The subset of `export_lerobot.build_modality` GauDP reads."""
    bounds = _joint_bounds()
    state, action = {}, {}
    for index, robot in enumerate(ROBOTS):
        base = index * JOINTS_PER_ROBOT
        for group, (lo, hi) in bounds.items():
            state[f"{robot}_{group}"] = {"start": base + lo, "end": base + hi}
            action[f"{robot}_{group}"] = {"start": base + lo, "end": base + hi}
        action[f"{robot}_navigate_command"] = {
            "start": index * 3,
            "end": index * 3 + 3,
            "original_key": "teleop.navigate_command",
        }
        action[f"{robot}_base_height_command"] = {
            "start": index,
            "end": index + 1,
            "original_key": "teleop.base_height_command",
        }
    return {"state": state, "action": action}


def _columns(frames: int = 3) -> tuple[dict[str, np.ndarray], dict]:
    """Distinct, decodable values -- column c of key k is 1000*k + c."""
    widths = {
        "observation.state": len(ROBOTS) * JOINTS_PER_ROBOT,
        "action": len(ROBOTS) * JOINTS_PER_ROBOT,
        "teleop.navigate_command": len(ROBOTS) * 3,
        "teleop.base_height_command": len(ROBOTS),
    }
    columns, features = {}, {}
    for offset, (key, width) in enumerate(widths.items()):
        row = np.arange(width, dtype=np.float32) + 1000.0 * (offset + 1)
        columns[key] = np.repeat(row[None], frames, axis=0)
        features[key] = {"names": [f"{key}/{i}" for i in range(width)]}
    return columns, {"features": features}


def test_lerobot_named_slices_produce_the_86d_70d_groot_contract():
    columns, info = _columns()
    state, action = _lerobot_state_action(columns, info, _modality())

    assert state.shape == (3, PROPRIO_DIM)
    assert action.shape == (3, ACTION_DIM)
    assert STATE_SCHEMA == mhbench_keys.JOINT_GROUPS
    assert ACTION_SCHEMA == (
        *mhbench_keys.ACTION_JOINT_GROUPS,
        *mhbench_keys.ACTION_COMMAND_KEYS,
    )

    bounds = _joint_bounds()
    for index, robot in enumerate(ROBOTS):
        joint_base = index * JOINTS_PER_ROBOT

        # State is the URDF order verbatim, so it is the source row unchanged.
        np.testing.assert_array_equal(
            state[0, index * JOINTS_PER_ROBOT : (index + 1) * JOINTS_PER_ROBOT],
            columns["observation.state"][0, joint_base : joint_base + JOINTS_PER_ROBOT],
        )

        # Action reorders: arms before hands, then waist, height, navigate.
        offset = index * mhbench_keys.ACTION_DIMS_PER_ROBOT
        for group in mhbench_keys.ACTION_JOINT_GROUPS:
            lo, hi = bounds[group]
            width = hi - lo
            np.testing.assert_array_equal(
                action[0, offset : offset + width],
                columns["action"][0, joint_base + lo : joint_base + hi],
            )
            offset += width
        np.testing.assert_array_equal(
            action[0, offset : offset + 1],
            columns["teleop.base_height_command"][0, index : index + 1],
        )
        offset += 1
        np.testing.assert_array_equal(
            action[0, offset : offset + 3],
            columns["teleop.navigate_command"][0, index * 3 : index * 3 + 3],
        )


def test_raw_hdf5_uses_the_same_slices_the_exporter_would_write():
    """The raw path has no `meta/modality.json`, so it builds its own.

    `export_lerobot.episode_columns` lays `action` out exactly like
    `observation.state` -- each robot's 43 joint targets in URDF order, taken
    from `processed_actions` -- and the two locomotion commands sit in their
    own `teleop.*` columns. `_direct_modality` has to say the same thing the
    exporter's `build_modality` does, or the two conversion paths produce
    differently-ordered files under one contract.
    """
    direct = _direct_modality()
    expected = _modality()
    wanted = {
        "state": [f"{robot}_{group}" for robot in ROBOTS for group in STATE_SCHEMA],
        "action": [f"{robot}_{group}" for robot in ROBOTS for group in ACTION_SCHEMA],
    }
    for section, keys in wanted.items():
        # Exactly the keys GauDP reads -- the exporter also describes the leg
        # action columns, which no config commands and GauDP does not select.
        assert sorted(direct[section]) == sorted(keys)
        for key in keys:
            assert direct[section][key] == expected[section][key], f"{section}.{key}"

    # And the conversion itself agrees with the modality-driven LeRobot path.
    columns, info = _columns()
    np.testing.assert_array_equal(
        np.concatenate(_lerobot_state_action(columns, info, direct), axis=1),
        np.concatenate(_lerobot_state_action(columns, info, expected), axis=1),
    )


def test_conversion_refuses_a_dataset_without_the_named_slices():
    columns, info = _columns()
    modality = _modality()
    del modality["state"]["robot_b_right_hand"]
    with pytest.raises(KeyError, match="robot_b_right_hand"):
        _lerobot_state_action(columns, info, modality)

    columns, info = _columns()
    columns.pop("teleop.base_height_command")
    with pytest.raises(KeyError, match="teleop.base_height_command"):
        _lerobot_state_action(columns, info, _modality())


def _joint_attrs(target: h5py.File) -> None:
    target.attrs["state_dim"] = PROPRIO_DIM
    target.attrs["action_dim"] = ACTION_DIM
    target.attrs["action_type"] = "joint"
    target.attrs["state_schema"] = json.dumps(STATE_SCHEMA)
    target.attrs["action_schema"] = json.dumps(ACTION_SCHEMA)


def test_rgb_only_conversion_rejects_reconstruction_but_not_images(tmp_path):
    path = tmp_path / "rgb_only.hdf5"
    with h5py.File(path, "w") as target:
        target.create_dataset("episode_ends", data=np.asarray([1], dtype=np.int64))
        target.create_dataset("rgb_0", data=np.zeros((1, 2, 2, 3), dtype=np.uint8))
        target.create_dataset("rgb_1", data=np.zeros((1, 2, 2, 3), dtype=np.uint8))
        target.attrs["camera_order"] = json.dumps(["ego_a", "ego_b"])
        target.attrs["source_format"] = "lerobot-v2.1"
        target.attrs["gaussian_supervision"] = False
        _joint_attrs(target)

    with pytest.raises(ValueError, match="no Gaussian reconstruction supervision"):
        GaussianFrameDataset(path, train=True)


def test_an_ee_era_dataset_is_refused_rather_than_read_as_joint_space(tmp_path):
    path = tmp_path / "legacy_ee.hdf5"
    with h5py.File(path, "w") as target:
        target.create_dataset("episode_ends", data=np.asarray([1, 2], dtype=np.int64))
        target.attrs["camera_order"] = json.dumps(["ego_a", "ego_b"])
        target.attrs["state_dim"] = 42
        target.attrs["action_dim"] = 44
    with pytest.raises(ValueError, match="86D state / 70D joint-action"):
        _LazyH5Dataset(path)

    path = tmp_path / "unmarked.hdf5"
    with h5py.File(path, "w") as target:
        target.create_dataset("episode_ends", data=np.asarray([1, 2], dtype=np.int64))
        target.attrs["camera_order"] = json.dumps(["ego_a", "ego_b"])
        target.attrs["state_dim"] = PROPRIO_DIM
        target.attrs["action_dim"] = ACTION_DIM
    with pytest.raises(ValueError, match="action_type=joint"):
        _LazyH5Dataset(path)


def test_current_lerobot_camera_metadata_and_axis_contract():
    matrix = [[160.0, 0.0, 160.0], [0.0, 160.0, 120.0], [0.0, 0.0, 1.0]]
    info = {
        "features": {
            "observation.depth.ego_a": {"info": {"camera.intrinsics": matrix}},
        }
    }
    np.testing.assert_array_equal(_camera_intrinsics(info, "ego_a"), matrix)

    # Dataset contract: OpenCV [right, down, forward] -> Isaac actor
    # [forward, left, up] = [z, -x, -y].
    right = _OPENCV_TO_ISAAC_CAMERA[:3, :3] @ np.asarray([1.0, 0.0, 0.0])
    down = _OPENCV_TO_ISAAC_CAMERA[:3, :3] @ np.asarray([0.0, 1.0, 0.0])
    forward = _OPENCV_TO_ISAAC_CAMERA[:3, :3] @ np.asarray([0.0, 0.0, 1.0])
    np.testing.assert_array_equal(right, [0.0, -1.0, 0.0])
    np.testing.assert_array_equal(down, [0.0, 0.0, -1.0])
    np.testing.assert_array_equal(forward, [1.0, 0.0, 0.0])


def test_official_split_masks_follow_episode_ids_after_selection():
    info = {"total_episodes": 60, "splits": {"train": "0:50", "val": "50:60"}}
    train, val = _official_split_masks(info, [0, 10, 49, 50, 59])
    np.testing.assert_array_equal(train, [True, True, True, False, False])
    np.testing.assert_array_equal(val, [False, False, False, True, True])
    assert _official_split_masks(info, list(range(10))) is None


def test_converted_split_masks_override_legacy_95_5(tmp_path):
    path = tmp_path / "split.hdf5"
    with h5py.File(path, "w") as target:
        target.create_dataset("episode_ends", data=np.asarray([1, 2, 3, 4], dtype=np.int64))
        target.create_dataset("train_mask", data=np.asarray([True, True, False, False]))
        target.create_dataset("val_mask", data=np.asarray([False, False, True, True]))
        target.attrs["split_source"] = "meta/info.json"
        target.attrs["camera_order"] = json.dumps(["ego_a", "ego_b"])
        _joint_attrs(target)
    dataset = _LazyH5Dataset(path)
    assert dataset.split_ids(train=True) == [0, 1]
    assert dataset.split_ids(train=False) == [2, 3]
    assert dataset.split_source == "meta/info.json"


# --- reusing a Gaussian feature cache built before the joint switch ---------

_CAMERAS = ["ego_a", "ego_b"]
_EPISODE_ENDS = [2, 4]


def _write_converted(path, *, source, episode_ends=_EPISODE_ENDS, cameras=_CAMERAS, joint=True):
    frames = episode_ends[-1]
    with h5py.File(path, "w") as target:
        target.create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64))
        target.create_dataset("episode_ids", data=np.arange(len(episode_ends), dtype=np.int64))
        target.create_dataset("state", data=np.zeros((frames, PROPRIO_DIM), np.float32))
        target.create_dataset("action", data=np.zeros((frames, ACTION_DIM), np.float32))
        for index in range(len(cameras)):
            target.create_dataset(f"rgb_{index}", data=np.zeros((frames, 4, 4, 3), np.uint8))
        target.attrs["camera_order"] = json.dumps(cameras)
        target.attrs["source"] = str(source)
        if joint:
            _joint_attrs(target)
        else:
            target.attrs["state_dim"] = 42
            target.attrs["action_dim"] = 44
    return path


def _write_cache(path, *, source_data, episode_ends=_EPISODE_ENDS, cameras=_CAMERAS):
    from XPolicyLab.policy.GauDP.gaudp.dataset import IMAGE_SIZE

    shape = (episode_ends[-1], len(cameras), 13, *IMAGE_SIZE)
    with h5py.File(path, "w") as target:
        target.create_dataset("gaussian_features", data=np.zeros(shape, np.float16))
        target.attrs["camera_order"] = json.dumps(cameras)
        target.attrs["source_data"] = str(source_data)
        target.attrs["gaussian_checkpoint"] = "/somewhere/gaussian/best.ckpt"
        target.attrs["complete"] = True
    return path


def _sequence_dataset(data, cache):
    from XPolicyLab.policy.GauDP.gaudp.dataset import GauDPSequenceDataset

    return GauDPSequenceDataset(data, True, horizon=2, n_obs_steps=1, gaussian_features=cache)


def test_sequence_loader_reads_only_observations_but_keeps_the_action_horizon(
    tmp_path, monkeypatch
):
    from XPolicyLab.policy.GauDP.gaudp import dataset as dataset_module

    export = tmp_path / "lerobot"
    data = _write_converted(tmp_path / "joint.hdf5", source=export)
    cache = _write_cache(tmp_path / "features.hdf5", source_data=data)
    sequence = _sequence_dataset(data, cache)

    reads = []
    original_read_rows = dataset_module._read_rows

    def record_read_rows(source, indices):
        reads.append((source.name, np.asarray(indices).copy()))
        return original_read_rows(source, indices)

    monkeypatch.setattr(dataset_module, "_read_rows", record_read_rows)
    sample = sequence[0]

    assert sample["images"].shape == (1, 2, 3, *dataset_module.IMAGE_SIZE)
    assert sample["state"].shape == (1, PROPRIO_DIM)
    assert sample["action"].shape == (2, ACTION_DIM)
    assert sample["gaussian_features"].shape == (
        1,
        len(_CAMERAS),
        13,
        *dataset_module.IMAGE_SIZE,
    )
    read_lengths = {name: len(indices) for name, indices in reads}
    assert read_lengths["/state"] == 1
    assert read_lengths["/action"] == 2
    assert read_lengths["/rgb_0"] == 1
    assert read_lengths["/rgb_1"] == 1
    assert read_lengths["/gaussian_features"] == 1


def test_an_ee_era_feature_cache_is_reused_for_the_same_export(tmp_path):
    """94 GB of features is worth reusing, but only when it provably lines up.

    The encoder never saw a state or an action, so the joint-space switch does
    not invalidate the cache -- what would invalidate it is a different export,
    a different episode selection, or a different camera list, none of which
    the frame count alone rules out.
    """
    export = tmp_path / "lerobot"
    old = _write_converted(tmp_path / "old-ee.hdf5", source=export, joint=False)
    new = _write_converted(tmp_path / "new-joint.hdf5", source=export)
    cache = _write_cache(tmp_path / "features.hdf5", source_data=old)

    dataset = _sequence_dataset(new, cache)
    assert dataset.gaussian_checkpoint == "/somewhere/gaussian/best.ckpt"
    assert len(dataset) == 2  # the 95:5 fallback keeps episode 0 for training


def test_missing_handover_easy_source_is_verified_against_renamed_export(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    export = tmp_path / "datasets" / "handover" / "lerobot"
    (export / "meta").mkdir(parents=True)
    (export / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 2, "total_frames": 4}), encoding="utf-8"
    )
    current = _write_converted(
        data_dir / "mhbench-handover-unitree_g1x2_centralized-joint.hdf5",
        source=export,
    )
    retired = data_dir / "mhbench-handover_easy-handover_easy-ee.hdf5"
    cache = _write_cache(tmp_path / "features.hdf5", source_data=retired)

    assert len(_sequence_dataset(current, cache)) == 2


def test_existing_handover_easy_source_matches_renamed_sibling_export(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    datasets = tmp_path / "datasets"
    retired_export = datasets / "handover_easy" / "lerobot"
    current_export = datasets / "handover" / "lerobot"
    old = _write_converted(data_dir / "old-ee.hdf5", source=retired_export, joint=False)
    current = _write_converted(data_dir / "new-joint.hdf5", source=current_export)
    cache = _write_cache(tmp_path / "features.hdf5", source_data=old)

    assert len(_sequence_dataset(current, cache)) == 2


def test_a_feature_cache_from_a_different_export_or_shape_is_refused(tmp_path):
    export = tmp_path / "lerobot"
    old = _write_converted(tmp_path / "old-ee.hdf5", source=export, joint=False)

    other = _write_converted(
        tmp_path / "other-joint.hdf5", source=tmp_path / "another-lerobot"
    )
    cache = _write_cache(tmp_path / "features.hdf5", source_data=old)
    with pytest.raises(ValueError, match="same source dataset"):
        _sequence_dataset(other, cache)

    # Same export, but the episodes converted from it are not the same ones.
    shifted_ends = [2, 5]
    shifted = _write_converted(
        tmp_path / "shifted.hdf5", source=export, episode_ends=shifted_ends
    )
    shifted_cache = _write_cache(
        tmp_path / "shifted-features.hdf5", source_data=old, episode_ends=shifted_ends
    )
    with pytest.raises(ValueError, match="episode boundaries"):
        _sequence_dataset(shifted, shifted_cache)

    # The cache's own source file is gone, so nothing can be proved about it.
    new = _write_converted(tmp_path / "new-joint.hdf5", source=export)
    missing_cache = _write_cache(
        tmp_path / "missing-features.hdf5", source_data=tmp_path / "deleted.hdf5"
    )
    with pytest.raises(ValueError, match="cannot prove image compatibility"):
        _sequence_dataset(new, missing_cache)


def test_a_three_view_cache_is_refused_for_a_two_view_dataset(tmp_path):
    export = tmp_path / "lerobot"
    new = _write_converted(tmp_path / "new-joint.hdf5", source=export)
    cache = _write_cache(
        tmp_path / "features.hdf5",
        source_data=new,
        cameras=[*_CAMERAS, "scene"],
    )
    with pytest.raises(ValueError, match="Gaussian feature shape mismatch"):
        _sequence_dataset(new, cache)
