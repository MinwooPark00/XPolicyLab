"""Lazy HDF5 datasets shared by Gaussian and policy training."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .schema import ACTION_DIM, ACTION_SCHEMA, PROPRIO_DIM, STATE_SCHEMA, pose7_xyzw_to_matrix

IMAGE_SIZE = (240, 320)
_OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
_OPENCV_TO_ISAAC_CAMERA = np.asarray(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

_HANDOVER_DATASET_NAMES = frozenset(("handover", "handover_easy", "handovereasy"))


def _canonical_source_path(value: str | Path) -> Path:
    """Normalize only the retired handover dataset directory spelling.

    Keeping the rest of the resolved path intact means two exports in different
    workspaces still compare unequal; this does not turn the provenance check
    into a frame-count-only check.
    """
    path = Path(value).expanduser().resolve()
    if path.name == "lerobot" and path.parent.name in _HANDOVER_DATASET_NAMES:
        return path.parent.parent / "handover" / path.name
    return path


def _legacy_ee_names(task: str) -> set[str]:
    names = {task}
    if task == "handover":
        names.update(("handover_easy", "handovereasy"))
    return {
        filename
        for name in names
        for filename in (
            f"mhbench-{name}-{name}-ee.hdf5",
            f"{name}-experiment-{name}-ee.hdf5",
        )
    }


def _episode_ranges(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    starts = np.concatenate(([0], episode_ends[:-1]))
    return [(int(start), int(end)) for start, end in zip(starts, episode_ends)]


def split_episode_ids(count: int, train: bool, ratio: float = 0.95) -> list[int]:
    if count <= 1:
        return [0]
    boundary = min(max(int(round(count * ratio)), 1), count - 1)
    return list(range(boundary)) if train else list(range(boundary, count))


def _resize_chw(image: torch.Tensor, size=IMAGE_SIZE, mode: str = "bilinear") -> torch.Tensor:
    squeeze = image.ndim == 3
    if squeeze:
        image = image.unsqueeze(0)
    result = F.interpolate(image, size=size, mode=mode, align_corners=False if mode != "nearest" else None)
    return result[0] if squeeze else result


def _read_rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Read padded windows without relying on h5py's strict fancy-index rules."""
    return np.stack([dataset[int(index)] for index in indices], axis=0)


class _LazyH5Dataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.path = str(Path(path).resolve())
        self._file: h5py.File | None = None
        with h5py.File(self.path, "r") as source:
            self.episode_ends = np.asarray(source["episode_ends"], dtype=np.int64)
            self.camera_order = json.loads(source.attrs["camera_order"])
            self.source_format = str(source.attrs.get("source_format", "mhbench-hdf5"))
            self.gaussian_supervision = bool(source.attrs.get("gaussian_supervision", True))
            self.camera_pose_convention = str(source.attrs.get("camera_pose_convention", "opengl"))
            count = len(self.episode_ends)
            if "train_mask" in source and "val_mask" in source:
                self.train_mask = np.asarray(source["train_mask"], dtype=bool)
                self.val_mask = np.asarray(source["val_mask"], dtype=bool)
                if self.train_mask.shape != (count,) or self.val_mask.shape != (count,):
                    raise ValueError("train_mask/val_mask must contain one value per episode")
                if np.any(self.train_mask & self.val_mask):
                    raise ValueError("train_mask and val_mask overlap")
                if not self.train_mask.any() or not self.val_mask.any():
                    raise ValueError("train_mask and val_mask must both contain at least one episode")
                self.split_source = str(source.attrs.get("split_source", "dataset"))
            else:
                train_ids = split_episode_ids(count, train=True)
                val_ids = split_episode_ids(count, train=False)
                self.train_mask = np.zeros(count, dtype=bool)
                self.train_mask[train_ids] = True
                self.val_mask = np.zeros(count, dtype=bool)
                self.val_mask[val_ids] = True
                self.split_source = "95:5-fallback"
            if int(source.attrs["state_dim"]) != PROPRIO_DIM or int(source.attrs["action_dim"]) != ACTION_DIM:
                raise ValueError("dataset does not follow GauDP's 86D state / 70D joint-action contract")
            if str(source.attrs.get("action_type", "")) != "joint":
                raise ValueError("dataset is not marked action_type=joint; regenerate it with process_data.sh")
            state_schema = tuple(json.loads(source.attrs.get("state_schema", "[]")))
            action_schema = tuple(json.loads(source.attrs.get("action_schema", "[]")))
            if state_schema != STATE_SCHEMA or action_schema != ACTION_SCHEMA:
                raise ValueError(
                    f"dataset schema differs from centralized GR00T: state={state_schema}, action={action_schema}"
                )

    def split_ids(self, train: bool) -> list[int]:
        mask = self.train_mask if train else self.val_mask
        return np.flatnonzero(mask).tolist()

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        return state

    def __del__(self):
        handle = getattr(self, "_file", None)
        self._file = None
        if handle is not None:
            try:
                handle.close()
            except (TypeError, ValueError):
                # h5py module globals may already be cleared during Python
                # interpreter shutdown.
                pass


class GauDPSequenceDataset(_LazyH5Dataset):
    def __init__(
        self,
        path: str | Path,
        train: bool,
        horizon: int = 8,
        n_obs_steps: int = 1,
        gaussian_features: str | Path | None = None,
    ) -> None:
        super().__init__(path)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if not 1 <= self.n_obs_steps <= self.horizon:
            raise ValueError("n_obs_steps must be between 1 and horizon")
        if gaussian_features is None:
            raise ValueError("offline Gaussian feature file is required for GauDP policy training")
        self.gaussian_features_path = str(Path(gaussian_features).resolve())
        self._gaussian_file: h5py.File | None = None
        with h5py.File(self.gaussian_features_path, "r") as features:
            expected_shape = (int(self.episode_ends[-1]), len(self.camera_order), 13, *IMAGE_SIZE)
            if "gaussian_features" not in features:
                raise ValueError(f"missing gaussian_features dataset in {self.gaussian_features_path}")
            if tuple(features["gaussian_features"].shape) != expected_shape:
                raise ValueError(
                    f"Gaussian feature shape mismatch: expected {expected_shape}, "
                    f"got {tuple(features['gaussian_features'].shape)}"
                )
            feature_cameras = json.loads(features.attrs["camera_order"])
            if feature_cameras != self.camera_order:
                raise ValueError(
                    f"Gaussian feature camera order {feature_cameras} does not match dataset {self.camera_order}"
                )
            if not bool(features.attrs.get("complete", False)):
                raise ValueError(f"Gaussian feature extraction is incomplete: {self.gaussian_features_path}")
            self._validate_feature_source(features)
            self.gaussian_checkpoint = str(features.attrs.get("gaussian_checkpoint", ""))
        ranges = _episode_ranges(self.episode_ends)
        self.ranges = [ranges[i] for i in self.split_ids(train)]
        self.samples = [(episode, t) for episode, (start, end) in enumerate(self.ranges) for t in range(start, end)]

    def _validate_feature_source(self, features: h5py.File) -> None:
        """Allow an EE-era cache only when its image timeline is identical."""
        recorded = Path(str(features.attrs.get("source_data", ""))).expanduser()
        current = Path(self.path)
        if recorded.resolve() == current.resolve():
            return
        if not recorded.is_file():
            # Pre-joint caches point at the converted ``*-ee.hdf5`` file. That
            # intermediate may have been deleted after the new joint file was
            # generated even though both came from the same LeRobot export.
            # Accept only the narrow, independently verifiable case: same
            # GauDP data directory, legacy EE filename for this task, and a
            # complete LeRobot export whose declared frame/episode counts are
            # exactly the new conversion's timeline. Shape and camera order
            # were already checked against the cache above.
            with h5py.File(current, "r") as new:
                raw_source = Path(str(new.attrs.get("source", ""))).expanduser().resolve()
                episode_ids = np.asarray(new.get("episode_ids", []), dtype=np.int64)
            canonical_source = _canonical_source_path(raw_source)
            raw_info = raw_source / "meta" / "info.json"
            canonical_info = canonical_source / "meta" / "info.json"
            info_path = raw_info if raw_info.is_file() else canonical_info
            task = (
                canonical_source.parent.name
                if canonical_source.name == "lerobot"
                else canonical_source.name
            )
            legacy_names = _legacy_ee_names(task)
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                info = {}
            complete_ids = np.arange(len(self.episode_ends), dtype=np.int64)
            if not (
                recorded.parent.resolve() == current.parent.resolve()
                and recorded.name in legacy_names
                and int(info.get("total_frames", -1)) == int(self.episode_ends[-1])
                and int(info.get("total_episodes", -1)) == len(self.episode_ends)
                and np.array_equal(episode_ids, complete_ids)
            ):
                raise ValueError(
                    f"Gaussian cache records unavailable source data {recorded}; "
                    "cannot prove image compatibility"
                )
            return
        with h5py.File(recorded, "r") as old:
            old_cameras = json.loads(old.attrs["camera_order"])
            old_ends = np.asarray(old["episode_ends"], dtype=np.int64)
            old_source = str(old.attrs.get("source", ""))
        with h5py.File(current, "r") as new:
            new_source = str(new.attrs.get("source", ""))
        if old_cameras != self.camera_order or not np.array_equal(old_ends, self.episode_ends):
            raise ValueError(
                "Gaussian cache source has different cameras or episode boundaries; re-extract features"
            )
        if (
            not old_source
            or not new_source
            or _canonical_source_path(old_source) != _canonical_source_path(new_source)
        ):
            raise ValueError(
                "Gaussian cache and joint dataset were not converted from the same source dataset"
            )

    @property
    def gaussian_file(self) -> h5py.File:
        if self._gaussian_file is None:
            self._gaussian_file = h5py.File(self.gaussian_features_path, "r")
        return self._gaussian_file

    def __getstate__(self):
        state = super().__getstate__()
        state["_gaussian_file"] = None
        return state

    def __del__(self):
        super().__del__()
        handle = getattr(self, "_gaussian_file", None)
        self._gaussian_file = None
        if handle is not None:
            try:
                handle.close()
            except (TypeError, ValueError):
                pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode, current = self.samples[index]
        start, end = self.ranges[episode]
        first = current - (self.n_obs_steps - 1)
        action_indices = np.clip(np.arange(first, first + self.horizon), start, end - 1)
        observation_indices = action_indices[: self.n_obs_steps]
        state = torch.from_numpy(
            _read_rows(self.file["state"], observation_indices).astype(np.float32)
        )
        action = torch.from_numpy(
            _read_rows(self.file["action"], action_indices).astype(np.float32)
        )
        images = []
        for camera_index in range(len(self.camera_order)):
            frames = _read_rows(
                self.file[f"rgb_{camera_index}"], observation_indices
            ).astype(np.uint8)
            tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
            images.append(_resize_chw(tensor))
        gaussian_features = torch.from_numpy(
            _read_rows(self.gaussian_file["gaussian_features"], observation_indices)
        )
        return {
            "images": torch.stack(images, dim=1),
            "state": state,
            "action": action,
            "gaussian_features": gaussian_features,
        }

    def normalization_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        chunks = [slice(start, end) for start, end in self.ranges]
        return (
            np.concatenate([np.asarray(self.file["state"][chunk], dtype=np.float32) for chunk in chunks]),
            np.concatenate([np.asarray(self.file["action"][chunk], dtype=np.float32) for chunk in chunks]),
        )


class GaussianFrameDataset(_LazyH5Dataset):
    def __init__(self, path: str | Path, train: bool) -> None:
        super().__init__(path)
        required = [
            f"{field}_{camera_index}"
            for camera_index in range(len(self.camera_order))
            for field in ("depth", "intrinsics", "pose")
        ]
        with h5py.File(self.path, "r") as source:
            missing = [key for key in required if key not in source]
        if not self.gaussian_supervision or missing:
            raise ValueError(
                f"{self.source_format} data has RGB but no Gaussian reconstruction supervision "
                f"(missing {missing}). train_gaussian.sh and eval_gaussian.sh require depth, "
                "camera intrinsics, and camera poses in the converted dataset. "
                "Use a pretrained/already fine-tuned NoPoSplat checkpoint with "
                "extract_gaussian_features.sh instead."
            )
        ranges = _episode_ranges(self.episode_ends)
        # Kept, not just consumed: `eval_gaussian.py --dump-recon` picks frames
        # at fractions through each episode, which needs the episode boundaries
        # in the same order `indices` concatenates them.
        self.episode_ids = self.split_ids(train)
        self.episode_ranges = [ranges[i] for i in self.episode_ids]
        self.indices = [index for start, end in self.episode_ranges for index in range(start, end)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = self.indices[item]
        images, depths, intrinsics, world_from_camera = [], [], [], []
        for camera_index in range(len(self.camera_order)):
            rgb = np.asarray(self.file[f"rgb_{camera_index}"][index], dtype=np.uint8)
            depth = np.asarray(self.file[f"depth_{camera_index}"][index], dtype=np.float32).squeeze()
            intrinsic = np.asarray(self.file[f"intrinsics_{camera_index}"][index], dtype=np.float32).reshape(3, 3)
            pose = np.asarray(self.file[f"pose_{camera_index}"][index], dtype=np.float32).reshape(7)

            height, width = rgb.shape[:2]
            intrinsic[0] /= float(width)
            intrinsic[1] /= float(height)
            if self.camera_pose_convention == "opengl":
                camera_basis = _OPENGL_TO_OPENCV
            elif self.camera_pose_convention == "isaac_x_forward_y_left_z_up":
                # LeRobot's observation.camera_pose is the Isaac camera actor
                # pose. OpenCV (right, down, forward) maps to actor
                # coordinates as (z, -x, -y), matching the dataset contract.
                camera_basis = _OPENCV_TO_ISAAC_CAMERA
            else:
                raise ValueError(f"unsupported camera pose convention {self.camera_pose_convention!r}")
            camera_matrix = pose7_xyzw_to_matrix(pose) @ camera_basis

            image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
            depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
            images.append(_resize_chw(image_tensor))
            depths.append(_resize_chw(depth_tensor, mode="nearest")[0, 0])
            intrinsics.append(torch.from_numpy(intrinsic.copy()))
            world_from_camera.append(torch.from_numpy(camera_matrix))

        extrinsics = torch.stack(world_from_camera)
        canonical_from_world = torch.linalg.inv(extrinsics[0])
        extrinsics = canonical_from_world.unsqueeze(0) @ extrinsics
        depth_stack = torch.stack(depths)
        valid = torch.isfinite(depth_stack) & (depth_stack > 0)
        if valid.any():
            near = max(float(depth_stack[valid].amin()) * 0.8, 0.01)
            far = max(float(depth_stack[valid].amax()) * 1.2, near + 0.1)
        else:
            near, far = 0.05, 20.0
        views = len(images)
        return {
            "images": torch.stack(images),
            "depth": depth_stack,
            "intrinsics": torch.stack(intrinsics),
            "extrinsics": extrinsics,
            "near": torch.full((views,), near, dtype=torch.float32),
            "far": torch.full((views,), far, dtype=torch.float32),
        }


class GaussianImageDataset(_LazyH5Dataset):
    """All converted RGB frames in global order for offline feature extraction."""

    def __len__(self) -> int:
        return int(self.episode_ends[-1])

    def __getitem__(self, index: int) -> torch.Tensor:
        images = []
        for camera_index in range(len(self.camera_order)):
            rgb = np.asarray(self.file[f"rgb_{camera_index}"][index], dtype=np.uint8)
            image = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
            images.append(_resize_chw(image))
        return torch.stack(images)
