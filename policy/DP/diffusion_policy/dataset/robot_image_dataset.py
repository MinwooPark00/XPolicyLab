from typing import Dict
import os
import numba
import torch
import numpy as np
import zarr
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

def _cam_obs_key(zarr_key: str) -> str:
    """zarr array name -> obs dict / shape_meta key, e.g. "head_camera" -> "head_cam".

    Legacy fallback for zarrs written by the old per-robot process_data.py,
    which have no "camera_map" attr. MHBench zarrs (scripts/data_convertion.py)
    carry a camera_map instead -- see _build_cam_obs_names.
    """
    return zarr_key.replace("_camera", "_cam")


EGO_TO_SLOT = {"ego_a": "left_cam", "ego_b": "right_cam"}
"""model.py's MHBENCH_CAMERA_SLOT, inverted to the obs-dict-key side: which
shape_meta camera key a given ego view lands under for the centralized
(2-camera) case."""


def _build_cam_obs_names(camera_map: dict) -> dict:
    """zarr camera key (e.g. "img") -> obs dict / shape_meta key, from a
    zarr's own "camera_map" attr (e.g. {"img": "ego_a"}).

    One entry means a decentralized, single-camera checkpoint -- that camera
    becomes "head_cam", matching model.py's _encode_mhbench_robot_obs. Two
    entries means the centralized checkpoint -- ego_a/ego_b become
    left_cam/right_cam via EGO_TO_SLOT, matching model.py's MHBENCH_CAMERA_SLOT
    and encode_obs's mhbench_dual_robot branch.
    """
    if not camera_map:
        return {}
    if len(camera_map) == 1:
        return {next(iter(camera_map)): "head_cam"}
    return {zarr_key: EGO_TO_SLOT[physical] for zarr_key, physical in camera_map.items()}


def _read_camera_map(zarr_path) -> dict:
    """A zarr's own root-level "camera_map" attr.

    ReplayBuffer.copy_from_path's default (in-memory) backend copies only
    "data"/"meta" arrays, not root group attrs -- self.replay_buffer.root ends
    up a plain dict with no .attrs at all -- so this reads the attr straight
    from the source store instead of going through the replay buffer.
    """
    return dict(zarr.open(os.path.expanduser(zarr_path), mode="r").attrs.get("camera_map", {}))


class RobotImageDataset(BaseImageDataset):

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=128,
        max_train_episodes=None,
        val_zarr_path=None,
    ):

        super().__init__()
        self.val_zarr_path = val_zarr_path
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=None)
        self.camera_keys = sorted(
            k for k in self.replay_buffer.keys() if k not in ("state", "action")
        )
        self.cam_obs_names = _build_cam_obs_names(_read_camera_map(zarr_path))

        if val_zarr_path:
            # A held-out split (e.g. from `scripts/data_convertion.py --split
            # val`) rather than a random carve of this same zarr -- every
            # episode here is training.
            train_mask = np.ones(self.replay_buffer.n_episodes, dtype=bool)
        else:
            val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
            train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    def _obs_key(self, zarr_key: str) -> str:
        return self.cam_obs_names.get(zarr_key, _cam_obs_key(zarr_key))

    def get_validation_dataset(self):
        if self.val_zarr_path:
            val_set = copy.copy(self)
            val_set.replay_buffer = ReplayBuffer.copy_from_path(self.val_zarr_path, keys=None)
            val_set.cam_obs_names = _build_cam_obs_names(_read_camera_map(self.val_zarr_path))
            val_set.train_mask = np.ones(val_set.replay_buffer.n_episodes, dtype=bool)
            val_set.sampler = SequenceSampler(
                replay_buffer=val_set.replay_buffer,
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=val_set.train_mask,
            )
            return val_set

        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for cam_key in self.camera_keys:
            normalizer[self._obs_key(cam_key)] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"].astype(np.float32)  # (agent_posx2, block_posex3)
        obs = {self._obs_key(k): np.moveaxis(sample[k], -1, 1) / 255 for k in self.camera_keys}  # T, 3, H, W each
        obs["agent_pos"] = agent_pos  # T, D
        data = {
            "obs": obs,
            "action": sample["action"].astype(np.float32),  # T, D
        }
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError  # Specialized
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        agent_pos = samples["state"].to(device, non_blocking=True)
        obs = {
            self._obs_key(k): samples[k].to(device, non_blocking=True).movedim(-1, 2) / 255.0  # B, T, 3, H, W each
            for k in self.camera_keys
        }
        obs["agent_pos"] = agent_pos  # B, T, D
        action = samples["action"].to(device, non_blocking=True)
        return {
            "obs": obs,
            "action": action,  # B, T, D
        }


def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
# No parallel=True variant: numba's prange spins up its own thread pool, and
# that pool does not survive the DataLoader's fork() into worker processes --
# a worker calling into it segfaults (PyTorch reports this as "Unexpected
# segmentation fault encountered in worker"). robot_dp.yaml's
# dataloader.num_workers already parallelizes across batches at the process
# level, so intra-batch threading here is redundant on top of that anyway.


def batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)
