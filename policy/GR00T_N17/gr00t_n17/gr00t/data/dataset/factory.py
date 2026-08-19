# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from gr00t.configs.base_config import Config
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.stats import generate_rel_stats, generate_stats
from gr00t.experiment.dist_utils import barrier


def _parse_split_range(spec: str, num_episodes: int) -> list[int]:
    """Turn a LeRobot ``splits`` entry (``"0:50"``, ``"12"``) into episode positions."""
    spec = str(spec).strip()
    if ":" in spec:
        start_text, _, end_text = spec.partition(":")
        start = int(start_text) if start_text.strip() else 0
        end = int(end_text) if end_text.strip() else num_episodes
    else:
        start = int(spec)
        end = start + 1
    start = max(0, min(start, num_episodes))
    end = max(start, min(end, num_episodes))
    return list(range(start, end))


def resolve_episode_splits(
    dataset_path: str | Path, eval_set_split_ratio: float
) -> tuple[list[int] | None, list[int] | None]:
    """Return ``(train_episodes, eval_episodes)`` for one LeRobot dataset directory.

    The dataset's own ``meta/info.json`` ``splits`` is the authority (LeRobot v2.1
    keeps ``{"train": "0:50", "val": "50:65"}`` there); re-deriving the boundary
    would let the two disagree. Falls back to the last ``eval_set_split_ratio`` of
    episodes, and returns ``(None, None)`` when there is nothing to hold out.
    """
    info_path = Path(dataset_path) / "meta" / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
    num_episodes = int(info["total_episodes"])

    splits = info.get("splits") or {}
    eval_key = next((k for k in ("val", "validation", "eval", "test") if k in splits), None)
    if eval_key is not None:
        eval_episodes = _parse_split_range(splits[eval_key], num_episodes)
        if "train" in splits:
            train_episodes = _parse_split_range(splits["train"], num_episodes)
        else:
            held_out = set(eval_episodes)
            train_episodes = [i for i in range(num_episodes) if i not in held_out]
        overlap = sorted(set(train_episodes) & set(eval_episodes))
        assert not overlap, (
            f"{info_path} declares overlapping train/{eval_key} splits: episodes {overlap}"
        )
        source = f"meta/info.json splits ({eval_key})"
    else:
        num_eval = int(math.floor(num_episodes * eval_set_split_ratio))
        if num_eval < 1:
            return None, None
        train_episodes = list(range(num_episodes - num_eval))
        eval_episodes = list(range(num_episodes - num_eval, num_episodes))
        source = f"eval_set_split_ratio={eval_set_split_ratio}"

    if not eval_episodes or not train_episodes:
        return None, None
    logging.info(
        f"[DatasetFactory] {dataset_path}: {len(train_episodes)} train / "
        f"{len(eval_episodes)} eval episodes from {source}"
    )
    return train_episodes, eval_episodes


class DatasetFactory:
    """
    Factory class for building training datasets. Model-agnostic.
    """

    def __init__(self, config: Config):
        self.config = config

    def build(
        self, processor: BaseProcessor
    ) -> tuple[ShardedMixtureDataset, ShardedMixtureDataset | None]:
        """Build the dataset. Returns a tuple of (train_dataset, eval_dataset)."""
        # Enabling eval is the only thing that moves episodes out of the training set.
        want_eval = self.config.training.eval_strategy != "no"

        all_datasets = []
        all_weights = []
        all_eval_datasets = []
        all_eval_weights = []
        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing datasets",
        ):
            datasets = []
            eval_datasets = []
            for dataset_path in dataset_spec.dataset_paths:
                embodiment_tag = dataset_spec.embodiment_tag
                assert embodiment_tag is not None, "Embodiment tag is required"
                assert self.config.data.mode == "single_turn", "Only single turn mode is supported"
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        generate_stats(dataset_path)
                        generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                else:
                    generate_stats(dataset_path)
                    generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                barrier()
                train_episodes, eval_episodes = (None, None)
                if want_eval:
                    train_episodes, eval_episodes = resolve_episode_splits(
                        dataset_path, self.config.training.eval_set_split_ratio
                    )
                    assert eval_episodes, (
                        f"eval_strategy={self.config.training.eval_strategy!r} but no validation "
                        f"episodes could be resolved for {dataset_path}: declare a validation "
                        f"range in meta/info.json's splits, or set eval_set_split_ratio > 0."
                    )
                common_kwargs = dict(
                    dataset_path=dataset_path,
                    embodiment_tag=EmbodimentTag(embodiment_tag),
                    modality_configs=self.config.data.modality_configs[embodiment_tag],
                    video_backend=self.config.data.video_backend,
                    shard_size=self.config.data.shard_size,
                    episode_sampling_rate=self.config.data.episode_sampling_rate,
                    seed=self.config.data.seed,
                    allow_padding=self.config.data.allow_padding,
                )
                datasets.append(
                    ShardedSingleStepDataset(**common_kwargs, episode_indices=train_episodes)
                )
                if eval_episodes:
                    eval_datasets.append(
                        ShardedSingleStepDataset(**common_kwargs, episode_indices=eval_episodes)
                    )
            dataset_lengths = np.array([len(dataset) for dataset in datasets])
            dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            for dataset, relative_length in zip(datasets, dataset_relative_lengths):
                weight = relative_length * dataset_spec.mix_ratio
                all_datasets.append(dataset)
                all_weights.append(weight)
            if eval_datasets:
                eval_lengths = np.array([len(dataset) for dataset in eval_datasets])
                eval_relative_lengths = eval_lengths / eval_lengths.sum()
                for dataset, relative_length in zip(eval_datasets, eval_relative_lengths):
                    all_eval_datasets.append(dataset)
                    all_eval_weights.append(relative_length * dataset_spec.mix_ratio)

        train_dataset = ShardedMixtureDataset(
            datasets=all_datasets,
            weights=all_weights,
            processor=processor,
            seed=self.config.data.seed,
            training=True,
            num_shards_per_epoch=self.config.data.num_shards_per_epoch,
            override_pretraining_statistics=self.config.data.override_pretraining_statistics,
        )
        if not all_eval_datasets:
            return train_dataset, None

        # Built second so the train mixture's statistics are the ones the shared
        # processor keeps (they are identical -- statistics are per dataset directory).
        eval_dataset = ShardedMixtureDataset(
            datasets=all_eval_datasets,
            weights=all_eval_weights,
            processor=processor,
            seed=self.config.data.seed,
            training=False,
            num_shards_per_epoch=self.config.data.num_shards_per_epoch,
            override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            max_samples=self.config.data.eval_max_samples or None,
        )
        return train_dataset, eval_dataset
