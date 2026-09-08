#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gr00t.configs.base_config import get_default_config
from gr00t.configs.data.data_config import SingleDatasetConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.experiment.experiment import run
from transformers import AutoConfig

# Disable albumentations phone-home version checks in training workers.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


EXPECTED_ACTION_KEYS = [
    "action.wrists.left.xyz",
    "action.wrists.left.rpy",
    "action.wrists.right.xyz",
    "action.wrists.right.rpy",
    "action.hands.left_thumb.xyz",
    "action.hands.left_thumb.rpy",
    "action.hands.left_index.xyz",
    "action.hands.left_index.rpy",
    "action.hands.left_middle.xyz",
    "action.hands.left_middle.rpy",
    "action.hands.right_thumb.xyz",
    "action.hands.right_thumb.rpy",
    "action.hands.right_index.xyz",
    "action.hands.right_index.rpy",
    "action.hands.right_middle.xyz",
    "action.hands.right_middle.rpy",
]
EXPECTED_VIDEO_KEYS = ["egocentric"]
EXPECTED_ANNOTATION_KEYS = ["human.task_description"]


def _load_modality_meta(dataset_path: Path) -> dict:
    modality_path = dataset_path / "meta" / "modality.json"
    if not modality_path.exists():
        raise FileNotFoundError(f"Missing modality.json: {modality_path}")
    with open(modality_path, "r") as f:
        return json.load(f)


def _build_ee_modality_config(
    dataset_path: Path, meta: dict, action_horizon: int
) -> dict[str, ModalityConfig]:
    state_keys = list(meta.get("state", {}).keys())
    action_keys = list(meta.get("action", {}).keys())
    video_keys = list(meta.get("video", {}).keys())
    annotation_keys = list(meta.get("annotation", {}).keys())

    if not state_keys:
        raise ValueError(f"{dataset_path}: no state keys found in modality.json")

    missing_action = sorted(set(EXPECTED_ACTION_KEYS) - set(action_keys))
    if missing_action:
        raise ValueError(
            f"{dataset_path}: action keys mismatch; missing={missing_action}, actual={action_keys}"
        )
    if set(video_keys) != set(EXPECTED_VIDEO_KEYS):
        raise ValueError(f"{dataset_path}: video keys mismatch; actual={video_keys}")
    if set(annotation_keys) != set(EXPECTED_ANNOTATION_KEYS):
        raise ValueError(f"{dataset_path}: annotation keys mismatch; actual={annotation_keys}")

    return {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=video_keys,
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=state_keys,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(action_horizon)),
            modality_keys=EXPECTED_ACTION_KEYS,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
                for _ in EXPECTED_ACTION_KEYS
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=[f"annotation.{k}" for k in annotation_keys],
        ),
    }


def _sum_modality_dim(meta: dict, section: str, keys: list[str]) -> int:
    total = 0
    for key in keys:
        if key not in meta.get(section, {}):
            raise KeyError(f"Missing {section} key '{key}' in modality.json")
        entry = meta[section][key]
        start = entry.get("start")
        end = entry.get("end")
        if start is None or end is None:
            raise KeyError(f"{section}.{key} must define start/end in modality.json")
        total += int(end) - int(start)
    return total


def _infer_state_action_dims(meta: dict, modality_cfg: dict[str, ModalityConfig]) -> tuple[int, int]:
    state_dim = _sum_modality_dim(
        meta,
        "state",
        modality_cfg["state"].modality_keys,
    )
    action_dim = _sum_modality_dim(
        meta,
        "action",
        modality_cfg["action"].modality_keys,
    )
    return state_dim, action_dim


def _adopt_model_config_from_base(cfg, base_model_path: str) -> None:
    """Copy overlapping model config fields from a base GR00T checkpoint config."""
    base_cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    base_dict = base_cfg.to_dict() if hasattr(base_cfg, "to_dict") else vars(base_cfg)
    for key, value in base_dict.items():
        if hasattr(cfg.model, key):
            setattr(cfg.model, key, value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mixed-embodiment HE pretraining (G1+H1) using GR00T launch_train path."
    )
    parser.add_argument("--g1-dataset-path", default="/hfm/data/he_g1_pretrain")
    parser.add_argument("--h1-dataset-path", default="/hfm/data/he_h1_pretrain")
    parser.add_argument("--base-model-path", default="nvidia/GR00T-N1.6-3B")
    parser.add_argument(
        "--adopt-base-model-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Load the base-model config and copy overlapping model fields into the run config "
            "(no checkpoint weights are loaded unless --scratch-gr00t is false)."
        ),
    )
    parser.add_argument(
        "--scratch-gr00t",
        action="store_true",
        help=(
            "Initialize GR00T model weights from config (no GR00T checkpoint load) "
            "while still using pretrained Eagle backbone weights from cfg.model.model_name."
        ),
    )
    parser.add_argument("--output-dir", default="./checkpoints/pretrain_he_g1_h1_mixed")
    parser.add_argument("--g1-mix-ratio", type=float, default=1.0)
    parser.add_argument("--h1-mix-ratio", type=float, default=1.0)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--global-batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-steps", type=int, default=10000)
    parser.add_argument("--save-total-limit", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--max-state-dim", type=int, default=128)
    parser.add_argument("--max-action-dim", type=int, default=128)
    parser.add_argument("--tune-top-llm-layers", type=int, default=4)
    parser.add_argument("--reinit-action-head", action="store_true")
    parser.add_argument("--override-pretraining-statistics", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    args = parser.parse_args()

    if args.action_horizon <= 0:
        raise ValueError(f"action-horizon must be > 0, got {args.action_horizon}")
    if args.tune_top_llm_layers < 0:
        raise ValueError(f"tune-top-llm-layers must be >= 0, got {args.tune_top_llm_layers}")

    g1_dataset = Path(args.g1_dataset_path).resolve()
    h1_dataset = Path(args.h1_dataset_path).resolve()

    cfg = get_default_config()
    cfg.load_config_path = None
    if args.adopt_base_model_config:
        _adopt_model_config_from_base(cfg, args.base_model_path)

    cfg.data.download_cache = False
    cfg.data.datasets = [
        SingleDatasetConfig(
            dataset_paths=[str(g1_dataset)],
            mix_ratio=args.g1_mix_ratio,
            embodiment_tag=EmbodimentTag.G1_EE_A16.value,
        ),
        SingleDatasetConfig(
            dataset_paths=[str(h1_dataset)],
            mix_ratio=args.h1_mix_ratio,
            embodiment_tag=EmbodimentTag.H1_EE_A16.value,
        ),
    ]
    g1_meta = _load_modality_meta(g1_dataset)
    h1_meta = _load_modality_meta(h1_dataset)

    g1_modality_cfg = _build_ee_modality_config(g1_dataset, g1_meta, args.action_horizon)
    h1_modality_cfg = _build_ee_modality_config(h1_dataset, h1_meta, args.action_horizon)
    cfg.data.modality_configs[EmbodimentTag.G1_EE_A16.value] = g1_modality_cfg
    cfg.data.modality_configs[EmbodimentTag.H1_EE_A16.value] = h1_modality_cfg

    g1_state_dim, g1_action_dim = _infer_state_action_dims(g1_meta, g1_modality_cfg)
    h1_state_dim, h1_action_dim = _infer_state_action_dims(h1_meta, h1_modality_cfg)

    cfg.model.max_state_dim = args.max_state_dim
    cfg.model.max_action_dim = args.max_action_dim
    print(
        f"[mixed-he] state_dim g1={g1_state_dim}, h1={h1_state_dim}, "
        f"configured max_state_dim={cfg.model.max_state_dim}; "
        f"action_dim g1={g1_action_dim}, h1={h1_action_dim}, "
        f"configured max_action_dim={cfg.model.max_action_dim}; "
        f"action_horizon={args.action_horizon}"
    )
    cfg.data.override_pretraining_statistics = args.override_pretraining_statistics

    cfg.model.tune_llm = False
    cfg.model.tune_top_llm_layers = args.tune_top_llm_layers
    cfg.model.tune_visual = False
    cfg.model.tune_projector = True
    cfg.model.tune_diffusion_model = True
    cfg.model.state_dropout_prob = 0.0
    cfg.model.random_rotation_angle = None
    cfg.model.color_jitter_params = {
        "brightness": 0.3,
        "contrast": 0.4,
        "saturation": 0.5,
        "hue": 0.08,
    }
    # Eagle-Block2A-2B-v2 asserts on scratch construction unless BF16 loading is enabled.
    cfg.model.load_bf16 = True if args.scratch_gr00t else False
    cfg.model.reproject_vision = False
    cfg.model.eagle_collator = True
    cfg.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    cfg.model.backbone_trainable_params_fp32 = True
    cfg.model.use_relative_action = True
    cfg.model.action_horizon = args.action_horizon

    cfg.training.start_from_checkpoint = None if args.scratch_gr00t else args.base_model_path
    cfg.training.reinit_action_head = args.reinit_action_head
    cfg.training.optim = "adamw_torch"
    cfg.training.output_dir = args.output_dir
    cfg.training.num_gpus = args.num_gpus
    cfg.training.max_steps = args.max_steps
    cfg.training.global_batch_size = args.global_batch_size
    cfg.training.learning_rate = args.learning_rate
    cfg.training.save_steps = args.save_steps
    cfg.training.save_total_limit = args.save_total_limit
    cfg.training.warmup_ratio = args.warmup_ratio
    cfg.training.weight_decay = args.weight_decay
    cfg.training.dataloader_num_workers = args.dataloader_num_workers
    cfg.training.use_wandb = args.use_wandb
    cfg.training.wandb_project = "pretrain-gr00t-he-mixed"

    run(cfg)


if __name__ == "__main__":
    main()
