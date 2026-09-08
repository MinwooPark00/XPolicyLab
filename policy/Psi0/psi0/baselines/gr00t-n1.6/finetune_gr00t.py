#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PRESET_ROOT = SCRIPT_DIR / "presets" / "train"
GR00T_PYTHON = REPO_ROOT / "src/gr00t/.venv/bin/python"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_preset(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate.resolve()
    preset_path = PRESET_ROOT / f"{name_or_path}.yaml"
    if preset_path.exists():
        return preset_path.resolve()
    raise FileNotFoundError(f"Preset not found: {name_or_path}")


def _load_preset(path: Path) -> dict[str, Any]:
    preset = _load_yaml(path)
    extends = preset.pop("extends", None)
    if extends is None:
        return preset

    extend_list = extends if isinstance(extends, list) else [extends]
    merged: dict[str, Any] = {}
    for entry in extend_list:
        parent_path = _resolve_preset(str((path.parent / entry).resolve() if not Path(entry).is_absolute() else Path(entry)))
        merged = _deep_merge(merged, _load_preset(parent_path))
    return _deep_merge(merged, preset)


def _nproc_from_visible_devices(value: str) -> int:
    return len([part.strip() for part in value.split(",") if part.strip()])


def _flag_name(key: str) -> str:
    return f"--{key.replace('_', '-')}"


def _append_args(cmd: list[str], args_map: dict[str, Any]) -> None:
    for key, value in args_map.items():
        if value is None:
            continue
        flag = _flag_name(key)
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if isinstance(value, list):
            cmd.append(flag)
            cmd.extend(str(item) for item in value)
            continue
        cmd.extend([flag, str(value)])


def _build_color_jitter_args(config: dict[str, Any]) -> list[str]:
    return [
        item
        for key, value in config.items()
        for item in (str(key), str(value))
    ]


def _build_launch_finetune_args(preset: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = preset.get("dataset", {})
    model_cfg = preset.get("model", {})
    training_cfg = preset.get("training", {})
    augment_cfg = preset.get("augmentation", {})
    args_map: dict[str, Any] = {
        "base_model_path": model_cfg.get("base_model_path"),
        "dataset_path": dataset_cfg.get("path"),
        "embodiment_tag": dataset_cfg.get("embodiment_tag"),
        "modality_config_path": dataset_cfg.get("modality_config_path"),
        "num_gpus": training_cfg.get("num_gpus"),
        "output_dir": training_cfg.get("output_dir"),
        "save_steps": training_cfg.get("save_steps"),
        "save_total_limit": training_cfg.get("save_total_limit"),
        "max_steps": training_cfg.get("max_steps"),
        "warmup_ratio": training_cfg.get("warmup_ratio"),
        "weight_decay": training_cfg.get("weight_decay"),
        "learning_rate": training_cfg.get("learning_rate"),
        "global_batch_size": training_cfg.get("global_batch_size"),
        "dataloader_num_workers": training_cfg.get("dataloader_num_workers"),
    }
    color_jitter = augment_cfg.get("color_jitter")
    if color_jitter:
        args_map["color_jitter_params"] = _build_color_jitter_args(color_jitter)
    args_map.update(preset.get("extra_args", {}))
    return args_map


def _build_mixed_he_args(preset: dict[str, Any]) -> dict[str, Any]:
    datasets_cfg = preset.get("datasets", {})
    g1_cfg = datasets_cfg.get("g1", {})
    h1_cfg = datasets_cfg.get("h1", {})
    model_cfg = preset.get("model", {})
    training_cfg = preset.get("training", {})
    args_map: dict[str, Any] = {
        "g1_dataset_path": g1_cfg.get("path"),
        "h1_dataset_path": h1_cfg.get("path"),
        "g1_mix_ratio": g1_cfg.get("mix_ratio"),
        "h1_mix_ratio": h1_cfg.get("mix_ratio"),
        "base_model_path": model_cfg.get("base_model_path"),
        "adopt_base_model_config": model_cfg.get("adopt_base_model_config"),
        "output_dir": training_cfg.get("output_dir"),
        "num_gpus": training_cfg.get("num_gpus"),
        "max_steps": training_cfg.get("max_steps"),
        "global_batch_size": training_cfg.get("global_batch_size"),
        "learning_rate": training_cfg.get("learning_rate"),
        "save_steps": training_cfg.get("save_steps"),
        "save_total_limit": training_cfg.get("save_total_limit"),
        "warmup_ratio": training_cfg.get("warmup_ratio"),
        "weight_decay": training_cfg.get("weight_decay"),
        "dataloader_num_workers": training_cfg.get("dataloader_num_workers"),
    }
    args_map.update(preset.get("extra_args", {}))
    return args_map


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonical GR00T finetune/pretrain launcher.")
    parser.add_argument("--preset", required=True, help="Preset name or YAML path.")
    parser.add_argument("--dataset-path", help="Override dataset path for single-dataset presets.")
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument("--base-model-path", help="Override base model path.")
    parser.add_argument("--cuda-visible-devices", help="Override CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--master-port", type=int, help="Override torchrun master port.")
    parser.add_argument("--num-gpus", type=int, help="Override launcher --num-gpus.")
    parser.add_argument("--dry-run", action="store_true", help="Print command and exit.")
    args = parser.parse_args()

    preset_path = _resolve_preset(args.preset)
    preset = _load_preset(preset_path)

    launcher = preset.get("launcher", "launch_finetune")
    launcher_script = {
        "launch_finetune": REPO_ROOT / "src/gr00t/gr00t/experiment/launch_finetune.py",
        "mixed_he": REPO_ROOT / "baselines/gr00t-n1.6/launch_train_he_mixed_ee.py",
    }.get(launcher)
    if launcher_script is None:
        raise ValueError(f"Unsupported launcher: {launcher}")

    env = os.environ.copy()
    env.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{REPO_ROOT / 'src/gr00t'}:{env.get('PYTHONPATH', '')}".rstrip(":")

    visible_devices = (
        args.cuda_visible_devices
        or preset.get("runtime", {}).get("cuda_visible_devices")
        or env.get("CUDA_VISIBLE_DEVICES")
        or "0"
    )
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    nproc_per_node = preset.get("runtime", {}).get("nproc_per_node")
    if nproc_per_node is None:
        nproc_per_node = _nproc_from_visible_devices(visible_devices)
    master_port = args.master_port or preset.get("runtime", {}).get("master_port", 29501)

    if launcher == "launch_finetune":
        launcher_args = _build_launch_finetune_args(preset)
        if args.dataset_path:
            launcher_args["dataset_path"] = args.dataset_path
    else:
        launcher_args = _build_mixed_he_args(preset)

    if args.output_dir:
        launcher_args["output_dir"] = args.output_dir
    if args.base_model_path:
        launcher_args["base_model_path"] = args.base_model_path
    if args.num_gpus is not None:
        launcher_args["num_gpus"] = args.num_gpus
    elif launcher_args.get("num_gpus") is None:
        launcher_args["num_gpus"] = nproc_per_node

    for key, value in preset.get("env", {}).items():
        env[str(key)] = str(value)

    cmd = [
        str(GR00T_PYTHON),
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={master_port}",
        str(launcher_script),
    ]
    _append_args(cmd, launcher_args)

    print(f"preset={preset_path}")
    print("command:")
    print(" ".join(cmd))
    if args.dry_run:
        return 0

    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
