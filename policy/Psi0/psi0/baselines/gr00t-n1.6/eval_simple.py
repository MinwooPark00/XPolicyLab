#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any

import yaml

from simple.evals.api import EvalConfig, EvalRunner


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PRESET_ROOT = SCRIPT_DIR / "presets" / "eval"
DEFAULT_GR00T_PYTHON = REPO_ROOT / "src/gr00t/.venv/bin/python"


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
        parent = Path(entry)
        parent_path = (
            parent.resolve()
            if parent.is_absolute()
            else _resolve_preset(str((path.parent / parent).resolve()))
        )
        merged = _deep_merge(merged, _load_preset(parent_path))
    return _deep_merge(merged, preset)


def _pick_free_port(start: int = 5556, end: int = 5999) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port found in range {start}-{end}")


def _wait_for_port(host: str, port: int, tries: int, sleep_s: float) -> None:
    for _ in range(tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(sleep_s)
    raise TimeoutError(f"Timed out waiting for {host}:{port}")


def _build_eval_config(values: dict[str, Any]) -> EvalConfig:
    return EvalConfig(
        env_id=values["env_id"],
        policy=values["policy"],
        split=values.get("split", "train"),
        host=values.get("host", "localhost"),
        port=int(values["port"]),
        data_format=values.get("data_format", "lerobot"),
        sim_mode=values.get("sim_mode", "mujoco_isaac"),
        headless=bool(values.get("headless", True)),
        eval_dir=values.get("eval_dir", "data/evals"),
        max_episode_steps=int(values.get("max_episode_steps", 360)),
        num_episodes=int(values.get("num_episodes", 10)),
        episode_start=int(values.get("episode_start", 0)),
        data_dir=values["data_dir"],
        success_criteria=float(values.get("success_criteria", 0.9)),
        save_video=bool(values.get("save_video", True)),
        num_workers=int(values.get("num_workers", 1)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonical GR00T SIMPLE eval launcher.")
    parser.add_argument("--preset", required=True, help="Preset name or YAML path.")
    parser.add_argument("--model-path", help="Override model path.")
    parser.add_argument("--data-dir", help="Override eval dataset root.")
    parser.add_argument("--num-episodes", type=int, help="Override number of eval episodes.")
    parser.add_argument("--num-workers", type=int, help="Override number of SIMPLE eval workers.")
    parser.add_argument("--port", type=int, help="Override server/eval port.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit.")
    args = parser.parse_args()

    preset_path = _resolve_preset(args.preset)
    preset = _load_preset(preset_path)

    server_cfg = dict(preset.get("server", {}))
    eval_cfg = dict(preset.get("eval", {}))
    runtime_cfg = dict(preset.get("runtime", {}))
    env_cfg = dict(preset.get("env", {}))

    port = args.port or server_cfg.get("port") or eval_cfg.get("port")
    if port is None:
        port = 5556 if args.dry_run else _pick_free_port()
    eval_cfg["port"] = port

    if args.model_path:
        server_cfg["model_path"] = args.model_path
    if args.data_dir:
        eval_cfg["data_dir"] = args.data_dir
    if args.num_episodes is not None:
        eval_cfg["num_episodes"] = args.num_episodes
    if args.num_workers is not None:
        eval_cfg["num_workers"] = args.num_workers

    gr00t_python = Path(server_cfg.pop("python", DEFAULT_GR00T_PYTHON)).resolve()
    server_host = server_cfg.pop("host", "0.0.0.0")
    server_cfg.pop("port", None)
    wait_host = eval_cfg.get("host", "localhost")
    wait_tries = int(runtime_cfg.get("wait_tries", 1200))
    wait_sleep_s = float(runtime_cfg.get("wait_sleep_s", 0.1))

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{REPO_ROOT / 'src/gr00t'}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env.setdefault("TORCHINDUCTOR_DISABLE", "1")
    env.setdefault("TORCH_COMPILE", "0")
    env.setdefault("HF_HOME", "/tmp/hf")
    env.setdefault("TRANSFORMERS_CACHE", f"{env['HF_HOME']}/transformers")
    env.setdefault("XDG_CACHE_HOME", env["HF_HOME"])
    for key, value in env_cfg.items():
        env[str(key)] = str(value)

    server_cmd = [
        str(gr00t_python),
        "-m",
        "gr00t.deploy.gr00t_serve_simple",
        "--host",
        str(server_host),
        "--port",
        str(port),
    ]
    for key, value in server_cfg.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                server_cmd.append(flag)
        elif value is not None:
            server_cmd.extend([flag, str(value)])

    eval_config = _build_eval_config(eval_cfg)
    print(f"preset={preset_path}")
    print(f"server_cmd={' '.join(server_cmd)}")
    print(f"eval_config={eval_config}")
    if args.dry_run:
        return 0

    log_path = Path(runtime_cfg.get("server_log", f"/tmp/gr00t_serve_{port}.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        server = subprocess.Popen(
            server_cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    try:
        _wait_for_port(wait_host, port, wait_tries, wait_sleep_s)
        result = EvalRunner(eval_config).run()
        print(f"success_rate={result.success_rate:.6f}")
        print(f"log_path={result.log_path}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
