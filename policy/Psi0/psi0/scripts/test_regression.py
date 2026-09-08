#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import selectors
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import zipfile

from rich.console import Console
from rich.table import Table


DEFAULT_TMPDIR = "/hfm/zhenyu/tmp"
DEFAULT_TORCH_EXTENSIONS_DIR = Path("/hfm/zhenyu/cache/torch_extensions/test_regression")
DEFAULT_SHORT_RUNTIME_DIR = Path("/hfm/zhenyu/tmp/tqs")
DEFAULT_DATASET_CACHE_DIR = Path("/hfm/zhenyu/cache/test_regression_data")
DEFAULT_DATASET_ZIP = "simple/G1WholebodyBendPick-v0.zip"
DEFAULT_DATASET_DIRNAME = "G1WholebodyBendPick-v0-psi0"
DEFAULT_EGOVLA_ALIAS = (
    "mix4data-30hz-transv2update2-fingertip-20e-hdof5-3d200-rot5-lr1e-4-h5p30f1skip6-b16-4"
)
DEFAULT_EGOVLA_ZIP = "vila-qwen2-vl-1.5b-instruct-sft-20240830191953.zip"
DEFAULT_EGOVLA_EXTRACTED = "vila-qwen2-vl-1.5b-instruct-sft-20240830191953"


@dataclass
class TestResult:
    name: str
    status: str
    detail: str = ""


@dataclass
class RuntimeRoots:
    source_root: Path
    tmp_parent: Path
    worktree_root: Path
    simple_worktree_root: Path
    flake_root: Path
    test_root: Path
    cache_root: Path
    log_root: Path
    short_root: Path
    tmp_root: Path
    torch_extensions_root: Path
    triton_root: Path
    xdg_root: Path
    wandb_root: Path


@dataclass
class Interpreters:
    psi: Path
    gr00t: Path
    hrdt: Path
    egovla: Path


@dataclass
class ResourcePaths:
    hrdt_data_dir: Path
    hrdt_vision_encoder_dir: Path
    hrdt_backbone_path: Path
    egovla_data_dir: Path
    egovla_base_model_path: Path
    gr00t_data_dir: str
    gr00t_pretrained_model_path: Path


class Runner:
    def __init__(self, worktree_root: Path, log_root: Path, flake_root: Path):
        self.worktree_root = worktree_root
        self.log_root = log_root
        self.flake_root = flake_root
        self.results: list[TestResult] = []
        self.console = Console()
        self._log_index = 0

    def record(self, name: str, status: str, detail: str = "") -> None:
        self.results.append(TestResult(name=name, status=status, detail=detail))
        style = {
            "PASS": "bold green",
            "FAIL": "bold red",
            "SKIP": "bold yellow",
        }.get(status, "bold")
        text = f"[{style}]{status:>4}[/{style}] {name}"
        if detail:
            text += f" [dim]- {detail}[/dim]"
        self.console.print(text)

    def log_path(self, name: str, suffix: str = ".log") -> Path:
        self._log_index += 1
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "test"
        return self.log_root / f"{self._log_index:02d}_{slug}{suffix}"

    @staticmethod
    def _prepare_log_path(log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w"):
            pass

    @staticmethod
    def _append_output_chunk(
        *,
        chunk: str,
        output_parts: list[str],
        log_file,
        status,
        status_prefix: str,
    ) -> None:
        if not chunk:
            return
        output_parts.append(chunk)
        log_file.write(chunk)
        log_file.flush()
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if not lines:
            return
        status.update(f"[bold blue]{status_prefix}[/bold blue] [dim]- {trim_line(lines[-1])}[/dim]")

    def run_capture(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None,
        status_prefix: str,
        log_path: Path,
    ) -> tuple[int, str]:
        output_parts: list[str] = []

        self._prepare_log_path(log_path)
        with open(log_path, "w", buffering=1) as log_file:
            proc = subprocess.Popen(
                nix_cmd(cmd, flake_dir=self.flake_root),
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None

            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ)
            with self.console.status(f"[bold blue]{status_prefix}[/bold blue]", spinner="dots") as status:
                while True:
                    if proc.poll() is not None:
                        self._append_output_chunk(
                            chunk=proc.stdout.read(),
                            output_parts=output_parts,
                            log_file=log_file,
                            status=status,
                            status_prefix=status_prefix,
                        )
                        break

                    events = selector.select(timeout=0.2)
                    for key, _ in events:
                        line = key.fileobj.readline()
                        self._append_output_chunk(
                            chunk=line,
                            output_parts=output_parts,
                            log_file=log_file,
                            status=status,
                            status_prefix=status_prefix,
                        )

                proc.wait()

            selector.unregister(proc.stdout)
            proc.stdout.close()
        return proc.returncode, "".join(output_parts)

    def run(
        self,
        name: str,
        cmd: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        skip_if: bool = False,
        skip_reason: str = "",
    ) -> bool:
        if skip_if:
            self.record(name, "SKIP", skip_reason)
            return False

        log_path = self.log_path(name)
        returncode, output = self.run_capture(
            cmd,
            cwd=cwd or self.worktree_root,
            env=env,
            status_prefix=f"RUN  {name}",
            log_path=log_path,
        )
        if returncode != 0:
            self.record(name, "FAIL", f"{tail_detail(output)} [log: {log_path}]")
            return False
        self.record(name, "PASS")
        return True

    def summary(self) -> int:
        counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1

        table = Table(title="Regression Test Summary")
        table.add_column("Status", style="bold")
        table.add_column("Count", justify="right")
        table.add_row("PASS", str(counts["PASS"]), style="green")
        table.add_row("FAIL", str(counts["FAIL"]), style="red")
        table.add_row("SKIP", str(counts["SKIP"]), style="yellow")
        self.console.print()
        self.console.print(table)
        return 1 if counts["FAIL"] else 0


def trim_line(text: str, limit: int = 140) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def tail_detail(output: str) -> str:
    lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    if not lines:
        return "exit!=0"
    return " | ".join(lines[-8:])


def run_git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def sh_python(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def nix_cmd(cmd: list[str], *, flake_dir: Path | None = None) -> list[str]:
    flake_target = f"path:{flake_dir}" if flake_dir is not None else "."
    return [
        "nix",
        "develop",
        flake_target,
        "--impure",
        "-c",
        "bash",
        "-lc",
        'if [ -n "${TEST_REGRESSION_VENV_BIN:-}" ]; then export PATH="$TEST_REGRESSION_VENV_BIN:$PATH"; fi; exec '
        + shlex.join(cmd),
    ]


def build_env(extra_pythonpath: list[Path] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if extra_pythonpath:
        env["PYTHONPATH"] = ":".join(str(path) for path in extra_pythonpath)
    return env


def with_venv(env: dict[str, str], python_path: Path) -> dict[str, str]:
    venv_root = python_path.parent.parent
    updated = env.copy()
    updated["VIRTUAL_ENV"] = str(venv_root)
    updated["TEST_REGRESSION_VENV_BIN"] = str(python_path.parent)
    updated["PATH"] = f"{python_path.parent}:{env.get('PATH', '')}"
    return updated


def build_simple_eval_env(base_env: dict[str, str], roots: RuntimeRoots) -> dict[str, str]:
    env = base_env.copy()
    env.update(
        {
            "TORCH_EXTENSIONS_DIR": str(roots.torch_extensions_root),
            "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0+PTX"),
            "ACCEPT_EULA": "Y",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "SIMPLE_DISABLE_TUI": "1",
        }
    )
    return env


def link_shared_path(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.is_dir() and not any(dst.iterdir()):
        dst.rmdir()
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)


def prepare_flake_root(source_root: Path, roots: RuntimeRoots) -> None:
    if roots.flake_root == roots.worktree_root:
        return
    shutil.rmtree(roots.flake_root, ignore_errors=True)
    roots.flake_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / "flake.nix", roots.flake_root / "flake.nix")
    shutil.copy2(source_root / "flake.lock", roots.flake_root / "flake.lock")
    shutil.copytree(source_root / "nix", roots.flake_root / "nix", symlinks=True)
    simple_dst = roots.flake_root / "third_party/SIMPLE"
    simple_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(roots.simple_worktree_root / "flake.nix", simple_dst / "flake.nix")
    shutil.copy2(roots.simple_worktree_root / "flake.lock", simple_dst / "flake.lock")
    shutil.copytree(roots.simple_worktree_root / "nix", simple_dst / "nix", symlinks=True)


def ensure_runtime_roots(worktree_root: Path, tmp_parent: Path) -> RuntimeRoots:
    test_root = worktree_root / ".test_regression"
    cache_root = test_root / "cache"
    roots = RuntimeRoots(
        source_root=worktree_root,
        tmp_parent=tmp_parent,
        worktree_root=worktree_root,
        simple_worktree_root=worktree_root / "third_party/SIMPLE",
        flake_root=test_root / "flake",
        test_root=test_root,
        cache_root=cache_root,
        log_root=test_root / "logs",
        short_root=Path(os.environ.get("TEST_REGRESSION_SHORT_DIR", str(DEFAULT_SHORT_RUNTIME_DIR))),
        tmp_root=Path(os.environ.get("TEST_REGRESSION_TMP_RUNTIME_DIR", str(DEFAULT_SHORT_RUNTIME_DIR / "tmp"))),
        torch_extensions_root=Path(
            os.environ.get("TEST_REGRESSION_TORCH_EXTENSIONS_DIR")
            or os.environ.get("TORCH_EXTENSIONS_DIR")
            or str(DEFAULT_TORCH_EXTENSIONS_DIR)
        ),
        triton_root=cache_root / "triton",
        xdg_root=cache_root / "xdg",
        wandb_root=cache_root / "wandb",
    )
    for path in [
        roots.test_root,
        roots.cache_root,
        roots.log_root,
        roots.flake_root,
        roots.short_root,
        roots.tmp_root,
        roots.torch_extensions_root,
        roots.triton_root,
        roots.xdg_root,
        roots.wandb_root,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return roots


def pick_free_port(start: int = 18080, end: int = 18999) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port found")


def hf_snapshot_download(*, repo_id: str, local_dir: Path, allow_patterns: list[str], repo_type: str = "model") -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        token=os.getenv("HF_TOKEN"),
    )


def ensure_regression_dataset() -> Path:
    pinned = os.environ.get("TEST_REGRESSION_DATA_DIR")
    if pinned:
        return Path(pinned)

    cache_dir = DEFAULT_DATASET_CACHE_DIR
    dataset_dir = cache_dir / DEFAULT_DATASET_DIRNAME
    if dataset_dir.is_dir():
        return dataset_dir

    zip_path = cache_dir / DEFAULT_DATASET_ZIP
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    hf_snapshot_download(
        repo_id="USC-PSI-Lab/psi-data",
        local_dir=cache_dir,
        allow_patterns=[DEFAULT_DATASET_ZIP],
        repo_type="dataset",
    )
    with zipfile.ZipFile(zip_path) as zip_ref:
        zip_ref.extractall(cache_dir)
    return dataset_dir


def ensure_hrdt_resources(source_root: Path) -> tuple[Path, Path, Path]:
    data_dir = Path(os.environ.get("HRDT_DATA_DIR", str(ensure_regression_dataset())))
    vision_dir = Path(os.environ.get("HRDT_VISION_ENCODER_DIR", str(source_root / "src/h_rdt/bak/dino-siglip")))
    backbone_path = Path(
        os.environ.get(
            "HRDT_BACKBONE_PATH",
            str(source_root / "src/h_rdt/checkpoints/pretrain-0618/checkpoint-500000/pytorch_model.bin"),
        )
    )

    paths_ready = vision_dir.is_dir() and backbone_path.is_file()
    env_pinned = "HRDT_VISION_ENCODER_DIR" in os.environ and "HRDT_BACKBONE_PATH" in os.environ
    if paths_ready or env_pinned:
        return data_dir, vision_dir, backbone_path

    hf_snapshot_download(
        repo_id="embodiedfoundation/H-RDT",
        local_dir=source_root / "src/h_rdt",
        allow_patterns=[
            "bak/dino-siglip/**",
            "checkpoints/pretrain-0618/**",
        ],
    )

    return data_dir, vision_dir, backbone_path


def ensure_egovla_base_model(source_root: Path) -> tuple[Path, Path]:
    data_dir = Path(os.environ.get("EGOVLA_DATA_DIR", str(ensure_regression_dataset())))
    base_model_path = Path(
        os.environ.get(
            "EGOVLA_BASE_MODEL_PATH",
            str(source_root / f"src/egovla/checkpoints/{DEFAULT_EGOVLA_ALIAS}"),
        )
    )

    if "EGOVLA_BASE_MODEL_PATH" in os.environ or base_model_path.exists():
        return data_dir, base_model_path

    checkpoints_dir = source_root / "src/egovla/checkpoints"
    zip_path = checkpoints_dir / DEFAULT_EGOVLA_ZIP
    extracted_dir = checkpoints_dir / DEFAULT_EGOVLA_EXTRACTED
    alias_path = checkpoints_dir / DEFAULT_EGOVLA_ALIAS

    hf_snapshot_download(
        repo_id="rchal97/egovla_base_vlm",
        local_dir=checkpoints_dir,
        allow_patterns=[DEFAULT_EGOVLA_ZIP],
        repo_type="model",
    )
    if zip_path.is_file() and not extracted_dir.is_dir():
        with zipfile.ZipFile(zip_path) as zip_ref:
            zip_ref.extractall(checkpoints_dir)
    if not alias_path.exists():
        alias_path.symlink_to(DEFAULT_EGOVLA_EXTRACTED)
    base_model_path = alias_path

    return data_dir, base_model_path


def resolve_resources(source_root: Path, worktree_root: Path) -> ResourcePaths:
    hrdt_data_dir, hrdt_vision_encoder_dir, hrdt_backbone_path = ensure_hrdt_resources(source_root)
    egovla_data_dir, egovla_base_model_path = ensure_egovla_base_model(source_root)
    return ResourcePaths(
        hrdt_data_dir=hrdt_data_dir,
        hrdt_vision_encoder_dir=hrdt_vision_encoder_dir,
        hrdt_backbone_path=hrdt_backbone_path,
        egovla_data_dir=egovla_data_dir,
        egovla_base_model_path=egovla_base_model_path,
        gr00t_data_dir=os.environ.get("GR00T_DATA_DIR", str(ensure_regression_dataset())),
        gr00t_pretrained_model_path=Path(
            os.environ.get(
                "GR00T_PRETRAINED_MODEL_PATH",
                str(worktree_root / "checkpoints/pretrain_he_g1_h1_mixed_scratch_gr00t/checkpoint-50000"),
            )
        ),
    )


def make_interpreters(worktree_root: Path) -> Interpreters:
    psi = worktree_root / ".venv-psi/bin/python"
    return Interpreters(
        psi=psi,
        gr00t=psi,
        hrdt=worktree_root / "src/h_rdt/.venv/bin/python",
        egovla=worktree_root / "src/egovla/.venv/bin/python",
    )


def make_envs(worktree_root: Path) -> dict[str, dict[str, str]]:
    return {
        "psi": build_env([worktree_root / "src", worktree_root / "third_party/SIMPLE/src"]),
        "gr00t": build_env(
            [worktree_root / "src", worktree_root / "src/gr00t", worktree_root / "third_party/SIMPLE/src"]
        ),
        "hrdt": build_env([worktree_root / "src/h_rdt"]),
        "egovla": build_env([worktree_root / "src/egovla/VILA"]),
    }


def run_quick_checks(
    runner: Runner,
    worktree_root: Path,
    py: Interpreters,
    envs: dict[str, dict[str, str]],
    resources: ResourcePaths,
) -> None:
    runner.run(
        "psi/simple import",
        [str(py.psi), "-c", "import psi, simple; print('ok')"],
        env=envs["psi"],
        skip_if=not sh_python(py.psi),
        skip_reason=f"missing {py.psi}",
    )
    runner.run(
        "simple.cli.datagen source compiles",
        [sys.executable, "-m", "py_compile", str(worktree_root / "third_party/SIMPLE/src/simple/cli/datagen.py")],
        skip_if=not sh_python(py.psi),
        skip_reason=f"missing {py.psi}",
    )
    runner.run(
        "simple.cli.eval source compiles",
        [sys.executable, "-m", "py_compile", str(worktree_root / "third_party/SIMPLE/src/simple/cli/eval.py")],
        skip_if=not sh_python(py.psi),
        skip_reason=f"missing {py.psi}",
    )
    runner.run(
        "gr00t train dry-run",
        [
            str(py.gr00t),
            "baselines/gr00t-n1.6/finetune_gr00t.py",
            "--preset",
            "finetune_simple",
            "--dataset-path",
            resources.gr00t_data_dir,
            "--base-model-path",
            str(resources.gr00t_pretrained_model_path),
            "--output-dir",
            str(worktree_root / ".test_regression/gr00t"),
            "--dry-run",
        ],
        env=envs["gr00t"],
        skip_if=not sh_python(py.gr00t),
        skip_reason=f"missing {py.gr00t}",
    )
    runner.run(
        "gr00t eval dry-run",
        [
            str(py.gr00t),
            "baselines/gr00t-n1.6/eval_simple.py",
            "--preset",
            "simple_local",
            "--data-dir",
            resources.gr00t_data_dir,
            "--dry-run",
        ],
        env=envs["gr00t"],
        skip_if=not sh_python(py.gr00t),
        skip_reason=f"missing {py.gr00t}",
    )
    runner.run(
        "hrdt main --help",
        [str(py.hrdt), "main.py", "--help"],
        cwd=worktree_root / "src/h_rdt",
        env=envs["hrdt"],
        skip_if=not sh_python(py.hrdt),
        skip_reason=f"missing {py.hrdt}",
    )
    runner.run(
        "hrdt test_serve --help",
        [str(py.hrdt), "tools/hrdt_test_serve.py", "--help"],
        cwd=worktree_root / "src/h_rdt",
        env=envs["hrdt"],
        skip_if=not sh_python(py.hrdt),
        skip_reason=f"missing {py.hrdt}",
    )
    runner.run(
        "egovla test_serve --help",
        [str(py.egovla), "tools/lerobot_test_serve.py", "--help"],
        cwd=worktree_root / "src/egovla",
        env=envs["egovla"],
        skip_if=not sh_python(py.egovla),
        skip_reason=f"missing {py.egovla}",
    )


def run_server_eval(
    runner: Runner,
    name: str,
    server_cmd: list[str],
    eval_cmd: list[str],
    *,
    server_cwd: Path,
    eval_cwd: Path,
    server_env: dict[str, str],
    eval_env: dict[str, str],
    port: int,
    wait_tries: int = 900,
    wait_sleep_s: float = 0.2,
    skip_if: bool = False,
    skip_reason: str = "",
) -> None:
    if skip_if:
        runner.record(name, "SKIP", skip_reason)
        return

    server_log_path = runner.log_path(f"{name}_server")
    eval_log_path = runner.log_path(name)
    runner._prepare_log_path(server_log_path)
    runner._prepare_log_path(eval_log_path)
    with open(server_log_path, "w", buffering=1) as server_log:
        server = subprocess.Popen(
            nix_cmd(server_cmd, flake_dir=runner.flake_root),
            cwd=server_cwd,
            env=server_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert server.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(server.stdout, selectors.EVENT_READ)
        try:
            with runner.console.status(f"[bold blue]RUN  {name}[/bold blue]", spinner="dots") as status:
                waited = 0.0
                ready = False
                while waited < wait_tries * wait_sleep_s:
                    events = selector.select(timeout=wait_sleep_s)
                    for key, _ in events:
                        line = key.fileobj.readline()
                        if not line:
                            continue
                        server_log.write(line)
                        stripped = line.strip()
                        if stripped:
                            status.update(f"[bold blue]RUN  {name}[/bold blue] [dim]- {trim_line(stripped)}[/dim]")
                    if server.poll() is not None:
                        break
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                        if sock.connect_ex(("127.0.0.1", port)) == 0:
                            ready = True
                            break
                    waited += wait_sleep_s

            if ready:
                returncode, output = runner.run_capture(
                    eval_cmd,
                    cwd=eval_cwd,
                    env=eval_env,
                    status_prefix=f"RUN  {name}",
                    log_path=eval_log_path,
                )
                if returncode != 0:
                    runner.record(name, "FAIL", f"{tail_detail(output)} [log: {eval_log_path}]")
                    return
                runner.record(name, "PASS")
                return

            server_output = server_log_path.read_text() if server_log_path.exists() else ""
            runner.record(name, "FAIL", f"{tail_detail(server_output)} [server log: {server_log_path}]")
        except Exception as exc:
            runner.record(name, "FAIL", f"{exc} [server log: {server_log_path}]")
        finally:
            selector.unregister(server.stdout)
            server.stdout.close()
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(server.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                server.wait(timeout=5)


def run_live_checks(
    runner: Runner,
    roots: RuntimeRoots,
    py: Interpreters,
    envs: dict[str, dict[str, str]],
    resources: ResourcePaths,
) -> None:
    hrdt_env = with_venv(os.environ.copy(), py.hrdt)
    hrdt_env.update(
        {
            "PYTHONPATH": str(roots.worktree_root / "src/h_rdt"),
            "TMPDIR": str(roots.tmp_root),
            "TMP": str(roots.tmp_root),
            "TEMP": str(roots.tmp_root),
            "XDG_CACHE_HOME": str(roots.xdg_root),
            "TRITON_CACHE_DIR": str(roots.triton_root / "hrdt"),
            "WANDB_DIR": str(roots.wandb_root / "hrdt"),
            "DINO_SIGLIP_DIR": str(resources.hrdt_vision_encoder_dir),
            "PRETRAINED_BACKBONE_PATH": str(resources.hrdt_backbone_path),
            "LEROBOT_VIDEO_BACKEND": "pyav",
            "WANDB_MODE": "offline",
            "USE_PRECOMP_LANG_EMBED": "0",
            "TRAIN_BATCH_SIZE": "1",
            "SAMPLE_BATCH_SIZE": "1",
            "MAX_TRAIN_STEPS": "1",
            "CHECKPOINTING_PERIOD": "1",
            "SAMPLE_PERIOD": "-1",
            "DATALOADER_NUM_WORKERS": "0",
        }
    )
    runner.run(
        "hrdt 1-step finetune",
        ["bash", "finetune_lerobot.sh", str(resources.hrdt_data_dir), str(roots.test_root / "hrdt")],
        cwd=roots.worktree_root / "src/h_rdt",
        env=hrdt_env,
        skip_if=not (
            sh_python(py.hrdt)
            and resources.hrdt_data_dir.is_dir()
            and resources.hrdt_vision_encoder_dir.is_dir()
            and resources.hrdt_backbone_path.is_file()
        ),
        skip_reason="H-RDT live prerequisites not available",
    )
    hrdt_model = Path(os.environ.get("HRDT_MODEL_PATH", str(roots.test_root / "hrdt/checkpoint-1/model.safetensors")))
    hrdt_port = pick_free_port()
    run_server_eval(
        runner,
        "hrdt simple eval",
        ["bash", "deploy.sh", str(hrdt_model), "127.0.0.1", str(hrdt_port)],
        [
            str(py.psi),
            "-m",
            "simple.cli.eval",
            "simple/G1WholebodyBendPick-v0",
            "hrdt",
            "train",
            "--host",
            "127.0.0.1",
            "--port",
            str(hrdt_port),
            "--data-format",
            "lerobot",
            "--sim-mode",
            "mujoco_isaac",
            "--headless",
            "--eval-dir",
            str(roots.test_root / "evals/hrdt"),
            "--max-episode-steps",
            "1",
            "--num-episodes",
            "1",
            "--data-dir",
            str(resources.hrdt_data_dir),
            "--success-criteria",
            "0.9",
        ],
        server_cwd=roots.worktree_root / "src/h_rdt",
        eval_cwd=roots.worktree_root,
        server_env=hrdt_env,
        eval_env=build_simple_eval_env(envs["psi"], roots),
        port=hrdt_port,
        skip_if=not hrdt_model.is_file(),
        skip_reason="H-RDT deploy checkpoint not available",
    )

    egovla_train_port = pick_free_port()
    egovla_env = with_venv(os.environ.copy(), py.egovla)
    egovla_env.update(
        {
            "PYTHONPATH": str(roots.worktree_root / "src/egovla/VILA"),
            "TMPDIR": str(roots.short_root / "tmp"),
            "TMP": str(roots.short_root / "tmp"),
            "TEMP": str(roots.short_root / "tmp"),
            "XDG_CACHE_HOME": str(roots.xdg_root),
            "TRITON_CACHE_DIR": str(roots.triton_root / "egovla"),
            "WANDB_DIR": str(roots.wandb_root / "egovla"),
            "DATA_ROOT": str(resources.egovla_data_dir.parent),
            "LEROBOT_TASK_DIR": resources.egovla_data_dir.name,
            "LEROBOT_VIDEO_BACKEND": "pyav",
            "WANDB_MODE": "offline",
            "NPROC_PER_NODE": "1",
            "MASTER_PORT": str(egovla_train_port),
            "PER_DEVICE_BS": "1",
            "GRAD_ACCUM_STEPS": "1",
            "MAX_STEPS": "1",
            "RUN_NAME": "test_regression",
            "OUTPUT_DIR": str(roots.short_root / "e"),
        }
    )
    runner.run(
        "egovla 1-step finetune",
        ["bash", "finetune.sh"],
        cwd=roots.worktree_root / "src/egovla",
        env=egovla_env,
        skip_if=not (sh_python(py.egovla) and resources.egovla_data_dir.is_dir() and resources.egovla_base_model_path.exists()),
        skip_reason="EgoVLA live prerequisites not available",
    )
    egovla_model = Path(os.environ.get("EGOVLA_MODEL_PATH", str(roots.short_root / "e/checkpoint-1")))
    egovla_port = pick_free_port()
    run_server_eval(
        runner,
        "egovla simple eval",
        ["bash", "deploy.sh", str(egovla_model), "127.0.0.1", str(egovla_port)],
        [
            str(py.psi),
            "-m",
            "simple.cli.eval",
            "simple/G1WholebodyBendPick-v0",
            "egovla",
            "train",
            "--host",
            "127.0.0.1",
            "--port",
            str(egovla_port),
            "--data-format",
            "lerobot",
            "--sim-mode",
            "mujoco_isaac",
            "--headless",
            "--eval-dir",
            str(roots.test_root / "evals/egovla"),
            "--max-episode-steps",
            "1",
            "--num-episodes",
            "1",
            "--data-dir",
            str(resources.egovla_data_dir),
            "--success-criteria",
            "0.9",
        ],
        server_cwd=roots.worktree_root / "src/egovla",
        eval_cwd=roots.worktree_root,
        server_env=egovla_env,
        eval_env=build_simple_eval_env(envs["psi"], roots),
        port=egovla_port,
        skip_if=not egovla_model.exists(),
        skip_reason="EgoVLA deploy checkpoint not available",
    )

    gr00t_model = Path(
        os.environ.get(
            "GR00T_MODEL_PATH",
            str(roots.worktree_root / "checkpoints/pretrained_mixed_scratch_downstream/checkpoint-50000"),
        )
    )
    gr00t_port = pick_free_port()
    run_server_eval(
        runner,
        "gr00t simple eval",
        [
            str(py.gr00t),
            "-m",
            "gr00t.deploy.gr00t_serve_simple",
            "--embodiment-tag",
            "G1_LOCO_DOWNSTREAM",
            "--model-path",
            str(gr00t_model),
            "--device",
            "cuda:0",
            "--host",
            "0.0.0.0",
            "--port",
            str(gr00t_port),
            "--use-sim-policy-wrapper",
            "--strict",
        ],
        [
            str(py.psi),
            "-m",
            "simple.cli.eval",
            "simple/G1WholebodyBendPick-v0",
            "gr00t_n16",
            "train",
            "--host",
            "127.0.0.1",
            "--port",
            str(gr00t_port),
            "--data-format",
            "lerobot",
            "--sim-mode",
            "mujoco_isaac",
            "--headless",
            "--eval-dir",
            str(roots.test_root / "evals/gr00t"),
            "--max-episode-steps",
            "1",
            "--num-episodes",
            "1",
            "--data-dir",
            resources.gr00t_data_dir,
            "--success-criteria",
            "0.9",
        ],
        server_cwd=roots.worktree_root,
        eval_cwd=roots.worktree_root,
        server_env=envs["gr00t"],
        eval_env=build_simple_eval_env(envs["psi"], roots),
        port=gr00t_port,
        skip_if=not gr00t_model.exists(),
        skip_reason="GR00T deploy checkpoint not available",
    )


def link_shared_inputs(source_root: Path, worktree_root: Path) -> None:
    if source_root == worktree_root:
        return
    for rel in [
        ".venv-psi",
        "checkpoints",
        "src/gr00t/.venv",
        "src/h_rdt/.venv",
        "src/h_rdt/checkpoints",
        "src/h_rdt/bak",
        "src/egovla/.venv",
        "src/egovla/checkpoints",
    ]:
        link_shared_path(source_root / rel, worktree_root / rel)


def create_clean_worktrees(source_root: Path, tmpdir: Path) -> tuple[Path, Path, Path]:
    tmpdir.mkdir(parents=True, exist_ok=True)
    tmp_parent = Path(tempfile.mkdtemp(prefix="test-regression.", dir=tmpdir))
    worktree_root = tmp_parent / "repo"
    simple_source_root = source_root / "third_party/SIMPLE"
    simple_worktree_root = tmp_parent / "SIMPLE"
    subprocess.run(
        ["git", "-C", str(source_root), "worktree", "add", "--detach", str(worktree_root), "HEAD"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(simple_source_root), "worktree", "add", "--detach", str(simple_worktree_root), "HEAD"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    simple_dst = worktree_root / "third_party/SIMPLE"
    if simple_dst.is_dir() and not any(simple_dst.iterdir()):
        simple_dst.rmdir()
    if not simple_dst.exists():
        simple_dst.parent.mkdir(parents=True, exist_ok=True)
        simple_dst.symlink_to(simple_worktree_root)
    return tmp_parent, worktree_root, simple_worktree_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regression tests in a clean detached worktree by default.")
    parser.add_argument("--mode", choices=["quick", "live"], default="quick")
    parser.add_argument("--current-worktree", action="store_true", help="Use the current checkout instead of a clean detached worktree.")
    parser.add_argument("--keep-worktree", action="store_true")
    parser.add_argument("--tmpdir", default=os.environ.get("TEST_REGRESSION_TMPDIR", DEFAULT_TMPDIR))
    return parser.parse_args()


def cleanup_worktree(source_root: Path, roots: RuntimeRoots, keep_worktree: bool) -> None:
    if keep_worktree:
        print(f"[keep] worktree kept at {roots.worktree_root}")
        return
    if roots.worktree_root == source_root:
        shutil.rmtree(roots.test_root, ignore_errors=True)
        return
    subprocess.run(
        ["git", "-C", str(source_root / "third_party/SIMPLE"), "worktree", "remove", "--force", str(roots.simple_worktree_root)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    subprocess.run(
        ["git", "-C", str(source_root), "worktree", "remove", "--force", str(roots.worktree_root)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    shutil.rmtree(roots.tmp_parent, ignore_errors=True)


def main() -> int:
    args = parse_args()
    source_root = Path(run_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    if args.current_worktree:
        tmp_parent = Path(args.tmpdir)
        worktree_root = source_root
        simple_worktree_root = source_root / "third_party/SIMPLE"
    else:
        tmp_parent, worktree_root, simple_worktree_root = create_clean_worktrees(source_root, Path(args.tmpdir))
    roots = ensure_runtime_roots(worktree_root, tmp_parent)
    roots.source_root = source_root
    roots.simple_worktree_root = simple_worktree_root

    try:
        link_shared_inputs(source_root, worktree_root)
        prepare_flake_root(source_root, roots)
        resources = resolve_resources(source_root, worktree_root)
        py = make_interpreters(worktree_root)
        envs = make_envs(worktree_root)
        runner = Runner(worktree_root, roots.log_root, roots.flake_root)

        run_quick_checks(runner, worktree_root, py, envs, resources)
        if args.mode == "live":
            run_live_checks(runner, roots, py, envs, resources)
        return runner.summary()
    finally:
        cleanup_worktree(source_root, roots, args.keep_worktree)


if __name__ == "__main__":
    raise SystemExit(main())
