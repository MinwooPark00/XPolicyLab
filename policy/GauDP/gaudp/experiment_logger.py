"""Local JSONL and optional Weights & Biases experiment logging."""

from __future__ import annotations

import json
import numbers
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    return str(value)


class ExperimentLogger:
    """Always write JSONL and optionally mirror each record to W&B."""

    def __init__(
        self,
        output_dir: Path,
        *,
        config: Mapping[str, Any],
        wandb_mode: str = "offline",
        wandb_project: str = "MHBench-GauDP",
        wandb_run_name: str | None = None,
        wandb_entity: str | None = None,
        wandb_group: str | None = None,
        wandb_tags: Sequence[str] = (),
    ) -> None:
        if wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError(f"unsupported W&B mode: {wandb_mode}")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "metrics.jsonl"
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._wandb_run = None

        if wandb_mode != "disabled":
            try:
                import wandb
            except ImportError as error:
                self._file.close()
                raise RuntimeError(
                    "W&B logging is enabled but wandb is not installed. Run install.sh, "
                    "or pass --wandb-mode disabled for JSONL-only logging."
                ) from error
            self._wandb_run = wandb.init(
                dir=str(self.output_dir),
                project=wandb_project,
                name=wandb_run_name,
                entity=wandb_entity,
                group=wandb_group,
                tags=list(wandb_tags),
                mode=wandb_mode,
                config=_json_value(config),
            )

    def log(self, metrics: Mapping[str, Any], *, step: int) -> None:
        payload = {
            "timestamp": time.time(),
            "step": int(step),
            **{str(key): _json_value(value) for key, value in metrics.items()},
        }
        self._file.write(json.dumps(payload, sort_keys=True) + "\n")
        if self._wandb_run is not None:
            self._wandb_run.log(dict(payload), step=int(step))

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def parse_wandb_tags(value: str) -> list[str]:
    return [tag.strip() for tag in value.split(",") if tag.strip()]
