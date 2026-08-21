import json
import sys
from types import SimpleNamespace

from XPolicyLab.policy.GauDP.gaudp.experiment_logger import ExperimentLogger, parse_wandb_tags


def test_jsonl_logger_without_wandb(tmp_path):
    with ExperimentLogger(
        tmp_path,
        config={"checkpoint": tmp_path / "initial.ckpt"},
        wandb_mode="disabled",
    ) as logger:
        logger.log({"epoch": 0, "train/loss": 1.25}, step=3)

    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["step"] == 3
    assert records[0]["epoch"] == 0
    assert records[0]["train/loss"] == 1.25
    assert isinstance(records[0]["timestamp"], float)


def test_parse_wandb_tags():
    assert parse_wandb_tags("gaussian, cocarry,seed-0") == ["gaussian", "cocarry", "seed-0"]


def test_default_wandb_mode_is_online(monkeypatch, tmp_path):
    calls = {}

    class FakeRun:
        def finish(self):
            pass

    def init(**kwargs):
        calls["init"] = kwargs
        return FakeRun()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    with ExperimentLogger(tmp_path, config={"lr": 1e-5}):
        pass

    assert calls["init"]["mode"] == "online"


def test_wandb_mirrors_jsonl_record(monkeypatch, tmp_path):
    calls = {}

    class FakeRun:
        def log(self, payload, step):
            calls["log"] = (payload, step)

        def finish(self):
            calls["finished"] = True

    def init(**kwargs):
        calls["init"] = kwargs
        return FakeRun()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    with ExperimentLogger(
        tmp_path,
        config={"lr": 1e-5},
        wandb_mode="offline",
        wandb_project="test-project",
        wandb_run_name="test-run",
        wandb_tags=["gaussian"],
    ) as logger:
        logger.log({"val/psnr": 23.4}, step=7)

    assert calls["init"]["mode"] == "offline"
    assert calls["init"]["project"] == "test-project"
    assert calls["log"][0]["val/psnr"] == 23.4
    assert calls["log"][1] == 7
    assert calls["finished"] is True
