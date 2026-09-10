import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "launcher_args.sh"
BASELINES = Path(__file__).resolve().parents[4]
COMMON = BASELINES / "scripts" / "_mhbench_common.sh"
SERVE = BASELINES / "scripts" / "serve" / "GauDP.sh"


def _run(script: str, *args: Path | str) -> list[str]:
    result = subprocess.run(
        ["bash", "-c", script, "_", *(str(arg) for arg in args)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def test_handover_aliases_all_write_the_canonical_run_and_data_names(tmp_path):
    command = r'''
        set -euo pipefail
        POLICY_DIR=$1
        source "$2"
        gaudp_parse_stage_args "$3" 0 0
        printf '%s\n' "$task" "$scene" "$ckpt" "$(gaudp_run_dir)" "$(gaudp_data_path)"
    '''
    expected_run = tmp_path / "checkpoints/mhbench-handover-unitree_g1x2_centralized-joint-0"
    expected_data = tmp_path / "data/mhbench-handover-unitree_g1x2_centralized-joint.hdf5"
    for alias in ("handover", "handover_easy", "handovereasy"):
        values = _run(command, tmp_path, SCRIPT, alias)
        assert values == ["handover", "handover", "handover", str(expected_run), str(expected_data)]


def test_canonical_handover_run_finds_the_retired_gaussian_artifacts(tmp_path):
    gaussian = tmp_path / "checkpoints/mhbench-handover_easy-handover_easy-ee-0/gaussian"
    gaussian.mkdir(parents=True)
    (gaussian / "features.hdf5").write_bytes(b"features")
    (gaussian / "best.ckpt").write_bytes(b"checkpoint")
    command = r'''
        set -euo pipefail
        POLICY_DIR=$1
        source "$2"
        gaudp_parse_stage_args handover 0 0
        gaudp_find_gaussian_artifact features.hdf5
        gaudp_find_gaussian_artifact best.ckpt
    '''

    assert _run(command, tmp_path, SCRIPT) == [
        str(gaussian / "features.hdf5"),
        str(gaussian / "best.ckpt"),
    ]


def test_eval_ignores_an_empty_canonical_run_before_using_legacy(tmp_path):
    canonical = tmp_path / "checkpoints/mhbench-handover-unitree_g1x2_centralized-joint-0"
    canonical.mkdir(parents=True)
    legacy = tmp_path / "checkpoints/mhbench-handover_easy-unitree_g1x2_centralized-joint-0"
    (legacy / "policy").mkdir(parents=True)
    (legacy / "policy" / "best.ckpt").write_bytes(b"checkpoint")
    command = r'''
        set -euo pipefail
        POLICY_DIR=$1 CKPT_DIR=$1/checkpoints CKPT_NAME=handover SEED=0 CKPT_TAG= MODEL_DIR_KEY=model_dir
        source "$2"
        source "$3"
        run_names
    '''

    assert _run(command, tmp_path, COMMON, SERVE) == [
        "mhbench-handover_easy-unitree_g1x2_centralized-joint-0:model_dir"
    ]

    (canonical / "policy").mkdir()
    (canonical / "policy" / "last.ckpt").write_bytes(b"checkpoint")
    assert _run(command, tmp_path, COMMON, SERVE) == [
        "mhbench-handover-unitree_g1x2_centralized-joint-0:model_dir"
    ]
