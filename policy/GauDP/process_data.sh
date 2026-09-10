#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${POLICY_DIR}/launcher_args.sh"
gaudp_parse_data_args "$@"
gaudp_require_joint_action_type

MHBENCH_ROOT="$(cd "${POLICY_DIR}/../../../.." && pwd)"
DATASETS_ROOT="${GAUDP_DATASETS_ROOT:-${MHBENCH_ROOT}/datasets}"

is_lerobot_root() {
    [[ -f "$1/meta/info.json" && -d "$1/data" && -d "$1/videos" ]]
}

# Accept either the LeRobot root itself or a task directory containing a
# legacy `lerobot/` child. This also makes MHBENCH_DATASET_PATH forgiving.
resolve_lerobot_root() {
    local candidate=$1
    if is_lerobot_root "${candidate}"; then
        printf '%s\n' "${candidate}"
        return 0
    fi
    if is_lerobot_root "${candidate}/lerobot"; then
        printf '%s\n' "${candidate}/lerobot"
        return 0
    fi
    return 1
}

# Dataset folders may use compact spellings while env_cfg names contain an
# underscore.
task_names=("${task}")
case "${task}" in
  door_passage) task_names+=(doorpassage) ;;
  frame_hang)   task_names+=(framehang) ;;
  # Legacy long-form callers can still supply the retired task spelling even
  # though task-first calls canonicalize it in launcher_args.sh.
  handover_easy) task_names+=(handover) ;;
esac

source_path=""
if [[ -n "${MHBENCH_DATASET_PATH:-}" ]]; then
    source_path="$(resolve_lerobot_root "${MHBENCH_DATASET_PATH}" || true)"
    searched=("${MHBENCH_DATASET_PATH}" "${MHBENCH_DATASET_PATH}/lerobot")
else
    searched=()
    for task_name in "${task_names[@]}"; do
        candidate="${DATASETS_ROOT}/${task_name}"
        searched+=("${candidate}" "${candidate}/lerobot")
        source_path="$(resolve_lerobot_root "${candidate}" || true)"
        if [[ -n "${source_path}" ]]; then
            break
        fi
    done
fi

if [[ -z "${source_path}" ]]; then
    echo "[GauDP] no LeRobot dataset found for task '${task}'" >&2
    echo "[GauDP] searched:" >&2
    printf '  - %s\n' "${searched[@]}" >&2
    echo "[GauDP] Expected meta/info.json, data/, and videos/ under the dataset root." >&2
    echo "[GauDP] Default location: ${DATASETS_ROOT}/<task>" >&2
    echo "[GauDP] Available task folders:" >&2
    find "${DATASETS_ROOT}" -mindepth 2 -maxdepth 3 -path '*/meta/info.json' \
        -printf '  - %h\n' 2>/dev/null | sed 's#/meta$##' >&2 || true
    echo "[GauDP] Pass the task as argument 5 or set MHBENCH_DATASET_PATH." >&2
    exit 2
fi
echo "[GauDP] task     ${task}"
echo "[GauDP] source   ${source_path}"
# Say which export this is. There are several copies of `datasets/` on a
# machine that has more than one worktree of this repo, they are NOT
# interchangeable, and nothing downstream can tell them apart: a scene gets
# redesigned and the directory keeps its name. cocarry went from a board on two
# stands to a laundry basket in a kitchen between 2026-08-14 and 2026-08-21,
# same `datasets/cocarry/lerobot` path, and a policy trained on one and
# evaluated in the other produces a video rather than an error.
if [[ -f "${source_path}/meta/mhbench_provenance.json" ]]; then
  python3 - "${source_path}" <<'PROV'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
prov = json.loads((root / "meta" / "mhbench_provenance.json").read_text())
info = json.loads((root / "meta" / "info.json").read_text())
written = prov.get("env_args", {}).get("provenance", {})
print(f"[GauDP] exported {written.get('written_at', '?')[:19]} @ {str(written.get('commit', '?'))[:7]}"
      f"  {info.get('total_episodes')} episodes / {info.get('total_frames')} frames")
sentences = prov.get("task", {}).get("sentences") or ["?"]
print(f"[GauDP] task     {sentences[0]}")
PROV
fi

output="$(gaudp_data_path)"
extra=()
if [[ -n "${max_demos}" ]]; then extra+=(--max-demos "${max_demos}"); fi
if [[ "${GAUDP_USE_SCENE:-0}" == "1" ]]; then extra+=(--use-scene); fi

python_bin="${GAUDP_PYTHON:-python}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "[GauDP] Python executable not found: ${python_bin}; activate the GauDP environment or set GAUDP_PYTHON" >&2
    exit 2
fi
PYTHONNOUSERSITE=1 "${python_bin}" "${POLICY_DIR}/process_data.py" "${source_path}" "${output}" "${extra[@]}"
