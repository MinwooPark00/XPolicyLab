#!/usr/bin/env bash
set -euo pipefail

# Usage: process_data.sh <bench> <ckpt> <env_cfg> <action_type> [max_demos]
bench=${1:?bench is required}
ckpt=${2:?ckpt is required}
env_cfg=${3:?env_cfg is required}
action_type=${4:?action_type is required}
max_demos=${5:-}
if [[ "${action_type}" != "ee" ]]; then
    echo "[GauDP] only action_type=ee is supported" >&2
    exit 2
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MHBENCH_ROOT="$(cd "${POLICY_DIR}/../../../.." && pwd)"
# The task is `env_cfg`, the word that names baselines/env_cfg/<name>.yml, not
# `bench` -- under the shared eval runner `bench` is always `mhbench`. The task
# directories are spelled without the underscore (door_passage -> doorpassage),
# the same pair of spellings baselines/scripts/eval_policy.sbatch accepts, and
# the LeRobot export is the `lerobot/` child of that directory.
#
# The two older layouts are still tried, in the order they existed, so a
# converted copy from before the datasets/ reorg still resolves:
#   datasets/<bench>_test   the pre-reorg per-task directory
#   datasets/data           the original raw-HDF5 root
case "${env_cfg}" in
  door_passage) task_dir=doorpassage ;;
  frame_hang)   task_dir=framehang ;;
  *)            task_dir="${env_cfg}" ;;
esac
for candidate in \
    "${MHBENCH_ROOT}/datasets/${task_dir}/lerobot" \
    "${MHBENCH_ROOT}/datasets/${bench}_test" \
    "${MHBENCH_ROOT}/datasets/data"; do
    if [[ -e "${candidate}" ]]; then default_source="${candidate}"; break; fi
done
source_path="${MHBENCH_DATASET_PATH:-${default_source:-${MHBENCH_ROOT}/datasets/${task_dir}/lerobot}}"
if [[ ! -e "${source_path}" ]]; then
    echo "[GauDP] no dataset at ${source_path}" >&2
    echo "[GauDP] Run scripts/export_lerobot.py for '${task_dir}' in the MHBench repo," >&2
    echo "[GauDP] or point MHBENCH_DATASET_PATH at an export kept elsewhere." >&2
    exit 2
fi
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
print(f"[GauDP] source   {root}")
print(f"[GauDP] exported {written.get('written_at', '?')[:19]} @ {str(written.get('commit', '?'))[:7]}"
      f"  {info.get('total_episodes')} episodes / {info.get('total_frames')} frames")
sentences = prov.get("task", {}).get("sentences") or ["?"]
print(f"[GauDP] task     {sentences[0]}")
PROV
fi

output="${POLICY_DIR}/data/${bench}-${ckpt}-${env_cfg}-${action_type}.hdf5"
extra=()
if [[ -n "${max_demos}" ]]; then extra+=(--max-demos "${max_demos}"); fi
if [[ "${GAUDP_USE_SCENE:-0}" == "1" ]]; then extra+=(--use-scene); fi

python_bin="${GAUDP_PYTHON:-python}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "[GauDP] Python executable not found: ${python_bin}; activate the GauDP environment or set GAUDP_PYTHON" >&2
    exit 2
fi
PYTHONNOUSERSITE=1 "${python_bin}" "${POLICY_DIR}/process_data.py" "${source_path}" "${output}" "${extra[@]}"
