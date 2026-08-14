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
default_source="${MHBENCH_ROOT}/datasets/${bench}_test"
if [[ ! -e "${default_source}" ]]; then
    # Backward-compatible fallback for the original HDF5 dataset layout.
    default_source="${MHBENCH_ROOT}/datasets/data"
fi
source_path="${MHBENCH_DATASET_PATH:-${default_source}}"
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
