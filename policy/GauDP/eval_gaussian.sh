#!/usr/bin/env bash
set -euo pipefail

# Usage: eval_gaussian.sh <bench> <ckpt> <env_cfg> <action_type> <seed> <gpu> [checkpoint] [extra args]
bench=${1:?bench is required}; ckpt=${2:?ckpt is required}; env_cfg=${3:?env_cfg is required}
action_type=${4:?action_type is required}; seed=${5:?seed is required}; gpu=${6:?gpu is required}
if [[ "${action_type}" != "ee" ]]; then echo "[GauDP] only action_type=ee is supported" >&2; exit 2; fi
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data="${POLICY_DIR}/data/${bench}-${ckpt}-${env_cfg}-${action_type}.hdf5"
run="${POLICY_DIR}/checkpoints/${bench}-${ckpt}-${env_cfg}-${action_type}-${seed}"
if [[ ! -f "${data}" ]]; then bash "${POLICY_DIR}/process_data.sh" "${bench}" "${ckpt}" "${env_cfg}" "${action_type}"; fi

source "${POLICY_DIR}/resolve_noposplat_checkpoint.sh"
require_gaussian_supervision "${data}"
shift 6
if (( $# > 0 )) && [[ "$1" != -* ]]; then
    checkpoint=$1
    shift
    label="$(basename "${checkpoint}")"
    label="${label%.ckpt}"
else
    checkpoint="$(resolve_noposplat_checkpoint "${data}" "${POLICY_DIR}")"
    label="noposplat"
fi

if [[ ! -s "${checkpoint}" ]]; then
    echo "[GauDP] Gaussian checkpoint not found or empty: ${checkpoint}" >&2
    exit 2
fi

output="${run}/gaussian_eval/${label}"
python_bin="${GAUDP_PYTHON:-python}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "[GauDP] Python executable not found: ${python_bin}; activate the GauDP environment or set GAUDP_PYTHON" >&2
    exit 2
fi
CUDA_VISIBLE_DEVICES="${gpu}" PYTHONNOUSERSITE=1 "${python_bin}" "${POLICY_DIR}/eval_gaussian.py" \
    --data "${data}" --output "${output}" --checkpoint "${checkpoint}" --seed "${seed}" "$@"
