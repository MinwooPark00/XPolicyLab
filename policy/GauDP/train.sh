#!/usr/bin/env bash
set -euo pipefail

# Usage: train.sh <bench> <ckpt> <env_cfg> <action_type> <seed> <gpu> [extra args]
bench=${1:?bench is required}; ckpt=${2:?ckpt is required}; env_cfg=${3:?env_cfg is required}
action_type=${4:?action_type is required}; seed=${5:?seed is required}; gpu=${6:?gpu is required}
if [[ "${action_type}" != "ee" ]]; then echo "[GauDP] only action_type=ee is supported" >&2; exit 2; fi
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data="${POLICY_DIR}/data/${bench}-${ckpt}-${env_cfg}-${action_type}.hdf5"
run="${POLICY_DIR}/checkpoints/${bench}-${ckpt}-${env_cfg}-${action_type}-${seed}"
gaussian_features="${GAUDP_GAUSSIAN_FEATURES:-${run}/gaussian/features.hdf5}"
python_bin="${GAUDP_PYTHON:-python}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "[GauDP] Python executable not found: ${python_bin}; activate the GauDP environment or set GAUDP_PYTHON" >&2
    exit 2
fi
if [[ ! -f "${data}" ]]; then
    bash "${POLICY_DIR}/process_data.sh" "${bench}" "${ckpt}" "${env_cfg}" "${action_type}"
fi
if [[ ! -f "${gaussian_features}" ]]; then
    echo "[GauDP] Offline Gaussian features are required: ${gaussian_features}" >&2
    echo "[GauDP] Run extract_gaussian_features.sh before policy training" >&2
    exit 2
fi
if [[ -n "${GAUDP_GAUSSIAN_CKPT:-}" ]]; then
    gaussian="${GAUDP_GAUSSIAN_CKPT}"
elif [[ -s "${run}/gaussian/best.ckpt" ]]; then
    gaussian="${run}/gaussian/best.ckpt"
else
    # The cache is authoritative: it records the exact checkpoint used during
    # extraction, including an official pretrained checkpoint outside run/.
    gaussian="$(PYTHONNOUSERSITE=1 "${python_bin}" - "${gaussian_features}" <<'PY'
import sys
import h5py

with h5py.File(sys.argv[1], "r") as source:
    print(str(source.attrs.get("gaussian_checkpoint", "")))
PY
)"
fi
if [[ ! -s "${gaussian}" ]]; then
    echo "[GauDP] Gaussian checkpoint not found or empty: ${gaussian}" >&2
    echo "[GauDP] Set GAUDP_GAUSSIAN_CKPT or re-run extract_gaussian_features.sh" >&2
    exit 2
fi
CUDA_VISIBLE_DEVICES="${gpu}" PYTHONNOUSERSITE=1 "${python_bin}" "${POLICY_DIR}/train_policy.py" \
    --data "${data}" --output "${run}/policy" --gaussian "${gaussian}" \
    --gaussian-features "${gaussian_features}" --seed "${seed}" "${@:7}"
