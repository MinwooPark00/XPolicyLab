#!/usr/bin/env bash
set -euo pipefail

# Usage: train_gaussian.sh <bench> <ckpt> <env_cfg> <action_type> <seed> <gpu> [extra args]
bench=${1:?bench is required}; ckpt=${2:?ckpt is required}; env_cfg=${3:?env_cfg is required}
action_type=${4:?action_type is required}; seed=${5:?seed is required}; gpu=${6:?gpu is required}
if [[ "${action_type}" != "ee" ]]; then echo "[GauDP] only action_type=ee is supported" >&2; exit 2; fi
if [[ -z "${GAUDP_NOPOSPLAT_CKPT:-}" ]]; then
    echo "[GauDP] set GAUDP_NOPOSPLAT_CKPT to the public NoPoSplat checkpoint" >&2; exit 2
fi
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data="${POLICY_DIR}/data/${bench}-${ckpt}-${env_cfg}-${action_type}.hdf5"
run="${POLICY_DIR}/checkpoints/${bench}-${ckpt}-${env_cfg}-${action_type}-${seed}"
if [[ ! -f "${data}" ]]; then bash "${POLICY_DIR}/process_data.sh" "${bench}" "${ckpt}" "${env_cfg}" "${action_type}"; fi
CUDA_VISIBLE_DEVICES="${gpu}" python "${POLICY_DIR}/train_gaussian.py" \
    --data "${data}" --output "${run}/gaussian" --pretrained "${GAUDP_NOPOSPLAT_CKPT}" --seed "${seed}" "${@:7}"
