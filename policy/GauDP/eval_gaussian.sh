#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${POLICY_DIR}/launcher_args.sh"
gaudp_parse_stage_args "$@"
gaudp_require_joint_action_type
data="$(gaudp_data_path)"
run="$(gaudp_run_dir)"
if [[ ! -f "${data}" ]]; then bash "${POLICY_DIR}/process_data.sh" "${bench}" "${ckpt}" "${env_cfg}" "${action_type}" "${task}"; fi

source "${POLICY_DIR}/resolve_noposplat_checkpoint.sh"
require_gaussian_supervision "${data}"
set -- "${extra[@]}"
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
