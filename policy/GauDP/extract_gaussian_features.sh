#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${POLICY_DIR}/launcher_args.sh"
gaudp_parse_stage_args "$@"
gaudp_require_joint_action_type
data="$(gaudp_data_path)"
run="$(gaudp_run_dir)"
if [[ ! -f "${data}" ]]; then bash "${POLICY_DIR}/process_data.sh" "${bench}" "${ckpt}" "${env_cfg}" "${action_type}" "${task}"; fi

set -- "${extra[@]}"
if (( $# > 0 )) && [[ "$1" != -* ]]; then
    gaussian=$1
    shift
elif [[ -n "${GAUDP_GAUSSIAN_CKPT:-}" ]]; then
    gaussian="${GAUDP_GAUSSIAN_CKPT}"
elif gaussian="$(gaudp_find_gaussian_artifact best.ckpt)"; then
    :
else
    # Keep Gaussian fine-tuning optional by falling back to the official
    # pretrained checkpoint when no run-local checkpoint exists.
    source "${POLICY_DIR}/resolve_noposplat_checkpoint.sh"
    gaussian="$(resolve_noposplat_checkpoint "${data}" "${POLICY_DIR}")"
fi
if [[ ! -s "${gaussian}" ]]; then
    echo "[GauDP] Gaussian checkpoint not found or empty: ${gaussian}" >&2
    exit 2
fi

# Reuse an existing cache rather than writing a second 94 GB copy: extraction
# depends on the RGB frames, the camera order and the encoder checkpoint, none
# of which the joint-space switch changed, and extract_gaussian_features.py
# re-validates all three before it decides the cache is complete.
if [[ -n "${GAUDP_GAUSSIAN_FEATURES:-}" ]]; then
    features="${GAUDP_GAUSSIAN_FEATURES}"
elif ! features="$(gaudp_find_gaussian_artifact features.hdf5)"; then
    features="${run}/gaussian/features.hdf5"
fi
echo "[GauDP] gaussian ${gaussian}"
echo "[GauDP] features ${features}"
python_bin="${GAUDP_PYTHON:-python}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "[GauDP] Python executable not found: ${python_bin}; activate the GauDP environment or set GAUDP_PYTHON" >&2
    exit 2
fi
CUDA_VISIBLE_DEVICES="${gpu}" PYTHONNOUSERSITE=1 "${python_bin}" "${POLICY_DIR}/extract_gaussian_features.py" \
    --data "${data}" --checkpoint "${gaussian}" --output "${features}" --seed "${seed}" "$@"
