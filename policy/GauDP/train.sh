#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${POLICY_DIR}/launcher_args.sh"
gaudp_parse_stage_args "$@"
gaudp_require_joint_action_type
data="$(gaudp_data_path)"
run="$(gaudp_run_dir)"
# The Gaussian encoder and its feature cache read RGB and camera geometry
# only, so a cache built before the joint-space switch is still valid for the
# same export; `gaudp_find_gaussian_artifact` searches this run first and the
# pre-switch `-ee-` run second, and `gaudp/dataset.py` proves the match.
if [[ -n "${GAUDP_GAUSSIAN_FEATURES:-}" ]]; then
    gaussian_features="${GAUDP_GAUSSIAN_FEATURES}"
elif ! gaussian_features="$(gaudp_find_gaussian_artifact features.hdf5)"; then
    gaussian_features="${run}/gaussian/features.hdf5"
fi
python_bin="${GAUDP_PYTHON:-python}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "[GauDP] Python executable not found: ${python_bin}; activate the GauDP environment or set GAUDP_PYTHON" >&2
    exit 2
fi
if [[ ! -f "${data}" ]]; then
    bash "${POLICY_DIR}/process_data.sh" "${bench}" "${ckpt}" "${env_cfg}" "${action_type}" "${task}"
fi
if [[ ! -f "${gaussian_features}" ]]; then
    echo "[GauDP] Offline Gaussian features are required: ${gaussian_features}" >&2
    echo "[GauDP] Run extract_gaussian_features.sh before policy training" >&2
    exit 2
fi
if [[ -n "${GAUDP_GAUSSIAN_CKPT:-}" ]]; then
    gaussian="${GAUDP_GAUSSIAN_CKPT}"
elif gaussian="$(gaudp_find_gaussian_artifact best.ckpt)"; then
    :
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
echo "[GauDP] data     ${data}"
echo "[GauDP] run      ${run}"
echo "[GauDP] gaussian ${gaussian}"
echo "[GauDP] features ${gaussian_features}"
CUDA_VISIBLE_DEVICES="${gpu}" PYTHONNOUSERSITE=1 "${python_bin}" "${POLICY_DIR}/train_policy.py" \
    --data "${data}" --output "${run}/policy" --gaussian "${gaussian}" \
    --gaussian-features "${gaussian_features}" --seed "${seed}" "${extra[@]}"
