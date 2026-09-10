#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${POLICY_DIR}/launcher_args.sh"
gaudp_parse_stage_args "$@"
gaudp_require_joint_action_type
data="$(gaudp_data_path)"
run="$(gaudp_run_dir)"
# Nothing in the reconstruction encoder reads a state or an action, so a
# finetune done before the joint-space switch is still the right encoder for
# the same scene. Redoing it would cost ~26 h to land the same weights, so an
# existing checkpoint ends the stage; GAUDP_FORCE_GAUSSIAN=1 finetunes anyway.
if [[ "${GAUDP_FORCE_GAUSSIAN:-0}" != "1" ]] && existing="$(gaudp_find_gaussian_artifact best.ckpt)"; then
    echo "[GauDP] reusing the fine-tuned Gaussian encoder: ${existing}"
    echo "[GauDP] set GAUDP_FORCE_GAUSSIAN=1 to fine-tune it again"
    exit 0
fi

if [[ ! -f "${data}" ]]; then bash "${POLICY_DIR}/process_data.sh" "${bench}" "${ckpt}" "${env_cfg}" "${action_type}" "${task}"; fi

source "${POLICY_DIR}/resolve_noposplat_checkpoint.sh"
require_gaussian_supervision "${data}"
pretrained="$(resolve_noposplat_checkpoint "${data}" "${POLICY_DIR}")"
python_bin="${GAUDP_PYTHON:-python}"

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONNOUSERSITE=1 "${python_bin}" "${POLICY_DIR}/train_gaussian.py" \
    --data "${data}" --output "${run}/gaussian" --pretrained "${pretrained}" --seed "${seed}" "${extra[@]}"
