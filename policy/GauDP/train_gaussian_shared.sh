#!/usr/bin/env bash
# One NoPoSplat encoder for every task, instead of one per task.
#
#   bash train_gaussian_shared.sh [seed] [gpu] [task...] [-- extra args]
#   bash train_gaussian_shared.sh 0 0 cocarry handover framehang pouring
#
# The encoder reads RGB, depth and camera geometry only -- no state, no action
# -- so nothing about it is task-specific, and a shared one means a new task
# inherits an encoder rather than paying another finetune. Every task's frames
# are concatenated into one training set; the checkpoint lands in the shared
# run directory that launcher_args.sh searches before the per-task ones.
#
# Measured on handover alone (31.5k frames): val PSNR 23.24 after one epoch,
# 26.18 at epoch 15, and best.ckpt tracks it exactly (min val/loss and max
# val/psnr agree). The benchmark's eight tasks are ~8x that, so one epoch here
# already carries more exposure than a per-task epoch and the same quality
# arrives within the first few; baselines/configs/GauDP.yaml sets the budget
# (GAUDP_SHARED_EPOCHS) and early stopping ends it when diversity stops paying.
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${POLICY_DIR}/launcher_args.sh"

seed=${1:-0}
gpu=${2:-0}
shift $(( $# >= 2 ? 2 : $# ))
tasks=()
extra=()
while (( $# )); do
    if [[ "$1" == "--" ]]; then shift; extra=("$@"); break; fi
    tasks+=("$1"); shift
done
[[ ${#tasks[@]} -gt 0 ]] || tasks=(cocarry handover framehang pouring copouring trashcollection cartservice tablealign)

bench=${GAUDP_BENCH:-mhbench}
action_type=$GAUDP_ACTION_TYPE
env_cfg=${GAUDP_ENV_CFG:-$GAUDP_DEFAULT_ENV_CFG}
out="$(gaudp_shared_gaussian_dir)/gaussian"

# Convert anything missing, and collect the datasets to train over.
datasets=()
for t in "${tasks[@]}"; do
    gaudp_task_config "$t"
    ckpt=$task
    data="$(gaudp_data_path)"
    if [[ ! -f "${data}" ]]; then
        echo "[GauDP][shared] converting ${task}"
        bash "${POLICY_DIR}/process_data.sh" "${bench}" "${ckpt}" "${env_cfg}" "${action_type}" "${task}"
    fi
    datasets+=("${data}")
done

source "${POLICY_DIR}/resolve_noposplat_checkpoint.sh"
for d in "${datasets[@]}"; do require_gaussian_supervision "${d}"; done
pretrained="$(resolve_noposplat_checkpoint "${datasets[0]}" "${POLICY_DIR}")"
python_bin="${GAUDP_PYTHON:-python}"

echo "[GauDP][shared] tasks    ${tasks[*]}"
echo "[GauDP][shared] datasets ${#datasets[@]}"
echo "[GauDP][shared] output   ${out}"
CUDA_VISIBLE_DEVICES="${gpu}" PYTHONNOUSERSITE=1 "${python_bin}" "${POLICY_DIR}/train_gaussian.py" \
    --data "${datasets[@]}" --output "${out}" --pretrained "${pretrained}" --seed "${seed}" \
    --wandb-run-name "shared-gaussian-seed${seed}${GAUDP_TAG:+-${GAUDP_TAG}}" \
    --wandb-tags "gaussian,shared,seed-${seed}" \
    ${extra[@]+"${extra[@]}"}
