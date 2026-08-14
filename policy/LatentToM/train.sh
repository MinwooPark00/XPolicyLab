#!/bin/bash
set -e

# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
# Output convention: checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/
#   checkpoints/{arm1,arm2}_latest.ckpt -- LatentToM's own two-checkpoint save
#   layout (train_diffusion_sheaf_split_workspace.py), one file per arm, kept
#   as-is rather than flattened to match single-policy adapters like DP.
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

if [[ -z "${bench_name}" || -z "${ckpt_name}" || -z "${env_cfg_type}" || -z "${action_type}" || -z "${seed}" || -z "${gpu_id}" ]]; then
  echo "Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
run_dir="${POLICY_DIR}/checkpoints/${run_setting}"
dataset_path="${POLICY_DIR}/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"

if [ ! -d "${dataset_path}" ]; then
    bash "${POLICY_DIR}/process_data.sh" "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}"
fi

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

python "${POLICY_DIR}/train.py" \
    --config-name=sheaf_xarm_split_diffusion_workspace \
    task=mhbench_cocarry_sheaf_split \
    task.dataset_path="${dataset_path}" \
    task_name="${ckpt_name}" \
    training.seed="${seed}" \
    hydra.run.dir="${run_dir}" \
    logging.mode=offline \
    "${@:7}"

echo "[LatentToM] checkpoints written to ${run_dir}/checkpoints/{arm1,arm2}_latest.ckpt"
