#!/bin/bash
set -e

# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [rotation_rep]
# Output convention: checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/
#   checkpoints/{arm1,arm2}_latest.ckpt -- LatentToM's own two-checkpoint save
#   layout (train_diffusion_sheaf_split_workspace.py), one file per arm, kept
#   as-is rather than flattened to match single-policy adapters like DP.
# rotation_rep defaults to "quat" (22D/arm action); "rot6d" (26D/arm) is the
# other option -- it's baked into the converted data on disk by
# process_data.sh, so it also selects which data/<...>-<rotation_rep>/
# directory this run reads and (when not the task yaml's own 22D default)
# overrides task.shape_meta.arm{1,2}_action.shape to match.
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6
rotation_rep=${7:-quat}

if [[ -z "${bench_name}" || -z "${ckpt_name}" || -z "${env_cfg_type}" || -z "${action_type}" || -z "${seed}" || -z "${gpu_id}" ]]; then
  echo "Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [rotation_rep]" >&2
  exit 1
fi
if [[ "${rotation_rep}" != "quat" && "${rotation_rep}" != "rot6d" ]]; then
  echo "[LatentToM] rotation_rep must be 'quat' or 'rot6d', got '${rotation_rep}'" >&2
  exit 1
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
run_dir="${POLICY_DIR}/checkpoints/${run_setting}"
dataset_path="${POLICY_DIR}/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${rotation_rep}"

if [ ! -d "${dataset_path}" ]; then
    bash "${POLICY_DIR}/process_data.sh" "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "" "${rotation_rep}"
fi

action_dim_args=()
if [[ "${rotation_rep}" == "rot6d" ]]; then
    action_dim_args=(task.shape_meta.arm1_action.shape=[26] task.shape_meta.arm2_action.shape=[26])
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
    "${action_dim_args[@]}" \
    "${@:8}"

echo "[LatentToM] checkpoints written to ${run_dir}/checkpoints/{arm1,arm2}_latest.ckpt"
