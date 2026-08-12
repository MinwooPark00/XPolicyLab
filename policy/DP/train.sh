#!/bin/bash
#SBATCH -J dp-cocarry
#SBATCH --partition=suma_a6000
#SBATCH --gres=gpu:1
#SBATCH --output=example.out
#SBATCH --time=01:00:00

# bash train.sh mhbench cocarry mhbench_cocarry joint 0 0
# sbatch -J dp-cocarry --partition=suma_a6000 --gres=gpu:1 --output=example.out --time=01:00:00 --wrap="bash train.sh mhbench cocarry mhbench_cocarry joint 0 0"
# sbatch -J dp-cocarry --gres=gpu:1 --output=example.out --time=01:00:00 --wrap="bash train.sh mhbench cocarry unitree_g1x2_decentralized joint 0 0"
# sbatch -J dp-cocarry --gres=gpu:1 --output=example.out --time=01:00:00 --wrap="bash train.sh mhbench cocarry unitree_g1x2_centralized joint 0 0"



bench_name=${1}
ckpt_name=${2} # run name
env_cfg_type=${3}
action_type=${4}
seed=${5}
gpu_id=${6}

DEBUG=False

addition_info=train
exp_name=${ckpt_name}-robot_dp-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# Get action/state dimensions and camera count from env_cfg_type
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}"); echo -e "\033[33m[INFO] Action dim: ${action_dim}\033[0m"
state_dim=$(bash "${UTILS_DIR}/get_state_dim.sh" "${ROOT_DIR}" "${env_cfg_type}"); echo -e "\033[33m[INFO] State dim: ${state_dim}\033[0m"
num_cameras=$(bash "${UTILS_DIR}/get_num_cameras.sh" "${ROOT_DIR}" "${env_cfg_type}"); echo -e "\033[33m[INFO] Num cameras: ${num_cameras}\033[0m"

# task/default_task.yaml only declares head_cam by default (the other camera
# keys are commented out there since most robots only use one); insert
# left_cam/right_cam here instead of uncommenting them in the shared yaml, so
# robots that don't set "num_cameras" in _robot_info.json are unaffected.
# Shape must match default_task.yaml's image_shape anchor ([3, 240, 320]).
EXTRA_CAMERA_ARGS=()
if [ "${num_cameras}" -ge 2 ]; then
    EXTRA_CAMERA_ARGS+=("+task.shape_meta.obs.left_cam.shape=[3,240,320]" "+task.shape_meta.obs.left_cam.type=rgb")
fi
if [ "${num_cameras}" -ge 3 ]; then
    EXTRA_CAMERA_ARGS+=("+task.shape_meta.obs.right_cam.shape=[3,240,320]" "+task.shape_meta.obs.right_cam.type=rgb")
fi

alg_name=robot_dp

if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
zarr_path="data/${data_setting}.zarr"

if [ ! -d "${zarr_path}" ]; then
    bash process_data.sh ${bench_name} ${ckpt_name} ${env_cfg_type} ${action_type}
fi

python train.py --config-name="${alg_name}.yaml" \
                bench_name="${bench_name}" \
                task.name="${ckpt_name}" \
                "task.shape_meta.action.shape=[${action_dim}]" \
                "task.shape_meta.obs.agent_pos.shape=[${state_dim}]" \
                "${EXTRA_CAMERA_ARGS[@]}" \
                task.dataset.zarr_path="${zarr_path}" \
                training.debug=$DEBUG \
                training.seed=${seed} \
                training.device="cuda:0" \
                exp_name=${exp_name} \
                logging.mode=${wandb_mode} \
                setting=${env_cfg_type}
