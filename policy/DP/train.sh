#!/bin/bash

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

# Get Action Dimension from env_cfg_type
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}"); echo -e "\033[33m[INFO] Action dim: ${action_dim}\033[0m"
state_dim=$(bash "${UTILS_DIR}/get_state_dim.sh" "${ROOT_DIR}" "${env_cfg_type}"); echo -e "\033[33m[INFO] State dim: ${state_dim}\033[0m"
num_cameras=$(bash "${UTILS_DIR}/get_num_cameras.sh" "${ROOT_DIR}" "${env_cfg_type}"); echo -e "\033[33m[INFO] Num cameras: ${num_cameras}\033[0m"
alg_name=robot_dp

# default_task.yaml declares no camera by default (struct-locked to just
# agent_pos/action), so every run appends its own: 1 camera -> head_cam
# (RobotImageDataset/model.py's single-camera slot), 2 -> left_cam+right_cam
# (model.py's MHBENCH_CAMERA_SLOT, ego_a/ego_b). Both sides read the same
# num_cameras/camera_map, so training and eval always agree on key names.
if [ "${num_cameras}" -ge 2 ]; then
    EXTRA_CAMERA_ARGS=(
        "+task.shape_meta.obs.left_cam.shape=[3,240,320]"
        "+task.shape_meta.obs.left_cam.type=rgb"
        "+task.shape_meta.obs.right_cam.shape=[3,240,320]"
        "+task.shape_meta.obs.right_cam.type=rgb"
    )
else
    EXTRA_CAMERA_ARGS=(
        "+task.shape_meta.obs.head_cam.shape=[3,240,320]"
        "+task.shape_meta.obs.head_cam.type=rgb"
    )
fi

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
val_zarr_path="data/${data_setting}_val.zarr"

if [ ! -d "${zarr_path}" ]; then
    bash process_data.sh ${bench_name} ${ckpt_name} ${env_cfg_type} ${action_type}
fi

val_override=()
if [ -d "${val_zarr_path}" ]; then
    echo -e "\033[33m[INFO] Held-out validation set found: ${val_zarr_path}\033[0m"
    val_override=(task.dataset.val_zarr_path="${val_zarr_path}")
fi

# Optional overrides, empty = robot_dp.yaml's own value. The temporal ones
# exist because that file's defaults were chosen for 10 Hz data: at MHBench's
# 50 Hz an 8-step horizon is 0.16 s, over which the demonstrated action moves
# 1.5% of each dimension's range -- close enough to "repeat agent_pos" that a
# network can score a near-zero training loss without learning the task.
# DP_RUN_TAG keeps a run's checkpoints out of an existing run's directory.
OVERRIDES=()
[ -n "${DP_HORIZON:-}" ]        && OVERRIDES+=("horizon=${DP_HORIZON}")
[ -n "${DP_N_OBS_STEPS:-}" ]    && OVERRIDES+=("n_obs_steps=${DP_N_OBS_STEPS}")
[ -n "${DP_N_ACTION_STEPS:-}" ] && OVERRIDES+=("n_action_steps=${DP_N_ACTION_STEPS}")
[ -n "${DP_NUM_EPOCHS:-}" ]     && OVERRIDES+=("training.num_epochs=${DP_NUM_EPOCHS}")
[ -n "${DP_CKPT_EVERY:-}" ]     && OVERRIDES+=("training.checkpoint_every=${DP_CKPT_EVERY}")
# task.dataset.batch_size interpolates dataloader.batch_size, so it follows.
[ -n "${DP_BATCH_SIZE:-}" ]     && OVERRIDES+=("dataloader.batch_size=${DP_BATCH_SIZE}" "val_dataloader.batch_size=${DP_BATCH_SIZE}")
[ -n "${DP_RUN_TAG:-}" ]        && OVERRIDES+=("checkpoint.run_tag=${DP_RUN_TAG}")
[ -n "${DP_WANDB_MODE:-}" ]     && OVERRIDES+=("logging.mode=${DP_WANDB_MODE}")
[ -n "${DP_LR:-}" ]             && OVERRIDES+=("optimizer.lr=${DP_LR}")
[ -n "${DP_NUM_WORKERS:-}" ]    && OVERRIDES+=("dataloader.num_workers=${DP_NUM_WORKERS}" "val_dataloader.num_workers=${DP_NUM_WORKERS}")
[ -n "${DP_OBS_NOISE:-}" ]      && OVERRIDES+=("training.obs_noise=${DP_OBS_NOISE}")
# `random_crop: True` in robot_dp.yaml is a silent no-op while crop_shape is
# null -- MultiImageObsEncoder only builds a CropRandomizer when a crop shape
# is given, so the vision encoder trains with no augmentation at all. Written
# HxW (e.g. 216x288) rather than as a list, because sbatch --export splits its
# own argument on commas and `[216,288]` would not survive the trip.
if [ -n "${DP_CROP_SHAPE:-}" ]; then
    OVERRIDES+=("policy.obs_encoder.crop_shape=[${DP_CROP_SHAPE%x*},${DP_CROP_SHAPE#*x}]")
fi
[ ${#OVERRIDES[@]} -gt 0 ] && echo -e "\033[33m[INFO] overrides: ${OVERRIDES[*]}\033[0m"

python train.py --config-name="${alg_name}.yaml" \
                bench_name="${bench_name}" \
                task.name="${ckpt_name}" \
                "task.shape_meta.action.shape=[${action_dim}]" \
                "task.shape_meta.obs.agent_pos.shape=[${state_dim}]" \
                "${EXTRA_CAMERA_ARGS[@]}" \
                task.dataset.zarr_path="${zarr_path}" \
                "${val_override[@]}" \
                training.debug=$DEBUG \
                training.seed=${seed} \
                training.device="cuda:0" \
                exp_name=${exp_name} \
                logging.mode=${wandb_mode} \
                setting=${env_cfg_type} \
                ${OVERRIDES[@]+"${OVERRIDES[@]}"}
