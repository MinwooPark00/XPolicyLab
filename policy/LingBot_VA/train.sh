#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBODOJO_TEST_ROOT="$(cd "${POLICY_DIR}/../../.." && pwd)"

resolve_lerobot_repo_id() {
  if [[ -n "${LEROBOT_DATASET_REPO_ID:-}" ]]; then
    echo "${LEROBOT_DATASET_REPO_ID}"
    return
  fi
  case "${env_cfg_type}" in
    arx_x5) echo "RoboDojo_sim_arx-x5_v30" ;;
    *) echo "RoboDojo_sim_${env_cfg_type}" ;;
  esac
}

ckpt_setting="${LINGBOT_CKPT_SETTING:-${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"

if [[ "${bench_name}" == "mhbench" ]]; then
  # MHBench trains on the latent dataset prepare_multitask_data.sh builds; the
  # action normalisation and the empty-prompt embedding travel with it.
  data_id="mhbench-${ckpt_name}-${env_cfg_type}-${action_type}"
  export LINGBOT_VA_DATASET_PATH="${LINGBOT_VA_DATASET_PATH:-${POLICY_DIR}/data/${data_id}/lerobot}"
  export LINGBOT_NORM_STAT="${LINGBOT_NORM_STAT:-${LINGBOT_VA_DATASET_PATH}/meta/lingbot_norm_stat.json}"
  export CONFIG_NAME="${LINGBOT_VA_CONFIG_NAME:-mhbench_train}"
  # A requeued job picks up its own newest training state; a fresh run finds
  # none and starts at step 0.
  export LINGBOT_RESUME_FROM="${LINGBOT_RESUME_FROM:-auto}"
  export WANDB_PROJECT="${WANDB_PROJECT:-MHBench-LingBot_VA}"
  export WANDB_NAME="${WANDB_NAME:-${ckpt_setting}}"
  export WANDB_RUN_ID="${WANDB_RUN_ID:-${ckpt_setting}}"
else
  export XPOLICYLAB_LEROBOT_DATA_ROOT="${XPOLICYLAB_LEROBOT_DATA_ROOT:-${LEROBOT_DATA_ROOT:-${ROBODOJO_TEST_ROOT}/data}}"
  export LEROBOT_DATA_ROOT="${XPOLICYLAB_LEROBOT_DATA_ROOT}"
  export LEROBOT_DATASET_REPO_ID="${LEROBOT_DATASET_REPO_ID:-$(resolve_lerobot_repo_id)}"
  export CONFIG_NAME="${LINGBOT_VA_CONFIG_NAME:-robotwin30_train}"
  export LINGBOT_VA_DATASET_PATH="${LINGBOT_VA_DATASET_PATH:-${LEROBOT_DATA_ROOT}/${LEROBOT_DATASET_REPO_ID}}"
fi

mkdir -p "${ckpt_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export NGPU
NGPU="$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES}" | sed '/^$/d' | wc -l | xargs)"
export WORLD_SIZE="${WORLD_SIZE:-${NGPU}}"
export LINGBOT_VA_BASE_MODEL_PATH="${LINGBOT_VA_BASE_MODEL_PATH:-}"
export PYTHONHASHSEED="${seed}"

echo "[LingBot_VA] bench=${bench_name} config=${CONFIG_NAME}"
echo "[LingBot_VA] dataset=${LINGBOT_VA_DATASET_PATH}"
echo "[LingBot_VA] base_model=${LINGBOT_VA_BASE_MODEL_PATH:-<unset>}"
echo "[LingBot_VA] checkpoint_dir=${ckpt_dir}"
echo "[LingBot_VA] gpus=${NGPU} lora_rank=${LINGBOT_LORA_RANK:-32}" \
     "global_batch=${LINGBOT_GLOBAL_BATCH:-32} max_steps=${LINGBOT_MAX_STEPS:-40000}"

bash "${POLICY_DIR}/lingbot_va/script/run_va_posttrain.sh" \
  --save-root "${ckpt_dir}" \
  --seed "${seed}"
