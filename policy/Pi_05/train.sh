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
# ckpt_setting is the run directory name; pass it verbatim as ckpt_name to eval.sh.
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
gpu_count=$(awk -F',' '{print NF}' <<<"${gpu_id}")
fsdp_devices="${OPENPI_FSDP_DEVICES:-$(( gpu_count < 2 ? 1 : 2 ))}"

# MHBench derives everything from the task and the target, so that a run cannot
# be trained under one TrainConfig and served under another. Elsewhere the
# config name and the LeRobot repo are the caller's to choose.
#   centralized:   ckpt_name = <task>
#   decentralized: ckpt_name = <task>_robot_a | <task>_robot_b
mhbench_repo_id=""
if [[ "${bench_name}" == "mhbench" ]]; then
  case "${env_cfg_type}" in
    unitree_g1x2_centralized)   mhbench_task="${ckpt_name}";            mhbench_target="centralized" ;;
    unitree_g1x2_decentralized) mhbench_task="${ckpt_name%_robot_*}";   mhbench_target="robot_${ckpt_name##*_robot_}" ;;
    *) echo "unknown mhbench env_cfg_type ${env_cfg_type}" >&2; exit 2 ;;
  esac
  train_config_name="pi05_mhbench_${mhbench_task}_${mhbench_target}"
  mhbench_repo_id="mhbench-${mhbench_task}"
  # openpi's init_wandb passes only name and project, and takes the entity from
  # the environment. The account default happens to be mhbench_baselines, but a
  # default is not a guarantee, so name it -- and derive the tags here rather
  # than in the sbatch runners, because those are spooled at submit time and a
  # queued job would never see a change to them.
  export WANDB_ENTITY="${WANDB_ENTITY:-mhbench_baselines}"
  export WANDB_TAGS="${WANDB_TAGS:-pi05,lora,${mhbench_task},${mhbench_target}}"
else
  train_config_name="${OPENPI_TRAIN_CONFIG_NAME:-pi05_base_aloha_full_sim_arx-x5_seed_0}"
  lerobot_repo_id="${OPENPI_LEROBOT_REPO_ID:-${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}}"
fi

mkdir -p "${ckpt_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

# LeRobot loads parquet via HuggingFace datasets, which builds pyarrow mmap cache
# under HF_DATASETS_CACHE. Keep dataset on shared storage, but use per-host local
# cache to avoid NFS lock contention when multiple nodes train concurrently.
LOCAL_CACHE_ROOT="${OPENPI_LOCAL_CACHE_ROOT:-/tmp/openpi-cache-$(hostname)}"
mkdir -p "${LOCAL_CACHE_ROOT}/hf/datasets" "${LOCAL_CACHE_ROOT}/jax"
export HF_DATASETS_CACHE="${LOCAL_CACHE_ROOT}/hf/datasets"
export JAX_COMPILATION_CACHE_DIR="${LOCAL_CACHE_ROOT}/jax"

# Resume if this run already has an orbax step directory, overwrite only when
# starting from nothing. `initialize_checkpoint_dir` rmtree's the directory on
# --overwrite, and train.sh creates it above, so hard-coding --overwrite (as the
# upstream script does) means a requeued job throws away everything it had done.
resume_flag="--overwrite"
if compgen -G "${ckpt_dir}/[0-9]*" > /dev/null; then
  resume_flag="--resume"
fi

echo "[Pi_05] train_config_name=${train_config_name}"
echo "[Pi_05] checkpoint_dir=${ckpt_dir} (${resume_flag})"
echo "[Pi_05] local_cache_root=${LOCAL_CACHE_ROOT}"

cd "${POLICY_DIR}/openpi/"

if [[ "${bench_name}" == "mhbench" ]]; then
  echo "[Pi_05] lerobot_repo_id=${mhbench_repo_id} (from the config; not overridden)"
  # Training hard-fails without norm stats, and they are cheap next to a run.
  # The asset path is assets/<config>/<repo_id>/, which is where
  # compute_norm_stats writes and where DataConfigFactory reads.
  if [[ ! -f "assets/${train_config_name}/${mhbench_repo_id}/norm_stats.json" ]]; then
    echo "[Pi_05] norm stats missing; computing them first"
    uv run scripts/compute_norm_stats.py --config-name "${train_config_name}"
  fi
  # No --data.repo-id: it would also change asset_id, and the stats just
  # computed would stop being found.
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    uv run scripts/train.py "${train_config_name}" \
      --exp-name="${ckpt_setting}" \
      --fsdp-devices="${fsdp_devices}" \
      --checkpoint-dir-override="${ckpt_dir}" \
      --seed="${seed}" \
      "${resume_flag}"
else
  echo "[Pi_05] lerobot_repo_id=${lerobot_repo_id}"
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    uv run scripts/train.py "${train_config_name}" \
      --exp-name="${ckpt_setting}" \
      --data.repo-id="${lerobot_repo_id}" \
      --fsdp-devices="${fsdp_devices}" \
      --checkpoint-dir-override="${ckpt_dir}" \
      --seed="${seed}" \
      "${resume_flag}"
fi
