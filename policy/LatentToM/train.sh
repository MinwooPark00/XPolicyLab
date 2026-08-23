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
# Defaults to this run's own directory, which ties one conversion to one
# ckpt_name. LATENTTOM_DATA_DIR unties them so a sweep can share one converted
# copy while `ckpt_name` still keeps the checkpoints apart.
dataset_path="${LATENTTOM_DATA_DIR:-${POLICY_DIR}/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}}"

if [ ! -d "${dataset_path}" ]; then
    # Only the default path is auto-converted: a LATENTTOM_DATA_DIR that does
    # not exist is a typo, not a request to convert.
    if [[ -n "${LATENTTOM_DATA_DIR:-}" ]]; then
        echo "[LatentToM] LATENTTOM_DATA_DIR=${LATENTTOM_DATA_DIR} is not a directory" >&2
        exit 1
    fi
    bash "${POLICY_DIR}/process_data.sh" "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}"
fi

# Trailing arguments (from $7 on) are hydra overrides. `training.debug=true`
# is the plumbing check, so that one logs offline; everything else stays
# online. The config's own tags never name the task, hence the list here.
wandb_args=("logging.tags=[latenttom,${env_cfg_type},${action_type},${ckpt_name}]")
for arg in "${@:7}"; do
    case "${arg,,}" in
        training.debug=true|training.debug=1)
            wandb_args+=(logging.mode=offline)
            break
            ;;
    esac
done

export HYDRA_FULL_ERROR=1

if [[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[LatentToM] slurm allocated CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; ignoring gpu_id=${gpu_id}"
else
    export CUDA_VISIBLE_DEVICES=${gpu_id}
fi

# One BLAS/OpenMP thread per process. DataLoader workers cannot set this
# themselves (see _limit_threads_once in the dataset), so they inherit it.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# The policy env's own interpreter, not whatever `python` resolves to: a venv
# activated by the login shell (MHBench's .venv, which holds Isaac Sim and none
# of the training dependencies) keeps its PATH entry ahead of conda's, so
# `python` there is the wrong one and the run dies on the first import.
PY="${CONDA_PREFIX:-}/bin/python"
[ -x "$PY" ] || PY=python

# One task config for every MHBench two-G1 scene -- they present identical
# shapes to this policy, and the scene is named by `dataset_path` and the run
# directory below. LATENTTOM_TASK_CONFIG points at a different one (upstream's
# `sheaf_split_coffee_bean_pouring`, say, or a per-task override).
"$PY" "${POLICY_DIR}/train.py" \
    --config-name=sheaf_xarm_split_diffusion_workspace \
    task="${LATENTTOM_TASK_CONFIG:-mhbench_sheaf_split}" \
    task.dataset_path="${dataset_path}" \
    task_name="${ckpt_name}" \
    training.seed="${seed}" \
    hydra.run.dir="${run_dir}" \
    "${wandb_args[@]}" \
    "${@:7}"

echo "[LatentToM] checkpoints written to ${run_dir}/checkpoints/{arm1,arm2}_latest.ckpt"
