#!/bin/bash
set -euo pipefail

# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [num_gpus]
bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}
seed=${5}
gpu_id=${6}
if [[ $# -ge 7 ]]; then
    num_gpus=${7}
elif [[ "${gpu_id}" == *,* ]]; then
    IFS=',' read -r -a gpu_ids <<< "${gpu_id}"
    num_gpus=${#gpu_ids[@]}
else
    num_gpus=1
fi
train_seed=${seed}
if [[ "${train_seed}" -le 0 ]]; then
    train_seed=1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
POLICY_DIR="${ROOT_DIR}/XPolicyLab/policy/FastWAM"
FASTWAM_DIR="${POLICY_DIR}/FastWAM"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_DIR}/checkpoints"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT_DIR}:${FASTWAM_DIR}:${FASTWAM_DIR}/src:${PYTHONPATH:-}"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
# Default dataset_id is the 4-tuple data_key. Set FASTWAM_DATASET_ID to point
# at a differently named dataset without changing the train.sh argument shape.
data_key="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"

# MHBench diverges from the robotwin defaults in three ways, all derived here
# so training and serving cannot disagree (the sim_mhbench.yaml task override
# is the same yaml):
#   - its own task yaml per mode (2 ego views stacked / 1 ego view);
#   - state is the raw joint-angle vector (86/43), NOT the robotwin
#     state=action convention -- the eval adapter reassembles exactly that
#     vector from mhbench_state[*].joint_pos;
#   - a separate lerobot_val dataset (the benchmark's fixed 50/10 split), so
#     the random val_set_proportion carve stays off;
#   - wandb on, into the shared MHBench projects, with the run id pinned to
#     the checkpoint run-dir name so eval can write its rollout back.
if [[ "${bench_name}" == "mhbench" ]]; then
    state_dim=$(bash "${UTILS_DIR}/get_state_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
    case "${env_cfg_type}" in
        unitree_g1x2_centralized)   task_name="mhbench_uncond_2cam_384_1e-4" ;;
        unitree_g1x2_decentralized) task_name="mhbench_uncond_1cam_240_1e-4" ;;
        *) echo "[ERROR] unknown mhbench env_cfg_type: ${env_cfg_type}" >&2; exit 2 ;;
    esac
else
    state_dim="${action_dim}"
    task_name="robotwin_uncond_3cam_384_1e-4"
fi
dataset_id="${FASTWAM_DATASET_ID:-${data_key}}"
converted_root="${POLICY_DIR}/data/${dataset_id}"
dataset_dir="${converted_root}/lerobot"
stats_path="${converted_root}/dataset_stats.json"
text_cache_dir="${FASTWAM_DIR}/data/text_embeds_cache/xpolicylab/${dataset_id}"
ckpt_setting="${FASTWAM_CKPT_SETTING:-${data_key}-${seed}}"
action_dit="${FASTWAM_DIR}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
batch_size="${FASTWAM_BATCH_SIZE:-8}"
gradient_accumulation_steps="${FASTWAM_GRADIENT_ACCUMULATION_STEPS:-1}"
num_workers="${FASTWAM_NUM_WORKERS:-8}"
# Upstream trainer.py:39 does `int(cfg.num_epochs)` unconditionally, so even when
# the task yaml sets `num_epochs: null` (i.e. you want pure max_steps-based
# training, which `_estimate_total_train_steps()` honors), num_epochs must still
# be a non-null integer or trainer init crashes. Inject a large dummy by default;
# user can override with FASTWAM_NUM_EPOCHS.
num_epochs_override="${FASTWAM_NUM_EPOCHS:-16}"
# Which ZeRO stage launches the run: 1 (upstream default), 2, or 2off. A 5.6B
# trainable DiT's fp32 optimizer partition alone is ~67 GB, so few-GPU runs
# want stage 2 (optimizer AND gradients sharded); MHBench's training hook
# sets 2. 2off adds CPU offload of that optimizer partition, which is what
# makes 24 GB cards viable at all: stage 2 still replicates the bf16 weights
# (~12 GB) on every GPU, so without offload even eight 24 GB GPUs leave no
# room for activations. Offload trades that for a CPU AdamW step, so pair it
# with gradient accumulation to keep the optimizer-step rate down.
zero_stage="${FASTWAM_ZERO:-1}"
case "${zero_stage}" in 1|2|2off) ;; *) echo "[ERROR] FASTWAM_ZERO must be 1, 2 or 2off, got ${zero_stage}" >&2; exit 2 ;; esac

if [[ ! -d "${dataset_dir}/meta" ]]; then
    echo "[ERROR] LeRobot dataset not found: ${dataset_dir}/meta"
    echo "Prepare a LeRobot v2.1 dataset under ${converted_root}/lerobot/ and dataset_stats.json."
    if [[ -n "${FASTWAM_DATASET_ID:-}" ]]; then
        echo "FASTWAM_DATASET_ID=${FASTWAM_DATASET_ID}"
    fi
    exit 1
fi

if [[ ! -f "${action_dit}" ]]; then
    echo "[ERROR] Missing ActionDiT backbone: ${action_dit}"
    echo "Run in the FastWAM policy environment:"
    echo "  cd ${FASTWAM_DIR}"
    echo "  python scripts/preprocess_action_dit_backbone.py --model-config configs/model/fastwam.yaml --output ${action_dit} --device cuda --dtype bfloat16"
    exit 1
fi

if [[ ! -d "${text_cache_dir}" || -z "$(find "${text_cache_dir}" -name '*.pt' -print -quit 2>/dev/null)" ]]; then
    echo "[ERROR] Missing T5 text embedding cache: ${text_cache_dir}"
    echo "Precompute it with the upstream script in the FastWAM policy environment:"
    echo "  cd ${FASTWAM_DIR}"
    echo "  python scripts/precompute_text_embeds.py \\"
    echo "    task=${task_name} \\"
    echo "    data.train.dataset_dirs=[${dataset_dir}] \\"
    echo "    data.val.dataset_dirs=[${dataset_dir}] \\"
    echo "    data.train.text_embedding_cache_dir=${text_cache_dir} \\"
    echo "    data.val.text_embedding_cache_dir=${text_cache_dir}"
    exit 1
fi

cd "${FASTWAM_DIR}"

# The val dataset: mhbench converts the held-out episodes separately
# (lerobot_val); elsewhere the robotwin behaviour (same dirs, random carve)
# is unchanged.
val_dataset_dir="${dataset_dir}"
if [[ "${bench_name}" == "mhbench" ]]; then
    val_dataset_dir="${converted_root}/lerobot_val"
    if [[ ! -d "${val_dataset_dir}/meta" ]]; then
        echo "[ERROR] LeRobot val dataset not found: ${val_dataset_dir}/meta"
        echo "Convert both splits (see MHBench's prepare_fastwam_data.sh)."
        exit 1
    fi
fi

train_common=(
    "task=${task_name}"
    "seed=${train_seed}"
    "batch_size=${batch_size}"
    "gradient_accumulation_steps=${gradient_accumulation_steps}"
    "num_workers=${num_workers}"
    "num_epochs=${num_epochs_override}"
    "data.train.dataset_dirs=[${dataset_dir}]"
    "data.val.dataset_dirs=[${val_dataset_dir}]"
    "data.train.text_embedding_cache_dir=${text_cache_dir}"
    "data.val.text_embedding_cache_dir=${text_cache_dir}"
    "data.train.pretrained_norm_stats=${stats_path}"
    "data.val.pretrained_norm_stats=${stats_path}"
    "data.train.shape_meta.action.0.raw_shape=${action_dim}"
    "data.train.shape_meta.action.0.shape=${action_dim}"
    "data.train.shape_meta.state.0.raw_shape=${state_dim}"
    "data.train.shape_meta.state.0.shape=${state_dim}"
    "data.val.shape_meta.action.0.raw_shape=${action_dim}"
    "data.val.shape_meta.action.0.shape=${action_dim}"
    "data.val.shape_meta.state.0.raw_shape=${state_dim}"
    "data.val.shape_meta.state.0.shape=${state_dim}"
    "data.train.processor.action_output_dim=${action_dim}"
    "data.train.processor.proprio_output_dim=${state_dim}"
    "data.val.processor.action_output_dim=${action_dim}"
    "data.val.processor.proprio_output_dim=${state_dim}"
    "output_dir=${POLICY_DIR}/checkpoints/${ckpt_setting}"
)

# Anything else, verbatim, as a space-separated list of hydra overrides --
# probe runs (num_epochs=1 wandb.mode=offline) that must not be named here.
# shellcheck disable=SC2206
if [[ -n "${FASTWAM_EXTRA:-}" ]]; then
    train_common+=(${FASTWAM_EXTRA})
fi

if [[ "${bench_name}" == "mhbench" ]]; then
    # wandb.init reads WANDB_RUN_ID from the environment (the trainer passes
    # no id of its own), so pinning it to the run-dir name lets a requeue
    # resume the same run and lets eval write its rollout numbers back.
    export WANDB_ENTITY="${WANDB_ENTITY:-mhbench_baselines}"
    export WANDB_RUN_ID="${WANDB_RUN_ID:-${ckpt_setting}}"
    export WANDB_RESUME="${WANDB_RESUME:-allow}"
    # One resumable state is enough, and a ZeRO-2 state is tens of GB: keep
    # only the newest unless the caller says otherwise.
    export FASTWAM_KEEP_STATES="${FASTWAM_KEEP_STATES:-1}"
    train_common+=(
        "wandb.enabled=true"
        "wandb.workspace=${WANDB_ENTITY}"
        "wandb.project=${WANDB_PROJECT:-MHBench-FastWAM}"
        "wandb.name=${ckpt_setting}"
        "wandb.mode=${WANDB_MODE:-online}"
    )
    # A requeued job resumes from the newest saved trainer state (optimizer,
    # scheduler, step), which is what makes #SBATCH --requeue on the MHBench
    # runner cost steps rather than the run.
    # `|| latest_state=` matters: with `set -o pipefail` a first run (no state
    # dir yet) would otherwise fail the whole pipeline on ls's exit 2 and kill
    # the script silently, since ls's stderr is discarded. `sort -V` because
    # plain sort puts step_2500 after step_10000.
    latest_state=$(ls -1d "${POLICY_DIR}/checkpoints/${ckpt_setting}/checkpoints/state/step_"* 2>/dev/null | sort -V | tail -1) || latest_state=
    if [[ -n "${latest_state}" ]]; then
        echo "[FastWAM] resuming from ${latest_state}"
        train_common+=("resume=${latest_state}")
    fi
fi

bash "scripts/train_zero${zero_stage}.sh" "${num_gpus}" "${train_common[@]}"
