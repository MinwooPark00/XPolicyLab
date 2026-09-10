#!/usr/bin/env bash
# Finetune Psi0 on MHBench.
#
#   bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
#   PSI0_MICRO_BATCH=8 GLOBAL_BATCH_SIZE=32 bash train.sh mhbench multitask \
#       unitree_g1x2_decentralized joint 0 0
#
# The recipe is the paper's, unchanged except for the batch size:
# scripts/train/psi0/finetune-real-psi0.sh -- full fine-tuning of the ~500M
# MM-DiT action expert with the Qwen3-VL-2B backbone frozen (`--model.no-tune-vlm`),
# lr 1e-4 cosine with 1000 warmup steps, wd 1e-6, betas 0.95/0.999, grad clip 1.0,
# bf16, 40k steps, a 30-step chunk, action and state padded to 36, 240x320 frames
# with augmentation, and training-time RTC masking at max_delay 8. No LoRA: the
# paper does not use one and neither does any script upstream ships.
#
# What differs, and why:
#   global batch 32   the benchmark's shared budget (32 x 40k for every language
#                     policy here) rather than the paper's 128. One number, so a
#                     comparison is about the method and not the compute.
#   layout            the dataset's columns are already in Psi0's own 36D order
#                     (scripts/build_psi0_lerobot.py, configs/psi0/psi0_layout.py),
#                     so nothing about the model config changes for MHBench.
#   run directory     --train.output_dir is the benchmark's run-dir name and
#                     --timestamp is fixed, because Psi0 composes its run
#                     directory as <output_dir>/<train.name>/<exp>...<timestamp>
#                     and a fresh timestamp per launch would make a requeued job
#                     start over in a new directory instead of resuming.
#   wandb             project/entity/id given explicitly: scripts/train.py passes
#                     the project to init_trackers as an argument, so WANDB_PROJECT
#                     in the environment is ignored, and the run *id* is what
#                     log_eval_to_wandb.py looks a run up by.
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
PSI_ROOT="${POLICY_DIR}/psi0"
PSI_PY="${MHBENCH_PSI0_PY:-${PSI_ROOT}/.venv/bin/python}"

# CKPT_TAG separates runs that differ in what they train but not in task, mode
# or seed -- without it a tagged run would resume from the untagged one.
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}${CKPT_TAG:+-${CKPT_TAG}}"
run_root="${POLICY_DIR}/checkpoints/${ckpt_setting}"
# The dataset carries no seed and no tag: tagged runs share it, as GR00T's do.
data_id="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
data_root="${PSI0_DATA_ROOT:-${POLICY_DIR}/data}"

[ -x "${PSI_PY}" ] || { echo "no Psi0 python at ${PSI_PY} -- run policy/Psi0/install.sh" >&2; exit 1; }
# scripts/train.py asserts load_dotenv() on its fourth line, before it prints
# anything -- so say what is missing here rather than let it fail on a node.
[ -f "${PSI_ROOT}/.env" ] || {
  echo "no ${PSI_ROOT}/.env -- scripts/train.py requires one; install.sh writes it" >&2; exit 1; }
[ -f "${data_root}/${data_id}/meta/stats_psi0.json" ] || {
  echo "missing Psi0 dataset ${data_root}/${data_id} (or its meta/stats_psi0.json)" >&2
  echo "build it with:  bash baselines/scripts/prepare_multitask_data.sh psi0" >&2
  exit 1; }
: "${PSI0_BASE_MODEL:?set it to the pretrained Psi0 VLM directory (site.env)}"
: "${PSI0_ACTION_HEADER:?set it to the pretrained Psi0 action header directory (site.env)}"

val_repo="${data_id}_val"
[ -d "${data_root}/${val_repo}" ] || {
  # Psi0 falls back to the train repo when val_repo_ids is empty, which would
  # report a validation loss measured on the training set.
  echo "missing held-out dataset ${data_root}/${val_repo} -- rebuild with prepare_multitask_data.sh psi0" >&2
  exit 1; }

export CUDA_VISIBLE_DEVICES="${gpu_id}"
NUM_GPUS="${NUM_GPUS:-$(tr ',' '\n' <<< "${gpu_id}" | sed '/^$/d' | wc -l | xargs)}"

# The benchmark's budget, split into what fits a card. train_batch_size is
# per device, so global = micro x accumulation x GPUs.
MAX_STEPS="${MAX_STEPS:-40000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
# 8 per device is what the probe measured at 1.34 it/s on a 48 GB A6000, and
# every pool the training hook targets has at least that much. A smaller
# micro-batch only buys more accumulation steps for the same global 32.
MICRO_BATCH="${PSI0_MICRO_BATCH:-8}"
per_step=$(( MICRO_BATCH * NUM_GPUS ))
if (( GLOBAL_BATCH_SIZE % per_step != 0 )); then
  echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} is not a multiple of PSI0_MICRO_BATCH x GPUs (${per_step})" >&2
  exit 1
fi
GRAD_ACCUM="${PSI0_GRAD_ACCUM:-$(( GLOBAL_BATCH_SIZE / per_step ))}"
# 10000, so the four steps the benchmark's sweep evaluates (10k/20k/30k/40k) are
# exactly the four written, and max_checkpoints_to_keep=5 cannot delete one.
SAVE_STEPS="${SAVE_STEPS:-10000}"
EVAL_STEPS="${PSI0_EVAL_STEPS:-1000}"
EVAL_BATCHES="${PSI0_EVAL_BATCHES:-20}"
LR="${PSI0_LR:-1e-4}"
WARMUP="${PSI0_WARMUP_STEPS:-1000}"
# The rest of the optimiser, defaulting to the paper's numbers. MHBench's
# training hook sets pi0.5's instead (2.5e-5 -> 2.5e-6 cosine, 0.9/0.95,
# wd 1e-10) so the three VLA baselines share one schedule.
WEIGHT_DECAY="${PSI0_WEIGHT_DECAY:-1e-6}"
BETA1="${PSI0_BETA1:-0.95}"
BETA2="${PSI0_BETA2:-0.999}"
EPS="${PSI0_EPS:-1e-8}"
MIN_LR="${PSI0_MIN_LR:-}"
LR_SCHEDULER="${PSI0_LR_SCHEDULER:-$([ -n "${MIN_LR}" ] && echo cosine_with_min_lr || echo cosine)}"
# LoRA in pi0.5's shape (psi/utils/lora.py): rank 0 leaves that part as the
# paper trains it. PSI0_TUNE_VISION=1 trains the vision tower and projector in
# full, which is what pi0.5 does with SigLIP; PSI0_GRAD_CKPT=1 checkpoints the
# VLM's activations.
LORA_LLM_RANK="${PSI0_LORA_LLM_RANK:-0}"
LORA_LLM_ALPHA="${PSI0_LORA_LLM_ALPHA:-16}"
LORA_DIT_RANK="${PSI0_LORA_DIT_RANK:-0}"
LORA_DIT_ALPHA="${PSI0_LORA_DIT_ALPHA:-32}"
LORA_DROPOUT="${PSI0_LORA_DROPOUT:-0}"
TUNE_VISION="${PSI0_TUNE_VISION:-0}"
TUNE_DIT_FULL="${PSI0_TUNE_DIT_FULL:-0}"
GRAD_CKPT="${PSI0_GRAD_CKPT:-0}"

# Fixed, so the run directory is the same on every launch and a requeue resumes.
TIMESTAMP="${PSI0_TIMESTAMP:-mhbench}"
TRAIN_NAME="${PSI0_TRAIN_NAME:-finetune}"

# Resume from whatever is already there. The run directory carries the batch
# size and GPU count in its name, so a run relaunched with a different split of
# the same global batch is a different directory -- find it rather than rebuild
# the name.
resume_args=()
if [ "${PSI0_RESUME:-1}" = 1 ]; then
  latest=$(ls -d "${run_root}/${TRAIN_NAME}"/*/checkpoints/ckpt_* 2>/dev/null | sort -V | tail -1 || true)
  if [ -n "${latest}" ]; then
    resume_dir=$(dirname "$(dirname "${latest}")")
    resume_args=(--train.resume_from_checkpoint="${resume_dir}")
    echo "[Psi0] resuming from ${latest#${run_root}/}"
  fi
fi

mkdir -p "${run_root}"
export WANDB_ENTITY="${WANDB_ENTITY:-mhbench_baselines}"
WANDB_PROJECT="${WANDB_PROJECT:-MHBench-Psi0}"
WANDB_RUN_ID="${WANDB_RUN_ID:-${ckpt_setting}}"

cd "${PSI_ROOT}"

args=(
  finetune_real_psi0_config
  --seed="${seed}"
  --exp="${ckpt_setting}"
  --timestamp="${TIMESTAMP}"
  --train.name="${TRAIN_NAME}"
  --train.output_dir="${run_root}"
  --train.data_parallel=ddp
  --train.mixed_precision=bf16
  --train.train_batch_size="${MICRO_BATCH}"
  --train.gradient_accumulation_steps="${GRAD_ACCUM}"
  --train.max_checkpoints_to_keep=5
  --train.learning_rate="${LR}"
  --train.max_training_steps="${MAX_STEPS}"
  --train.warmup_ratio=None
  --train.warmup_steps="${WARMUP}"
  --train.checkpointing_steps="${SAVE_STEPS}"
  --train.validation_steps="${EVAL_STEPS}"
  --train.val_num_batches="${EVAL_BATCHES}"
  --train.max_grad_norm=1.0
  --train.lr_scheduler_type="${LR_SCHEDULER}"
  --train.lr_scheduler_kwargs.weight_decay="${WEIGHT_DECAY}"
  --train.lr_scheduler_kwargs.betas "${BETA1}" "${BETA2}"
  --train.lr_scheduler_kwargs.eps="${EPS}"
  --log.report_to=wandb
  --wandb.project="${WANDB_PROJECT}"
  --wandb.entity="${WANDB_ENTITY}"
  --wandb.id="${WANDB_RUN_ID}"
  --wandb.name="${WANDB_NAME:-$WANDB_RUN_ID}"
  --wandb.resume="${WANDB_RESUME:-allow}"
  --data.root_dir="${data_root}"
  --data.train_repo_ids="${data_id}"
  --data.val_repo_ids="${val_repo}"
  # The camera keeps the name every other MHBench view has; the rest of the
  # repack defaults (states / action / task) already match the dataset.
  --data.transform.repack.image-keys observation.images.ego
  --data.transform.repack.state-key=states
  --data.transform.repack.action-key=action
  --data.transform.repack.instruction-key=task
  --data.transform.repack.pad-action-dim=36
  --data.transform.repack.pad-state-dim=36
  --data.transform.field.stat-path=meta/stats_psi0.json
  --data.transform.field.stat-action-key=action
  --data.transform.field.stat-state-key=states
  --data.transform.field.action_norm_type=bounds
  --data.transform.field.no-use-norm-mask
  --data.transform.field.normalize-state
  --data.transform.field.pad-action-dim=36
  --data.transform.field.pad-state-dim=36
  --data.transform.model.img-aug
  --data.transform.model.resize.size 240 320
  --data.transform.model.center_crop.size 240 320
  --model.model_name_or_path="${PSI0_BASE_MODEL}"
  --model.pretrained-action-header-path="${PSI0_ACTION_HEADER}"
  --model.noise-scheduler=flow
  --model.train-diffusion-steps=1000
  --model.n_conditions=0
  --model.action-chunk-size=30
  --model.action-dim=36
  --model.action-exec-horizon=30
  --model.observation-horizon=1
  --model.odim=36
  --model.view_feature_dim=2048
  --model.no-tune-vlm
  # One learning rate for every group, as pi0.5 has one; the paper's per-group
  # values only matter when the VLM trains in full, which it never does here.
  --model.lang_backbone_lr="${LR}"
  --model.vision_tower_lr="${LR}"
  --model.mm_projector_lr="${LR}"
  --model.lora_llm_rank="${LORA_LLM_RANK}"
  --model.lora_llm_alpha="${LORA_LLM_ALPHA}"
  --model.lora_dit_rank="${LORA_DIT_RANK}"
  --model.lora_dit_alpha="${LORA_DIT_ALPHA}"
  --model.lora_dropout="${LORA_DROPOUT}"
  --model.no-use_film
  --model.no-combined_temb
  --model.rtc
  --model.max-delay=8
  "${resume_args[@]}"
)
[ -n "${MIN_LR}" ] && args+=(--train.scheduler_specific_kwargs.min_lr="${MIN_LR}")
[ "${TUNE_VISION}" = 1 ] && args+=(--model.tune-mm-vision --model.tune-mm-mlp)
[ "${TUNE_DIT_FULL}" = 1 ] && args+=(--model.tune-dit-full)
[ "${GRAD_CKPT}" = 1 ] && args+=(--model.gradient_checkpointing)
[ "${PSI0_FROZEN_VLM_BF16:-1}" = 0 ] && args+=(--model.no-frozen-vlm-bf16)
# PSI0_EXTRA is split on whitespace on purpose: it is the escape hatch for a
# one-off flag, and anything with a space in it belongs in a named knob above.
if [ -n "${PSI0_EXTRA:-}" ]; then
  read -r -a extra <<< "${PSI0_EXTRA}"
  args+=("${extra[@]}")
fi

echo "[Psi0] run dir : ${run_root}/${TRAIN_NAME}/${ckpt_setting}.b${GLOBAL_BATCH_SIZE}.gpus${NUM_GPUS}.${TIMESTAMP}"
echo "[Psi0] batch   : ${MICRO_BATCH} x ${GRAD_ACCUM} accum x ${NUM_GPUS} gpu = ${GLOBAL_BATCH_SIZE}"
echo "[Psi0] steps   : ${MAX_STEPS} (save every ${SAVE_STEPS}, validate every ${EVAL_STEPS})"
echo "[Psi0] optim   : lr=${LR} min_lr=${MIN_LR:-none} sched=${LR_SCHEDULER} warmup=${WARMUP} wd=${WEIGHT_DECAY} betas=${BETA1}/${BETA2} eps=${EPS}"
echo "[Psi0] lora    : llm r=${LORA_LLM_RANK} a=${LORA_LLM_ALPHA}; dit r=${LORA_DIT_RANK} a=${LORA_DIT_ALPHA}; dropout=${LORA_DROPOUT}; tune_vision=${TUNE_VISION} tune_dit_full=${TUNE_DIT_FULL} grad_ckpt=${GRAD_CKPT}"
echo "[Psi0] wandb   : ${WANDB_ENTITY}/${WANDB_PROJECT} id=${WANDB_RUN_ID}"
echo "[Psi0] data    : ${data_root}/${data_id} (+ ${val_repo})"

if [ "${PSI0_PRINT_ONLY:-0}" = 1 ]; then
  printf '%q ' torchrun --nnodes=1 --nproc_per_node="${NUM_GPUS}" scripts/train.py "${args[@]}"; echo
  exit 0
fi

# torchrun's default rendezvous port is 29500 on every job, and two jobs on one
# node collide -- the second dies at startup with EADDRINUSE, which is how probe
# 2176684 was lost on node47. Ask for a free one; the job id is the fallback for
# a checkout without XPolicyLab's helper.
if [ -z "${MASTER_PORT:-}" ]; then
  MASTER_PORT=$(bash "${POLICY_DIR}/../../utils/get_free_port.sh" 2>/dev/null) || MASTER_PORT=""
  [ -n "${MASTER_PORT}" ] || MASTER_PORT=$(( 20000 + (${SLURM_JOB_ID:-$$} % 20000) ))
fi
echo "[Psi0] master_port: ${MASTER_PORT}"

exec "${PSI_ROOT}/.venv/bin/torchrun" \
  --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  scripts/train.py "${args[@]}"
