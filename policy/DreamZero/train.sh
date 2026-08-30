#!/bin/bash
set -e
set -o pipefail

usage() {
    cat <<'EOF'
Usage:
  bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

Training data resolution order:
  1. LEROBOT_DATA_PATH (explicit override)
  2. <policy>/data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type> (process_data.sh output)
  3. <demo_root>/RobotDojo/RoboDojo_sim_arx-x5_v30 (shared default)

Optional environment overrides:
  LEROBOT_DATA_PATH                 Explicit LeRobot dataset root
  DREAMZERO_PRETRAINED_MODEL_PATH   Default: ./checkpoints/DreamZero-AgiBot, or ./checkpoints for flat layout
  WAN_CKPT_DIR                      Default: ./checkpoints/Wan2.1-I2V-14B-480P
  TOKENIZER_DIR                     Default: ./checkpoints/umt5-xxl, or Wan2.1 nested tokenizer fallback
  DREAMZERO_PREFLIGHT_ONLY          If 1, validate dataset and weights then exit.
  DREAMZERO_DRY_RUN                 If 1, print resolved command and exit before torchrun.
EOF
}

if [ "$#" -ne 6 ]; then
    usage >&2
    exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
DREAMZERO_DIR="${SCRIPT_DIR}/dreamzero"
export DREAMZERO_DIR

data_tag="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
processed_data_path="${SCRIPT_DIR}/data/${data_tag}"
default_lerobot_path="${ROOT_DIR}/RobotDojo/RoboDojo_sim_arx-x5_v30"
# Data resolution order: explicit override > process_data.sh output (4-tuple) > shared default.
if [ -n "${LEROBOT_DATA_PATH:-}" ]; then
    dataset_path="${LEROBOT_DATA_PATH}"
elif [ -f "${processed_data_path}/meta/info.json" ]; then
    dataset_path="${processed_data_path}"
else
    dataset_path="${default_lerobot_path}"
fi
# CKPT_TAG separates runs over the same data (a 30-step probe, a LoRA-rank
# sweep) -- without it a probe would write into, and .latest-point at, the
# real run's directory. Same convention as GR00T/FastWAM.
run_basename="${DREAMZERO_CKPT_SETTING:-${data_tag}-${seed}${CKPT_TAG:+-${CKPT_TAG}}}"
output_dir="${SCRIPT_DIR}/checkpoints/${run_basename}"

if [ ! -f "${dataset_path}/meta/info.json" ]; then
    echo "[DreamZero train][ERROR] LeRobot dataset info.json not found: ${dataset_path}/meta/info.json"
    echo "[DreamZero train][ERROR] Set LEROBOT_DATA_PATH to a LeRobot v3 root or DreamZero-compatible dataset root."
    exit 1
fi


IFS=',' read -ra GPU_ARRAY <<< "${gpu_id}"
num_gpus=${#GPU_ARRAY[@]}
num_gpus=${DREAMZERO_NUM_GPUS:-${num_gpus}}

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
echo "[DreamZero train] dataset=${dataset_path}"
echo "[DreamZero train] output_dir=${output_dir}"
echo "[DreamZero train] gpu_id=${gpu_id}, num_gpus=${num_gpus}, action_dim=${action_dim}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export HYDRA_FULL_ERROR=1
# Remember whether the caller named a project BEFORE defaulting it, so the
# mhbench block below can pick its own default without a caller-set value
# looking like one (an unconditional default here made `MHBench-DreamZero`
# dead code -- every mhbench run logged to `dreamzero`).
caller_wandb_project="${WANDB_PROJECT:-}"
export WANDB_PROJECT="${caller_wandb_project:-dreamzero}"
export PYTHONPATH="${DREAMZERO_DIR}:${SCRIPT_DIR}:${ROOT_DIR}/XPolicyLab:${ROOT_DIR}:${PYTHONPATH:-}"

checkpoints_dir="${SCRIPT_DIR}/checkpoints"
default_pretrained_model_path="${checkpoints_dir}/DreamZero-AgiBot"
if [ ! -d "${default_pretrained_model_path}" ] && [ -f "${checkpoints_dir}/config.json" ]; then
    default_pretrained_model_path="${checkpoints_dir}"
fi

default_tokenizer_dir="${checkpoints_dir}/umt5-xxl"
if [ ! -d "${default_tokenizer_dir}" ] && [ -d "${checkpoints_dir}/Wan2.1-I2V-14B-480P/google/umt5-xxl" ]; then
    default_tokenizer_dir="${checkpoints_dir}/Wan2.1-I2V-14B-480P/google/umt5-xxl"
fi

wan_ckpt_dir="${WAN_CKPT_DIR:-${checkpoints_dir}/Wan2.1-I2V-14B-480P}"
tokenizer_dir="${TOKENIZER_DIR:-${default_tokenizer_dir}}"
pretrained_model_path="${DREAMZERO_PRETRAINED_MODEL_PATH:-${default_pretrained_model_path}}"
max_steps="${DREAMZERO_MAX_STEPS:-5000}"
save_steps="${DREAMZERO_SAVE_STEPS:-2500}"
batch_size="${DREAMZERO_PER_DEVICE_BATCH_SIZE:-1}"
# The batch an optimizer step actually sees. conf.yaml leaves
# `global_batch_size: null`, which means base.py never touches
# gradient_accumulation_steps and the effective batch is
# per_device_train_batch_size x world_size -- so the same script trained a
# different batch on every allocation shape. Setting this pins it: base.py
# asserts it divides by per-device x world and derives the accumulation.
global_batch_size="${DREAMZERO_GLOBAL_BATCH_SIZE:-}"
dataloader_workers="${DREAMZERO_DATALOADER_WORKERS:-1}"
image_width="${DREAMZERO_IMAGE_WIDTH:-320}"
image_height="${DREAMZERO_IMAGE_HEIGHT:-176}"
action_horizon="${DREAMZERO_ACTION_HORIZON:-24}"
num_frames="${DREAMZERO_NUM_FRAMES:-33}"
max_chunk_size="${DREAMZERO_MAX_CHUNK_SIZE:-4}"
# tensorboard upstream; mhbench overrides to wandb below, where the project,
# entity and run id are already pinned.
report_to="${DREAMZERO_REPORT_TO:-${REPORT_TO:-tensorboard}}"
# Three views (scene + both egos) upstream; decentralized is one.
num_views="${DREAMZERO_NUM_VIEWS:-3}"
native_dojo_action="${DREAMZERO_NATIVE_DOJO_ACTION:-false}"
data_config="${DREAMZERO_DATA_CONFIG:-dreamzero/agibot_relative}"
if [ "${native_dojo_action}" = "1" ] || [ "${native_dojo_action}" = "true" ]; then
    native_dojo_action=true
    data_config="${DREAMZERO_DATA_CONFIG:-dreamzero/robodojo_native_relative}"
fi

# MHBench: its own data config (the mhbench_g1x2 anchors under the reused
# `agibot` embodiment key -- the robodojo_native precedent), full checkpoints
# by default (the policy server loads standalone models, not LoRA adapters),
# and the wandb run id pinned to the run-dir name so a requeue resumes the
# same run and eval can write its rollout numbers back.
if [ "${bench_name}" = "mhbench" ]; then
    # centralized  : one 70D policy over scene + both ego views.
    # decentralized: one 35D policy per robot over that robot's own ego view.
    # Same network either way -- only the width (handled by the DiT overrides
    # at the bottom of this script) and the view count differ.
    case "${env_cfg_type}" in
        # frame_seqlen is the DiT's latent tokens per video frame, and it is a
        # property of the CANVAS, not of one view: the views are tiled 2x2, so
        # three views make a 640x352 canvas and one view is just 320x176. With
        # the Wan2.1 VAE (8x spatial) and patch 2 that is 40x22 -> 880 tokens
        # against 20x11 -> 220. The script defaulted to 880 for both, so the
        # decentralized run built a 9-latent-frame clip, divided it by the
        # 4-tile figure, got 2 frames and no image blocks at all, and died in
        # the DiT's block-layout check (job 2116860). 220 gives 9 frames ->
        # 4 blocks -> a register of 4*(24+1) = 100, which is exactly the
        # action_register_length that check reported.
        unitree_g1x2_centralized)
            data_config="${DREAMZERO_DATA_CONFIG:-dreamzero/mhbench_relative}"
            num_views="${DREAMZERO_NUM_VIEWS:-3}"
            mhbench_frame_seqlen=880 ;;
        unitree_g1x2_decentralized)
            data_config="${DREAMZERO_DATA_CONFIG:-dreamzero/mhbench_relative_decentralized}"
            num_views="${DREAMZERO_NUM_VIEWS:-1}"
            mhbench_frame_seqlen=220 ;;
        *) echo "[DreamZero train][ERROR] unknown mhbench env_cfg_type: ${env_cfg_type}" >&2; exit 2 ;;
    esac
    report_to="${DREAMZERO_REPORT_TO:-${REPORT_TO:-wandb}}"
    export WANDB_PROJECT="${caller_wandb_project:-MHBench-DreamZero}"
    export WANDB_ENTITY="${WANDB_ENTITY:-mhbench_baselines}"
    export WANDB_RUN_ID="${WANDB_RUN_ID:-${run_basename}}"
    export WANDB_RESUME="${WANDB_RESUME:-allow}"
    DREAMZERO_SAVE_LORA_ONLY="${DREAMZERO_SAVE_LORA_ONLY:-false}"
    DREAMZERO_SAVE_TOTAL_LIMIT="${DREAMZERO_SAVE_TOTAL_LIMIT:-5}"   # base.py asserts >= 5
fi
# The held-out split, beside the training one under the same 4-tuple name.
# This loader has no notion of LeRobot `splits`, so the benchmark's fixed
# 50/10 is two dataset roots (baselines/scripts/prepare_dreamzero_data.sh
# writes both and links `<data_tag>_val`). Without one the trainer keeps its
# empty eval_dataset, which is what every DreamZero run before this had --
# training is unchanged, there is simply no val curve to read.
val_dataset_path="${LEROBOT_VAL_DATA_PATH:-${processed_data_path}_val}"
if [ -f "${val_dataset_path}/meta/info.json" ]; then
    eval_steps="${DREAMZERO_EVAL_STEPS:-${save_steps}}"
    VAL_ARGS=(
        "agibot_val_data_root=${val_dataset_path}"
        "eval_strategy=steps"
        "do_eval=true"
        "eval_steps=${eval_steps}"
        # The upstream default is 64, sized for a small model; this one is
        # 22.9B and evaluates at whatever the training step fits.
        "per_device_eval_batch_size=${DREAMZERO_PER_DEVICE_EVAL_BATCH_SIZE:-${batch_size}}"
        # How much of the held-out split one evaluation reads. At 1.0 a pass is
        # every val step -- 50,000 forwards, ~43 h, longer than the training
        # run. 0.005 is the deterministic leading 50 steps of each shard (250
        # samples decentralized, ~13 min); see the val_dataset node in the
        # mhbench data configs.
        "val_dataset.mixture_kwargs.shard_sampling_rate=${DREAMZERO_VAL_SHARD_RATE:-0.005}"
    )
    echo "[DreamZero train] val=${val_dataset_path} every ${eval_steps} steps (shard rate ${DREAMZERO_VAL_SHARD_RATE:-0.005})"
else
    VAL_ARGS=("val_dataset=null")
    echo "[DreamZero train] no held-out split at ${val_dataset_path} -- training without a val curve"
fi

python_cmd="${PYTHON:-$(command -v python || command -v python3 || true)}"
if [ -z "${python_cmd}" ]; then
    echo "[DreamZero train][ERROR] Python executable not found. Activate the dreamzero conda env first."
    exit 1
fi

require_file() {
    local path="$1"
    local hint="$2"
    if [ ! -f "${path}" ]; then
        echo "[DreamZero train][ERROR] Required file not found: ${path}"
        echo "[DreamZero train][ERROR] ${hint}"
        exit 1
    fi
}

require_dir() {
    local path="$1"
    local hint="$2"
    if [ ! -d "${path}" ]; then
        echo "[DreamZero train][ERROR] Required directory not found: ${path}"
        echo "[DreamZero train][ERROR] ${hint}"
        exit 1
    fi
}

if [ "${DREAMZERO_DRY_RUN:-0}" != "1" ]; then
    # DREAMZERO_PRETRAINED_MODEL_PATH=none trains from the Wan2.1 weights alone
    # (the documented new-embodiment fallback when the DreamZero-AgiBot init
    # refuses a different action head).
    if [ "${pretrained_model_path}" != "none" ]; then
        require_dir "${pretrained_model_path}" \
            "Set DREAMZERO_PRETRAINED_MODEL_PATH to the local DreamZero-AgiBot checkpoint directory (or 'none')."
    fi
    require_file "${wan_ckpt_dir}/models_t5_umt5-xxl-enc-bf16.pth" \
        "Download Wan-AI/Wan2.1-I2V-14B-480P or set WAN_CKPT_DIR to its local directory."
    require_file "${wan_ckpt_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
        "Download Wan-AI/Wan2.1-I2V-14B-480P or set WAN_CKPT_DIR to its local directory."
    require_file "${wan_ckpt_dir}/Wan2.1_VAE.pth" \
        "Download Wan-AI/Wan2.1-I2V-14B-480P or set WAN_CKPT_DIR to its local directory."
    require_dir "${tokenizer_dir}" \
        "Download google/umt5-xxl locally or set TOKENIZER_DIR to its directory."

    if [ "${DREAMZERO_PREFLIGHT_ONLY:-0}" = "1" ]; then
        echo "[DreamZero train] Preflight passed."
        exit 0
    fi

    mkdir -p "${output_dir}" "${SCRIPT_DIR}/checkpoints"
    echo "${output_dir}" > "${SCRIPT_DIR}/checkpoints/${run_basename}.latest"
fi

cd "${DREAMZERO_DIR}"

"${python_cmd}" - <<'PY'
import importlib.util
import os
import sys

repo_dreamzero = os.path.realpath(os.environ["DREAMZERO_DIR"])
spec = importlib.util.find_spec("groot")
origin = os.path.realpath(spec.origin) if spec and spec.origin else "<not found>"
print(f"[DreamZero train] groot package source: {origin}")
if not origin.startswith(repo_dreamzero + os.sep):
    print(
        f"[DreamZero train][ERROR] groot resolves outside this repo. "
        f"Expected under {repo_dreamzero}, got {origin}.",
        file=sys.stderr,
    )
    sys.exit(1)

expected_files = (
    "groot/vla/experiment/base.py",
    "groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py",
)
for relative_path in expected_files:
    path = os.path.join(repo_dreamzero, relative_path)
    print(f"[DreamZero train] expected source: {path}")
    if not os.path.isfile(path):
        print(f"[DreamZero train][ERROR] Required source file missing: {path}", file=sys.stderr)
        sys.exit(1)
PY

TRAIN_CMD=(
torchrun --nproc_per_node "${num_gpus}" --standalone groot/vla/experiment/experiment.py
    report_to="${report_to}" \
    data="${data_config}" \
    wandb_project="${WANDB_PROJECT}" \
    train_architecture="${DREAMZERO_TRAIN_ARCHITECTURE:-lora}" \
    num_frames="${num_frames}" \
    action_horizon="${action_horizon}" \
    num_views="${num_views}" \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block="${action_horizon}" \
    num_state_per_block=1 \
    seed="${seed}" \
    training_args.learning_rate="${DREAMZERO_LEARNING_RATE:-1e-5}" \
    training_args.deepspeed="${DREAMZERO_DEEPSPEED_CONFIG:-groot/vla/configs/deepspeed/zero2.json}" \
    ++action_head_cfg.config.lora_rank="${DREAMZERO_LORA_RANK:-16}" \
    ++action_head_cfg.config.lora_alpha="${DREAMZERO_LORA_ALPHA:-16}" \
    save_steps="${save_steps}" \
    training_args.warmup_ratio="${DREAMZERO_WARMUP_RATIO:-0.05}" \
    output_dir="${output_dir}" \
    per_device_train_batch_size="${batch_size}" \
    max_steps="${max_steps}" \
    weight_decay="${DREAMZERO_WEIGHT_DECAY:-1e-5}" \
    save_total_limit="${DREAMZERO_SAVE_TOTAL_LIMIT:-10}" \
    upload_checkpoints=false \
    bf16="${DREAMZERO_BF16:-true}" \
    tf32="${DREAMZERO_TF32:-true}" \
    eval_bf16="${DREAMZERO_EVAL_BF16:-true}" \
    dataloader_pin_memory=false \
    dataloader_num_workers="${dataloader_workers}" \
    image_resolution_width="${image_width}" \
    image_resolution_height="${image_height}" \
    save_lora_only="${DREAMZERO_SAVE_LORA_ONLY:-true}" \
    max_chunk_size="${max_chunk_size}" \
    frame_seqlen="${DREAMZERO_FRAME_SEQLEN:-${mhbench_frame_seqlen:-880}}" \
    save_strategy=steps \
    agibot_data_root="${dataset_path}" \
    "${VAL_ARGS[@]}" \
    dit_version="${wan_ckpt_dir}" \
    text_encoder_pretrained_path="${wan_ckpt_dir}/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="${wan_ckpt_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="${wan_ckpt_dir}/Wan2.1_VAE.pth" \
    tokenizer_path="${tokenizer_dir}" \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true \
    ++action_head_cfg.config.native_dojo_action="${native_dojo_action}"
)
# The DiT's own action/state widths, which the shared action-head yaml pins to
# AgiBot's (`diffusion_model_cfg.action_dim: 32`, and max_state_dim left at the
# constructor default of 64). The global `max_action_dim`/`max_state_dim` the
# data config sets reach the transform and the head, but NOT the DiT, so
# without these the encoder bank is built 32-wide and the first training step
# dies in `MultiEmbodimentActionEncoder`:
#   RuntimeError: Expected size for first two dimensions of batch2 tensor to
#   be: [1, 70] but got: [1, 32].
# Overridden here rather than in the yaml so agibot/robodojo/droid keep the
# exact widths their released checkpoints were trained at.
if [ "${bench_name}" = "mhbench" ]; then
    state_dim=$(bash "${UTILS_DIR}/get_state_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
    echo "[DreamZero train] mhbench DiT widths: action_dim=${action_dim} max_state_dim=${state_dim}"
    TRAIN_CMD+=(
        "++action_head_cfg.config.diffusion_model_cfg.action_dim=${action_dim}"
        "++action_head_cfg.config.diffusion_model_cfg.max_state_dim=${state_dim}"
    )
fi

if [ -n "${global_batch_size}" ]; then
    TRAIN_CMD+=("global_batch_size=${global_batch_size}")
fi

if [ "${pretrained_model_path}" != "none" ]; then
    TRAIN_CMD+=("pretrained_model_path=${pretrained_model_path}")
fi

if [ "${DREAMZERO_DRY_RUN:-0}" = "1" ]; then
    printf '[DreamZero train] Dry run command:'
    printf ' %q' "${TRAIN_CMD[@]}"
    printf '\n'
    exit 0
fi

"${TRAIN_CMD[@]}"

echo "[DreamZero train] Training finished. Checkpoints saved to ${output_dir}"
