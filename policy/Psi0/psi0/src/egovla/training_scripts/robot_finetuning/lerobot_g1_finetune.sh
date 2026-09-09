#!/bin/bash
set -euo pipefail

n_node=1
bs=${PER_DEVICE_BS:-16}
nproc_per_node=${NPROC_PER_NODE:-8}
master_port=${MASTER_PORT:-25001}
grad_accum=${GRAD_ACCUM_STEPS:-4}
save_steps=${SAVE_STEPS:-2000}
save_total_limit=${SAVE_TOTAL_LIMIT:-2}
eval_steps=${EVAL_STEPS:-250}

echo "number of nodes:" $n_node
echo "per device batch size:" $bs
echo "gradient accumulation steps:" $grad_accum
echo "nproc_per_node:" $nproc_per_node
echo "master_port:" $master_port

export WANDB_PROJECT=ego_manip_release
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export LEROBOT_TASK_DIR="${LEROBOT_TASK_DIR:-Pick_bottle_and_turn_and_pour_into_cup}"
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"
export PYTHONPATH="$PWD/VILA:${PYTHONPATH:-}"

CHECKPOINTS_ROOT=./checkpoints
RUN_NAME=${RUN_NAME:-lerobot-g1-actionvec-100e-lr2e-5-b$bs-$n_node}
OUTPUT_DIR=${OUTPUT_DIR:-$CHECKPOINTS_ROOT/$RUN_NAME}

PRETRAINED_MODELS=./checkpoints

DATA_ROOT=${DATA_ROOT:-/hfm/data/real_teleop_g1/lerobot}
NORM_STATS_PATH="$DATA_ROOT/$LEROBOT_TASK_DIR/norm_stats.json"
if [[ -f "$NORM_STATS_PATH" ]]; then
  MIN_ACTION=$(python3 - <<PY
import json
data=json.load(open("$NORM_STATS_PATH"))
print(min(data["norm_stats"]["actions"]["min"]))
PY
)
  MAX_ACTION=$(python3 - <<PY
import json
data=json.load(open("$NORM_STATS_PATH"))
print(max(data["norm_stats"]["actions"]["max"]))
PY
)
else
  MIN_ACTION=-0.06
  MAX_ACTION=0.06
fi

EXTRA_TRAIN_ARGS=()
if [[ -n "${NUM_EPOCHS:-}" ]]; then
  EXTRA_TRAIN_ARGS+=(--num_train_epochs "$NUM_EPOCHS")
elif [[ -n "${MAX_STEPS:-}" ]]; then
  EXTRA_TRAIN_ARGS+=(--max_steps "$MAX_STEPS")
fi
if [[ -n "${LR_DROP_EPOCH:-}" ]]; then
  EXTRA_TRAIN_ARGS+=(--lr_drop_epoch "$LR_DROP_EPOCH")
fi
if [[ -n "${LR_DROP_VALUE:-}" ]]; then
  EXTRA_TRAIN_ARGS+=(--lr_drop_value "$LR_DROP_VALUE")
fi

torchrun --nnodes=$n_node --nproc_per_node=$nproc_per_node --master_port=$master_port \
    human_plan/vila_train/train_mem.py \
    --model_name_or_path $PRETRAINED_MODELS/mix4data-30hz-transv2update2-fingertip-20e-hdof5-3d200-rot5-lr1e-4-h5p30f1skip6-b16-4 \
    --version qwen2 \
    --vision_tower google/siglip-so400m-patch14-384 \
    --data_mixture lerobot_g1_train \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --tune_vision_tower True \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --group_by_modality_length False \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 1 \
    --gradient_accumulation_steps $grad_accum \
    --eval_data_mixture lerobot_g1_eval_sub20 \
    --evaluation_strategy "steps" \
    --eval_steps $eval_steps \
    --save_strategy "steps" \
    --save_steps $save_steps \
    --save_total_limit $save_total_limit \
    --learning_rate ${LEARNING_RATE:-2e-5} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $RUN_NAME \
    --future_index 0 \
    --predict_future_step 30 \
    --add_his_obs_step 0 \
    --add_his_imgs False \
    --add_his_img_skip 1 \
    --num_action_bins 256 \
    --action_tokenizer uniform \
    --min_action $MIN_ACTION \
    --max_action $MAX_ACTION \
    --invalid_token_weight 0.1 \
    --mask_input True \
    --traj_decoder_type transformer_action_vector \
    --traj_decoder "" \
    --raw_action_label True \
    --traj_action_output_dim 36 \
    --proprio_size 32 \
    --use_proprio True \
    --sep_proprio False \
    --sep_query_token False \
    --next_token_loss_coeff 0.0 \
    --loss_use_l1 False \
    --tf32 True \
    --ddp_find_unused_parameters True \
    "${EXTRA_TRAIN_ARGS[@]}"
