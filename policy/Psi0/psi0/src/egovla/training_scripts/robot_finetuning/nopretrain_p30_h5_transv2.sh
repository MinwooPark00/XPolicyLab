#!/bin/bash
n_node=1
bs=4

echo "number of nodes:" $n_node
echo "per device batch size:" $bs

export WANDB_API_KEY=`cat .wandb_api`

export WANDB_PROJECT=ego_manip_release

CHECKPOINTS_ROOT=./checkpoints

RUN_NAME=otv-fixed-set-30hz-nopretrain-transv2-100e-h5-3d20-rot5-lr2e-5-const-h5p30f1skip6-manoloss6l1-kp0-b$bs-$n_node

OUTPUT_DIR=$CHECKPOINTS_ROOT/$RUN_NAME
PRETRAINED_MODELS=./pretrained_models

torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$SLURM_PROCID \
    human_plan/vila_train/train_mem.py \
    --deepspeed VILA/scripts/zero3.json \
    --model_name_or_path $PRETRAINED_MODELS/vila-qwen2-vl-1.5b-instruct-sft-20240830191953 \
    --version qwen2 \
    --vision_tower google/siglip-so400m-patch14-384 \
    --data_mixture otv_sim_fixed_set_FIXED_SET_MIX_train \
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
    --num_train_epochs 100 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 1 \
    --gradient_accumulation_steps 1 \
    --eval_data_mixture otv_sim_fixed_set_FIXED_SET_MIX_train_sub50 \
    --evaluation_strategy "steps" \
    --eval_steps 250 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $RUN_NAME \
    --future_index 1 \
    --predict_future_step 30 \
    --max_action 1 \
    --min_action 0 \
    --add_his_obs_step 5 \
    --add_his_imgs True \
    --add_his_img_skip 6 \
    --num_action_bins 256 \
    --action_tokenizer uniform \
    --invalid_token_weight 0.1 \
    --mask_input True \
    --add_current_language_description False \
    --traj_decoder_type transformer_split_action_v2 \
    --raw_action_label True \
    --traj_action_output_dim 48 \
    --input_placeholder_diff_index True \
    --ee_loss_coeff 20.0 \
    --hand_loss_coeff 5.0 \
    --hand_loss_dim 6 \
    --ee_2d_loss_coeff 0.0 \
    --ee_rot_loss_coeff 5.0 \
    --hand_kp_loss_coeff 0.0 \
    --next_token_loss_coeff 0.0 \
    --traj_action_output_ee_2d_dim 0 \
    --traj_action_output_ee_dim 6 \
    --traj_action_output_hand_dim 30  \
    --traj_action_output_ee_rot_dim 12 \
    --ee_rot_representation rot6d \
    --correct_transformation True \
    --include_2d_label True \
    --include_rot_label True \
    --use_short_language_label True \
    --no_norm_ee_label True \
    --lazy_preprocess True \
    --tf32 True \
    --merge_hand True \
    --use_mano True \
    --sep_proprio True \
    --sep_query_token True\
    --loss_use_l1 True \
    --ee_movement_mask_idx 29 \