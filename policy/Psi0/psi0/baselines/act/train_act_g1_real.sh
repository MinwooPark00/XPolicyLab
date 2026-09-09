#!/bin/bash

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

source .venv-act/bin/activate

nprocs=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

ulimit -n 65535

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <task> [exp]"
    echo "Example: $0 Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw pick-toys"
    exit 1
fi

export task="$1"
task_words=$(echo "$task" | tr '[:upper:]' '[:lower:]' | tr '_' ' ')
default_exp=$(echo "$task_words" | awk '{if (NF>=2) print $1 "-" $2; else print $1}')
export exp=${2:-$default_exp}

echo "Task: $task"
echo "Experiment name: $exp"

args="
real_act_config \
--seed=2026 \
--exp=$exp \
--train.name=act-g1 \
--log.report-to=wandb \
--train.data_parallel=ddp \
--train.mixed_precision=bf16 \
--train.train-batch-size=32 \
--train.gradient_accumulation_steps=1 \
--train.validation_steps=500 \
--train.val_num_batches=20 \
--train.max-training-steps=42000 \
--train.learning-rate=1e-4 \
--train.max-grad-norm=1.0 \
--train.lr_scheduler_kwargs.weight_decay=1e-6 \
--train.lr_scheduler_kwargs.betas 0.95 0.999 \
--train.lr_scheduler_type=cosine \
--train.warmup-steps=1000 \
--train.warmup-ratio=None \
--train.checkpointing-steps=5000 \
--data.root_dir=real_teleop_g1/lerobot \
--data.dataset_paths=$task \
--data.transform.repack.pad-action-dim=36 \
--data.transform.repack.pad-state-dim=32 \
--data.transform.field.stat-path=meta/stats_psi0.json \
--data.transform.field.stat-action-key=action \
--data.transform.field.stat-state-key=states \
--data.transform.field.normalize-state \
--data.transform.field.action-norm-type=bounds \
--data.transform.model.img-aug \
--data.action-chunk-size=100 \
--model.chunk-size=100 \
--model.n-action-steps=100 \
--model.action-dim=36 \
--model.state-dim=32 \
--model.use-vae \
--model.kl-weight=10.0
"

torchrun --standalone --nnodes=1 --nproc-per-node=$nprocs scripts/train.py \
    $args
