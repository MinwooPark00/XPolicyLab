#!/bin/bash

# source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

export TORCH_HOME=$TORCH_HOME # e.g. /hfm/user/torch_cache
export HF_HOME=$HF_HOME # e.g. /hfm/user/torch_cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source .venv/bin/activate

echo "Training with $nprocs GPUs, which is/are $CUDA_VISIBLE_DEVICES"

python InternVLA/deploy/internvla_serve_sim.py \
    --host 0.0.0.0 \
    --port 22085 \
    --checkpoint-path $CHECKPOINT_PATH # e.g. .../***.pt