# Quick Start

This is the shortest `nix develop` + `uv` path I verified for LeRobot G1 finetuning plus open-loop eval.

## 1. Enter the Nix shell

```bash
cd /path/to/egovla
nix develop -c bash
```

The dev shell provides `uv`, `python3.11`, `gcc`, `ninja`, and `pkg-config`. By default it also sets `TMPDIR`, `UV_CACHE_DIR`, and `PIP_CACHE_DIR` under the repo directory.

## 2. Sync the uv environment

```bash
cd /path/to/egovla
uv sync --frozen
source .venv/bin/activate
```

## 3. Pick a task and common env vars

```bash
cd /path/to/egovla
source .venv/bin/activate

export DATA_ROOT=/path/to/lerobot
export LEROBOT_TASK_DIR=Pick_bottle_and_turn_and_pour_into_cup
export LEROBOT_VIDEO_BACKEND=pyav
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
```


Download the base checkpoint:

```bash
cd /path/to/egovla
source .venv/bin/activate
huggingface-cli download rchal97/egovla_base_vlm --repo-type model --local-dir checkpoints
```

That download currently gives you a zip file. Extract it and link it to the local path expected by `finetune.sh`:

```bash
cd /path/to/egovla/checkpoints
unzip -q vila-qwen2-vl-1.5b-instruct-sft-20240830191953.zip
ln -sfn \
  vila-qwen2-vl-1.5b-instruct-sft-20240830191953 \
  mix4data-30hz-transv2update2-fingertip-20e-hdof5-3d200-rot5-lr1e-4-h5p30f1skip6-b16-4
```

Make sure the pretrained base checkpoint exists at:

```bash
./checkpoints/mix4data-30hz-transv2update2-fingertip-20e-hdof5-3d200-rot5-lr1e-4-h5p30f1skip6-b16-4
```

## 4. Run a short finetune sanity check

```bash
NPROC_PER_NODE=1 \
PER_DEVICE_BS=1 \
GRAD_ACCUM_STEPS=1 \
MAX_STEPS=1 \
RUN_NAME=quick_start_sanity \
OUTPUT_DIR=$PWD/checkpoints/quick_start_sanity \
bash finetune.sh
```

## 5. Start the policy server

Use the produced checkpoint, for example `checkpoint-1` from the sanity run:

```bash
bash deploy.sh $PWD/checkpoints/quick_start_sanity/checkpoint-1 127.0.0.1 8010
```

## 6. Run open-loop eval in another shell

```bash
cd /path/to/egovla
source .venv/bin/activate

export DATA_ROOT=/path/to/lerobot
export LEROBOT_TASK_DIR=Pick_bottle_and_turn_and_pour_into_cup
export LEROBOT_VIDEO_BACKEND=pyav

SERVER_URL=http://127.0.0.1:8010/predict \
MAX_SAMPLES=5 \
bash test_serve.sh --episode_idx 0 --episode_mean
```

## Notes

- If your home or root filesystem is space-constrained, you can override Nix cache state before `nix develop`, for example:

```bash
mkdir -p /some/large/disk/nix-home /some/large/disk/.cache
HOME=/some/large/disk/nix-home XDG_CACHE_HOME=/some/large/disk/.cache nix develop -c bash
```

- If you want online W&B, set `WANDB_API_KEY` in your shell before training and remove `WANDB_MODE=offline`.
- `deploy.sh` keeps running until you stop it.
- For a real run, remove `MAX_STEPS=1` and set the batch/GPU configuration you want.
