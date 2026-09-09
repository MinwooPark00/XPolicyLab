# EgoVLA Quick Start

This guide assumes you already followed `psi.md`.

## Train on xmove-pick

```bash
cd /path/to/Psi0/src/egovla
source .venv/bin/activate
export DATA_ROOT="$(git rev-parse --show-toplevel)/third_party/SIMPLE/data"
export LEROBOT_TASK_DIR=G1WholebodyXMoveAndPickMP-v0
export LEROBOT_VIDEO_BACKEND=pyav
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
NPROC_PER_NODE=1 \
PER_DEVICE_BS=1 \
GRAD_ACCUM_STEPS=1 \
MAX_STEPS=1000 \
RUN_NAME=xmovepick_egovla_1k \
OUTPUT_DIR=$PWD/checkpoints/xmovepick_egovla_1k \
bash finetune.sh
```

## Run open-loop eval

Start the server:

```bash
cd /path/to/Psi0/src/egovla
source .venv/bin/activate
export DATA_ROOT="$(git rev-parse --show-toplevel)/third_party/SIMPLE/data"
export LEROBOT_TASK_DIR=G1WholebodyXMoveAndPickMP-v0
export LEROBOT_VIDEO_BACKEND=pyav
bash deploy.sh ./checkpoints/xmovepick_egovla_1k/checkpoint-1000 127.0.0.1 8009
```

In another shell:

```bash
cd /path/to/Psi0/src/egovla
source .venv/bin/activate
export DATA_ROOT="$(git rev-parse --show-toplevel)/third_party/SIMPLE/data"
export LEROBOT_TASK_DIR=G1WholebodyXMoveAndPickMP-v0
export LEROBOT_VIDEO_BACKEND=pyav
export SERVER_URL=http://127.0.0.1:8009/predict
export MAX_SAMPLES=8
bash test_serve.sh
```

## Run SIMPLE eval

Start the server as above, then from the repo root:

```bash
cd /path/to/Psi0
nix develop
source .venv-psi/bin/activate
python -m simple.cli.eval \
  simple/G1WholebodyXMoveAndPickMP-v0 \
  egovla \
  train \
  --host localhost \
  --port 8009 \
  --data-format lerobot \
  --sim-mode mujoco_isaac \
  --headless True \
  --eval-dir $PWD/third_party/SIMPLE/data/evals \
  --max-episode-steps 360 \
  --num-episodes 10 \
  --data-dir "$(git rev-parse --show-toplevel)/third_party/SIMPLE/data/G1WholebodyXMoveAndPickMP-v0" \
  --success-criteria 0.9
```

## Run DR SIMPLE eval

Start the server as above, then run the three domain-randomization levels:

```bash
cd /path/to/Psi0
nix develop
source .venv-psi/bin/activate

for level in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 python -m simple.cli.eval \
    simple/G1WholebodyXMoveAndPickMP-v0 \
    egovla \
    train \
    --host localhost \
    --port 8009 \
    --data-format lerobot \
    --sim-mode mujoco_isaac \
    --headless True \
    --save-video \
    --eval-dir "$PWD/third_party/SIMPLE/data/evals/egovla_dr/level-${level}" \
    --max-episode-steps 360 \
    --num-episodes 10 \
    --num-workers 1 \
    --dr-level "$level" \
    --data-dir "$(git rev-parse --show-toplevel)/third_party/SIMPLE/data/G1WholebodyXMoveAndPickMP-v0" \
    --success-criteria 0.9
done
```
