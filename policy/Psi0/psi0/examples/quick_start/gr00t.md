# GR00T Quick Start

This guide assumes you already followed `psi.md`.

## Train on xmove-pick

From the repo root:

```bash
cd /path/to/Psi0
source .venv-psi/bin/activate

DATASET_PATH="$(git rev-parse --show-toplevel)/third_party/SIMPLE/data/G1WholebodyXMoveAndPickMP-v0" \
PRETRAINED_MODEL_PATH=nvidia/GR00T-N1.6-3B \
HF_HOME="$(git rev-parse --show-toplevel)/.cache/huggingface" \
HUGGINGFACE_HUB_CACHE="$(git rev-parse --show-toplevel)/.cache/huggingface/hub" \
OUTPUT_DIR=./checkpoints/gr00t_n16_xmovepick \
CUDA_VISIBLE_DEVICES=0,1 \
MASTER_PORT=29501 \
bash baselines/gr00t-n1.6/train_gr00t_simple.sh
```

## Run SIMPLE eval

The canonical launcher is:

```bash
cd /path/to/Psi0
source .venv-psi/bin/activate

MODEL_PATH=./checkpoints/gr00t_n16_xmovepick/checkpoint-50000 \
RUN_DATA_DIR="$(git rev-parse --show-toplevel)/third_party/SIMPLE/data/G1WholebodyXMoveAndPickMP-v0" \
NUM_EPISODES=10 \
NUM_WORKERS=1 \
bash baselines/gr00t-n1.6/simple.sh
```

For a dry run:

```bash
cd /path/to/Psi0
source .venv-psi/bin/activate
python baselines/gr00t-n1.6/eval_simple.py --preset simple_local --dry-run
```
