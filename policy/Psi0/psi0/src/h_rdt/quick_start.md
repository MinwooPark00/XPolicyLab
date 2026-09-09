# Quick Start

Use the repo-local `nix develop` shell and `uv` environment:

```bash
cd /path/to/h_rdt
nix develop
```
In the dev shell, run:
```bash
uv sync --frozen
source .venv/bin/activate
```

Download the H-RDT release weights:

```bash
cd /path/to/h_rdt
huggingface-cli download --resume-download embodiedfoundation/H-RDT --local-dir ./
```

Download T5:

```bash
huggingface-cli download --resume-download google/t5-v1_1-xxl --local-dir ./models/t5-v1_1-xxl
```

Set model paths:

```bash
export DINO_SIGLIP_DIR="$PWD/bak/dino-siglip"
export PRETRAINED_BACKBONE_PATH="$(find "$PWD/checkpoints/pretrain-0618" -path '*/pytorch_model.bin' | head -n1)"
export T5_MODEL_PATH="$PWD/models/t5-v1_1-xxl"
export LEROBOT_VIDEO_BACKEND=pyav
```

Run a short fine-tune:

```bash
bash finetune_lerobot.sh "$LEROBOT_DATA_ROOT" ./outputs/$(basename "$LEROBOT_DATA_ROOT")
```


Tested 1-step sanity command:

```bash
WANDB_MODE=offline \
TRAIN_BATCH_SIZE=1 \
SAMPLE_BATCH_SIZE=1 \
MAX_TRAIN_STEPS=1 \
CHECKPOINTING_PERIOD=1 \
SAMPLE_PERIOD=-1 \
DATALOADER_NUM_WORKERS=0 \
bash finetune_lerobot.sh "$LEROBOT_DATA_ROOT" ./outputs/quick_start_sanity
```

Tested 1-step bend-pick sanity command without T5:

```bash
export LEROBOT_DATA_ROOT=/hfm/data/simple/G1WholebodyBendPick-v0-psi0
WANDB_MODE=offline \
USE_PRECOMP_LANG_EMBED=0 \
TRAIN_BATCH_SIZE=1 \
SAMPLE_BATCH_SIZE=1 \
MAX_TRAIN_STEPS=1 \
CHECKPOINTING_PERIOD=1 \
SAMPLE_PERIOD=-1 \
DATALOADER_NUM_WORKERS=0 \
bash finetune_lerobot.sh "$LEROBOT_DATA_ROOT" ./outputs/bend_pick_sanity
```

Run fine-tune plus open-loop sampling eval:

```bash
WANDB_MODE=offline SAMPLE_PERIOD=1 NUM_SAMPLE_BATCHES=1 \
bash finetune_lerobot.sh "$LEROBOT_DATA_ROOT" ./outputs/$(basename "$LEROBOT_DATA_ROOT")_with_eval
```

The open-loop metrics are logged from `train/sample.py` as:
- `action/metrics/overall_avg_mse`
- `action/metrics/overall_avg_l2err`

Deploy a checkpoint and run separate open-loop eval, in the EgoVLA style:

```bash
cd /path/to/h_rdt
source .venv/bin/activate

bash deploy.sh ./outputs/quick_start_sanity/checkpoint-1/model.safetensors 127.0.0.1 8010
```

In another shell:

```bash
cd /path/to/h_rdt
nix develop -c bash
source .venv/bin/activate

export LEROBOT_DATA_ROOT=/path/to/lerobot_dataset
MAX_SAMPLES=8 bash test_serve.sh "$LEROBOT_DATA_ROOT" http://127.0.0.1:8010/predict
```

For bend-pick without T5 embeddings:

```bash
export LEROBOT_DATA_ROOT=/hfm/data/simple/G1WholebodyBendPick-v0-psi0
USE_PRECOMP_LANG_EMBED=0 MAX_SAMPLES=8 \
bash test_serve.sh "$LEROBOT_DATA_ROOT" http://127.0.0.1:8010/predict
```

`deploy.sh` now serves both:
- `POST /predict` for the older open-loop test script
- `POST /act` for SIMPLE evaluation

## SIMPLE bend-pick eval

Train bend-pick for 1000 steps:

```bash
cd /path/to/h_rdt
nix develop
uv sync --frozen
source .venv/bin/activate

export DINO_SIGLIP_DIR="$PWD/bak/dino-siglip"
export PRETRAINED_BACKBONE_PATH="$(find "$PWD/checkpoints/pretrain-0618" -path '*/pytorch_model.bin' | head -n1)"
export LEROBOT_VIDEO_BACKEND=pyav
export WANDB_MODE=offline
export USE_PRECOMP_LANG_EMBED=0

TRAIN_BATCH_SIZE=1 \
SAMPLE_BATCH_SIZE=1 \
MAX_TRAIN_STEPS=1000 \
CHECKPOINTING_PERIOD=1000 \
SAMPLE_PERIOD=-1 \
DATALOADER_NUM_WORKERS=0 \
bash finetune_lerobot.sh /hfm/data/simple/G1WholebodyBendPick-v0-psi0 ./outputs/bend_pick_simple_eval_1000
```

Start the SIMPLE-compatible server:

```bash
cd /path/to/h_rdt
source .venv/bin/activate

export DINO_SIGLIP_DIR="$PWD/bak/dino-siglip"
bash deploy.sh ./outputs/bend_pick_simple_eval_1000/checkpoint-1000/model.safetensors 127.0.0.1 22086
```

In another shell, run SIMPLE:

```bash
cd /path/to/Psi0
nix develop -c bash
source .venv-psi/bin/activate
export CC=$(command -v gcc)
export CXX=$(command -v g++)
export CUDAHOSTCXX=$(command -v g++)
export TORCH_EXTENSIONS_DIR=$PWD/third_party/SIMPLE/.torch_extensions_flake
export TORCH_CUDA_ARCH_LIST=8.0+PTX
export NVCC_PREPEND_FLAGS="-ccbin $(command -v g++)"
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES

python -m simple.cli.eval \
  simple/G1WholebodyBendPick-v0 \
  hrdt \
  --host=127.0.0.1 \
  --port=22086 \
  --sim-mode=mujoco_isaac \
  --headless \
  --max-episode-steps=360 \
  --data-format=lerobot \
  --data-dir=/hfm/data/simple/G1WholebodyBendPick-v0-psi0 \
  --num-episodes=1 \
  --eval-dir=third_party/SIMPLE/data/evals
```

Videos are written under:

```bash
/path/to/Psi0/third_party/SIMPLE/data/evals/hrdt/train
```
