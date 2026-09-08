# H-RDT Quick Start

This guide assumes you already followed `psi.md`.

## Train on xmove-pick

```bash
cd /path/to/Psi0/src/h_rdt
source .venv/bin/activate
export DINO_SIGLIP_DIR="$PWD/bak/dino-siglip"
export PRETRAINED_BACKBONE_PATH="$(find "$PWD/checkpoints/pretrain-0618" -path '*/pytorch_model.bin' | head -n1)"
export LEROBOT_DATA_ROOT="$(git rev-parse --show-toplevel)/third_party/SIMPLE/data/G1WholebodyXMoveAndPickMP-v0"
export LEROBOT_VIDEO_BACKEND=pyav
export WANDB_MODE=offline
export USE_PRECOMP_LANG_EMBED=0
TRAIN_BATCH_SIZE=1 \
SAMPLE_BATCH_SIZE=1 \
MAX_TRAIN_STEPS=1000 \
CHECKPOINTING_PERIOD=1000 \
SAMPLE_PERIOD=-1 \
DATALOADER_NUM_WORKERS=0 \
bash finetune_lerobot.sh "$LEROBOT_DATA_ROOT" ./checkpoints/xmovepick_hrdt_1k
```

## Run open-loop eval

Start the server:

```bash
cd /path/to/Psi0/src/h_rdt
source .venv/bin/activate
export DINO_SIGLIP_DIR="$PWD/bak/dino-siglip"
export LEROBOT_VIDEO_BACKEND=pyav
bash deploy.sh ./checkpoints/xmovepick_hrdt_1k/checkpoint-1000/model.safetensors 127.0.0.1 8010
```

In another shell:

```bash
cd /path/to/Psi0/src/h_rdt
source .venv/bin/activate
export LEROBOT_DATA_ROOT="$(git rev-parse --show-toplevel)/third_party/SIMPLE/data/G1WholebodyXMoveAndPickMP-v0"
export LEROBOT_VIDEO_BACKEND=pyav
export USE_PRECOMP_LANG_EMBED=0
export SERVER_URL=http://127.0.0.1:8010/predict
export MAX_SAMPLES=8
bash test_serve.sh
```

## Run SIMPLE eval

H-RDT serves both `/predict` and `/act`.

Start the server as above, then from the repo root:

```bash
cd /path/to/Psi0
nix develop
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
  simple/G1WholebodyXMoveAndPickMP-v0 \
  hrdt \
  train \
  --host localhost \
  --port 8010 \
  --data-format lerobot \
  --sim-mode mujoco_isaac \
  --headless \
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
export CC=$(command -v gcc)
export CXX=$(command -v g++)
export CUDAHOSTCXX=$(command -v g++)
export TORCH_EXTENSIONS_DIR=$PWD/third_party/SIMPLE/.torch_extensions_flake
export TORCH_CUDA_ARCH_LIST=8.0+PTX
export NVCC_PREPEND_FLAGS="-ccbin $(command -v g++)"
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES

for level in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 python -m simple.cli.eval \
    simple/G1WholebodyXMoveAndPickMP-v0 \
    hrdt \
    train \
    --host localhost \
    --port 8010 \
    --data-format lerobot \
    --sim-mode mujoco_isaac \
    --headless \
    --save-video \
    --eval-dir "$PWD/third_party/SIMPLE/data/evals/hrdt_dr/level-${level}" \
    --max-episode-steps 360 \
    --num-episodes 10 \
    --num-workers 1 \
    --dr-level "$level" \
    --data-dir "$(git rev-parse --show-toplevel)/third_party/SIMPLE/data/G1WholebodyXMoveAndPickMP-v0" \
    --success-criteria 0.9
done
```
