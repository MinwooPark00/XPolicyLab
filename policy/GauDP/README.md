# GauDP for MHBench

This directory contains the standalone GauDP training and evaluation code for
MHBench. GauDP supports only `action_type=ee`.

## 1. Download a dataset

Run from the MHBench repository root. Demonstrations are hosted on Hugging
Face, one repository per task:

```bash
python -m pip install -U huggingface_hub

# Example: CoCarry
hf download meat000124/cocarry \
  --repo-type dataset \
  --local-dir datasets/cocarry

test -f datasets/cocarry/meta/info.json
```

Task names are:

| Dataset name | `env_cfg` argument |
|---|---|
| `cocarry` | `cocarry` |
| `handover` | `handover` |
| `handover_easy` | `handover_easy` |
| `framehang` | `frame_hang` |
| `doorpassage` | `door_passage` |

Downloaded datasets are stored under the MHBench root's `datasets/<task>` by
default. The task directory may itself contain `meta/`, `data/`, and `videos/`,
or contain them under a `lerobot/` child. Both formats are detected.
`MHBENCH_DATASET_PATH` remains available for a dataset stored elsewhere; it may
point to either the LeRobot root or its parent containing a `lerobot/` child.

## 2. Install

```bash
conda create -n gaudp python=3.10 -y
conda activate gaudp
cd baselines/XPolicyLab/policy/GauDP
bash install.sh
```

The first Gaussian command downloads the official NoPoSplat checkpoint unless
`GAUDP_NOPOSPLAT_CKPT` points to an existing checkpoint. CUDA training requires
`nvcc`; `install.sh` installs the required Python packages and CUDA extensions.

W&B is enabled by default:

```bash
wandb login
```

Both `train_gaussian.sh` and `train.sh` accept the following W&B arguments:

| Argument | Default | Description |
|---|---|---|
| `--wandb-mode` | `online` | `online`, `offline`, or `disabled` |
| `--wandb-project` | `MHBench-GauDP` | W&B project name |
| `--wandb-entity` | account default | W&B username or team slug |
| `--wandb-run-name` | auto-generated | Name shown for this run |
| `--wandb-group` | unset | Group several runs together |
| `--wandb-tags` | empty | Comma-separated tags, such as `gaussian,cocarry,seed0` |
| `--log-every` | `50` | Log batch progress every N batches; `0` disables batch progress logs |

For example, to fine-tune the Gaussian encoder and place the run in a specific
project under your W&B account:

```bash
bash train_gaussian.sh cocarry 0 0 \
  --finetune-mode heads \
  --wandb-project GauDP-Gaussian \
  --wandb-entity <your-wandb-username-or-team> \
  --wandb-run-name cocarry-heads-seed0 \
  --wandb-group gaussian \
  --wandb-tags "gaussian,cocarry,seed0"
```

Full fine-tune

```bash
bash train_gaussian.sh cocarry 0 0 \
  --finetune-mode full \
  --batch-size 8 \
  --gradient-accumulation-steps 8 \
  --num-workers 2 \
  --wandb-run-name cocarry-full-seed0
```

The same options can be appended to policy training:

```bash
bash train.sh cocarry 0 0 \
  --wandb-project GauDP-Policy \
  --wandb-run-name cocarry-policy-seed0 \
  --wandb-tags "policy,cocarry,seed0"
```

The project and entity can also be set once through environment variables:

```bash
export WANDB_PROJECT=GauDP-Experiments
export WANDB_ENTITY=<your-wandb-username-or-team>
```

`GAUDP_WANDB_MODE=offline` and `GAUDP_WANDB_MODE=disabled` are equivalent
environment-variable shortcuts for the mode option. JSONL metrics are written
to the run output directory regardless of the W&B mode. Pass
`--wandb-mode disabled` when W&B is not needed.

## 3. Convert the dataset

Run from `baselines/XPolicyLab/policy/GauDP`:

The normal interface only needs the task name:

```bash
bash process_data.sh handover_easy
```

Convert only the first 20 episodes:

```bash
bash process_data.sh handover_easy 20
```

The fixed values `bench=mhbench`, `ckpt=<task>`, `env_cfg=<task>`, and
`action_type=ee` are filled automatically. Compact aliases such as
`doorpassage`, `framehang`, and `handovereasy` are accepted. The original
XPolicyLab interface remains available for existing jobs:

```bash
bash process_data.sh <bench> <ckpt> <env_cfg> ee [task] [max_demos]
```

The training launchers use the same task-first form. Seed and GPU default to 0:

```bash
bash train_gaussian.sh handover_easy 0 0 --finetune-mode heads
bash extract_gaussian_features.sh handover_easy 0 0
bash train.sh handover_easy 0 0 --epochs 30
```

Resolution order is:

1. `MHBENCH_DATASET_PATH`, when set;
2. `datasets/<task>` under the MHBench repository root;
3. `datasets/<task>/lerobot` for locally exported datasets.

This creates:

```text
data/mhbench-<task>-<env_cfg>-ee.hdf5
```

The optional `max_demos` argument limits the number of converted episodes. GauDP uses
the train/validation episode ranges declared in `meta/info.json`. It falls back
to a deterministic 95:5 episode split only when the source has no usable split.

## 4. Prepare Gaussian features

### Option A: fine-tune the Gaussian encoder

Head-only fine-tuning is the recommended starting point:

```bash
bash train_gaussian.sh cocarry 0 0 \
  --finetune-mode heads \
  --batch-size 1 \
  --num-workers 2
```

The last two positional arguments are `<seed> <gpu>`. Full encoder fine-tuning
uses the official NoPoSplat 1-GPU optimizer recipe: Gaussian/intrinsic heads use
the base LR, all other pretrained parameters use 0.1x LR, and AdamW is followed
by linear warm-up and step-wise cosine decay. On a 24 GiB workstation GPU, use
micro-batch 1 with 8-step gradient accumulation to retain the official 1x8
effective batch size:

```bash
bash train_gaussian.sh cocarry 0 0 \
  --finetune-mode full \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-workers 2 \
  --wandb-run-name cocarry-full-seed0
```

The full-mode defaults corresponding to the official 1x8 recipe are:

| Setting | Default |
|---|---|
| Gaussian/intrinsic head LR (`--lr`) | `1e-4` |
| Pretrained LR multiplier (`--backbone-lr-multiplier`) | `0.1` (`1e-5`) |
| AdamW betas | `(0.9, 0.95)` |
| Weight decay (`--weight-decay`) | `0.05` |
| Linear warm-up (`--warm-up-steps`) | `2000` optimizer steps |
| Cosine minimum LR (`--min-lr-ratio`) | `0.1` of head LR |
| Gradient clipping (`--gradient-clip`) | `0.5` |

The cosine scheduler advances once per optimizer update, so accumulated
micro-batches do not shorten the LR schedule. The common cosine minimum is
`1e-5` with these defaults; consequently, the head LR decays from `1e-4` to
`1e-5`, while the pretrained group remains at its conservative `1e-5` after
warm-up, matching the official implementation. To override the recipe, append
the relevant options, for example:

```bash
bash train_gaussian.sh cocarry 0 0 \
  --finetune-mode full \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lr 2e-5 \
  --backbone-lr-multiplier 0.1 \
  --weight-decay 0.01
```

Head-only mode retains its previous defaults: LR `1e-5`, weight decay `1e-6`,
AdamW betas `(0.9, 0.999)`, constant LR, and gradient clipping at `1.0`. The
official warm-up/cosine recipe is enabled by default specifically for `full`
mode. Every checkpoint now stores both optimizer and scheduler state.

Checkpoints are written to:

```text
checkpoints/mhbench-cocarry-cocarry-ee-0/gaussian/best.ckpt
checkpoints/mhbench-cocarry-cocarry-ee-0/gaussian/last.ckpt
```

To evaluate reconstruction on the GauDP validation split:

```bash
# Official pretrained NoPoSplat checkpoint
bash eval_gaussian.sh cocarry 0 0

# Fine-tuned checkpoint
bash eval_gaussian.sh cocarry 0 0 \
  checkpoints/mhbench-cocarry-cocarry-ee-0/gaussian/best.ckpt
```

### Option B: use the official encoder without fine-tuning

Skip `train_gaussian.sh`. Feature extraction automatically uses the official
NoPoSplat checkpoint when no run-local `gaussian/best.ckpt` exists.

### Extract offline features

```bash
bash extract_gaussian_features.sh cocarry 0 0 \
  --batch-size 4 \
  --num-workers 8
```

To select a Gaussian checkpoint explicitly, place it after the GPU argument:

```bash
bash extract_gaussian_features.sh cocarry 0 0 \
  /absolute/path/to/gaussian/best.ckpt \
  --batch-size 4
```

The default cache is:

```text
checkpoints/mhbench-cocarry-cocarry-ee-0/gaussian/features.hdf5
```

Use another disk when necessary:

```bash
export GAUDP_GAUSSIAN_FEATURES=/fast/local/nvme/cocarry-features.hdf5
bash extract_gaussian_features.sh cocarry 0 0
```

Use `--overwrite` to replace an existing cache or `--debug` to extract one
batch as a smoke test.

## 5. Train the policy

```bash
bash train.sh cocarry 0 0 \
  --epochs 30 \
  --batch-size 16 \
  --num-workers 8 \
  --lr 1e-4
```

The last two positional arguments are `<seed> <gpu>`. `train.sh` requires the
completed feature cache from the previous step. Outputs are:

```text
checkpoints/mhbench-cocarry-cocarry-ee-0/policy/best.ckpt
checkpoints/mhbench-cocarry-cocarry-ee-0/policy/last.ckpt
checkpoints/mhbench-cocarry-cocarry-ee-0/policy/metrics.jsonl
```

Use `best.ckpt`, which minimizes `val/loss`, for evaluation. The program default
is 300 epochs, but validation can worsen substantially before that; 30 epochs is
a practical first run. Use `--debug` for a one-batch smoke test.

When the Gaussian checkpoint or feature cache is stored elsewhere:

```bash
GAUDP_GAUSSIAN_CKPT=/absolute/path/to/gaussian/best.ckpt \
GAUDP_GAUSSIAN_FEATURES=/fast/local/nvme/cocarry-features.hdf5 \
bash train.sh cocarry 0 0 --epochs 30
```

## 6. Evaluate in MHBench

Run the shared evaluator from the MHBench repository root:

```bash
sbatch --partition=suma_rtx4090 --qos=base_qos --exclude=cs-gpu-01 \
  --export=ALL,MHBENCH_WT=$PWD,MHBENCH_SIF=<sif>,ISAAC_ASSETS=<mirror> \
  -J gaudp-eval-cocarry \
  baselines/scripts/eval_policy.sbatch GauDP cocarry 50 0
```

The final arguments are `<policy> <task> <episodes> <seed>`. Replace the Slurm
partition, container, and asset paths for the target cluster.

For a single-episode wiring check, run from the GauDP directory:

```bash
bash eval.sh \
  mhbench Isaac-CoCarry-G1x2-v0 cocarry cocarry ee 0 0 0 \
  gaudp_env mhbench_env
```

The ten arguments are:

```text
<bench> <task_name> <ckpt> <env_cfg> <action_type> <seed>
<policy_gpu> <env_gpu> <policy_conda_env> <eval_conda_env>
```

## Optional settings

Use the scene camera only when the same setting is applied to every stage:

```bash
export GAUDP_USE_SCENE=1
```

Also set `use_scene: true` in `deploy.yml` before evaluation.

Useful path overrides are:

```bash
export MHBENCH_DATASET_PATH=/absolute/path/to/lerobot-root
export GAUDP_NOPOSPLAT_CKPT=/absolute/path/to/noposplat.ckpt
export GAUDP_GAUSSIAN_CKPT=/absolute/path/to/gaussian/best.ckpt
export GAUDP_GAUSSIAN_FEATURES=/absolute/path/to/features.hdf5
```
