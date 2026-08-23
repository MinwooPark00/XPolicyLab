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

export MHBENCH_DATASET_PATH="$(pwd)/datasets/cocarry"
test -f "$MHBENCH_DATASET_PATH/meta/info.json"
```

Task names are:

| Dataset name | `env_cfg` argument |
|---|---|
| `cocarry` | `cocarry` |
| `handover` | `handover` |
| `framehang` | `frame_hang` |
| `doorpassage` | `door_passage` |

`MHBENCH_DATASET_PATH` must point to the directory containing `meta/`, `data/`,
and `videos/`, not its `data/` child. A locally exported dataset at
`datasets/<task>/lerobot` is detected automatically when the variable is not
set.

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

Pass `--wandb-mode disabled` to either training command if W&B is not needed.

## 3. Convert the dataset

Run from `baselines/XPolicyLab/policy/GauDP`:

```bash
bash process_data.sh <bench> <ckpt> <env_cfg> ee [max_demos]
```

CoCarry example using every downloaded episode:

```bash
bash process_data.sh mhbench cocarry cocarry ee
```

This creates:

```text
data/mhbench-cocarry-cocarry-ee.hdf5
```

The optional fifth argument limits the number of converted episodes. GauDP uses
the train/validation episode ranges declared in `meta/info.json`. It falls back
to a deterministic 95:5 episode split only when the source has no usable split.

## 4. Prepare Gaussian features

### Option A: fine-tune the Gaussian encoder

Head-only fine-tuning is the recommended starting point:

```bash
bash train_gaussian.sh mhbench cocarry cocarry ee 0 0 \
  --finetune-mode heads \
  --batch-size 1 \
  --num-workers 2
```

The last two positional arguments are `<seed> <gpu>`. Full encoder fine-tuning
uses considerably more VRAM:

```bash
bash train_gaussian.sh mhbench cocarry cocarry ee 0 0 \
  --finetune-mode full
```

Checkpoints are written to:

```text
checkpoints/mhbench-cocarry-cocarry-ee-0/gaussian/best.ckpt
checkpoints/mhbench-cocarry-cocarry-ee-0/gaussian/last.ckpt
```

To evaluate reconstruction on the GauDP validation split:

```bash
# Official pretrained NoPoSplat checkpoint
bash eval_gaussian.sh mhbench cocarry cocarry ee 0 0

# Fine-tuned checkpoint
bash eval_gaussian.sh mhbench cocarry cocarry ee 0 0 \
  checkpoints/mhbench-cocarry-cocarry-ee-0/gaussian/best.ckpt
```

### Option B: use the official encoder without fine-tuning

Skip `train_gaussian.sh`. Feature extraction automatically uses the official
NoPoSplat checkpoint when no run-local `gaussian/best.ckpt` exists.

### Extract offline features

```bash
bash extract_gaussian_features.sh mhbench cocarry cocarry ee 0 0 \
  --batch-size 4 \
  --num-workers 8
```

To select a Gaussian checkpoint explicitly, place it after the GPU argument:

```bash
bash extract_gaussian_features.sh mhbench cocarry cocarry ee 0 0 \
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
bash extract_gaussian_features.sh mhbench cocarry cocarry ee 0 0
```

Use `--overwrite` to replace an existing cache or `--debug` to extract one
batch as a smoke test.

## 5. Train the policy

```bash
bash train.sh mhbench cocarry cocarry ee 0 0 \
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
bash train.sh mhbench cocarry cocarry ee 0 0 --epochs 30
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
