# GauDP for MHBench

This directory contains the standalone GauDP training and evaluation code for
MHBench.

GauDP is **centralized and joint-space**: one network reads both robots' 86D
URDF-ordered joint state and the two ego views (`ego_a`, `ego_b`) and predicts
the pair's 70D absolute joint-target action — the same contract GR00T N1.7 and
every other MHBench baseline uses (`configs/gr00t/mhbench_keys.py`). So
`action_type=joint`, and its runs are named the way the shared evaluator looks
for them:

```text
data/mhbench-<task>-unitree_g1x2_centralized-joint.hdf5
checkpoints/mhbench-<task>-unitree_g1x2_centralized-joint-<seed>/
```

GauDP used to predict pelvis-relative wrist poses (42D state / 44D action,
`action_type=ee`). Those datasets and checkpoints are kept and are **not**
overwritten, but nothing in a `*-ee-*` policy checkpoint transfers to joint
space — `model.py` refuses one by name rather than partially loading it, and a
run has to be retrained. The Gaussian artifacts are the exception: see
[Reusing pre-joint Gaussian artifacts](#reusing-pre-joint-gaussian-artifacts).

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
| `handover` | `handover` |
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

### Vision recipe for policy training

The observation encoder's image pipeline is configurable, and every setting is
recorded in the policy checkpoint so `model.py` rebuilds the network that was
trained. Checkpoints written before these options existed carry none of them and
deserialize through the legacy defaults, unchanged.

| Argument | Default | Upstream Policy-Lightning | Notes |
|---|---|---|---|
| `--crop-shape` | `216 288` | `crop_shape: null` | Random crop while training, centre crop at eval. A deliberate departure from upstream: MHBench's own DP baseline crops to 90% of the frame, and `robot_dp.yaml` records why -- without it validation loss bottoms early and then climbs. GauDP's first MHBench run showed exactly that curve (best at epoch 68, ~2.7x worse by 999). Pass `none` to match upstream. |
| `--image-norm` | `imagenet` | `imagenet_norm: True` | Restores upstream. This port previously used `(x-0.5)/0.5`; pass `symmetric` for that. |
| `--group-norm-divisor` | `16` | `num_features // 16` | Restores upstream's `use_group_norm` grouping. This port previously used `min(32, num_features)`; pass `none` for that. |

To reproduce a pre-existing run exactly:

```bash
bash train.sh handover 0 0 --crop-shape none --image-norm symmetric --group-norm-divisor none
```

Two upstream settings are deliberately *not* matched, because MHBench fixes them
for every baseline: `n_obs_steps` (upstream 3, here 1, as `robot_dp.yaml` also
sets) and `n_action_steps` (upstream 8, here 6). `GaussianConvEncoder`'s
3-channel output and its terminal ReLU are upstream's design and are unchanged.

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
bash process_data.sh handover
```

Convert only the first 20 episodes:

```bash
bash process_data.sh handover 20
```

The fixed values `bench=mhbench`, `ckpt=<task>`,
`env_cfg=unitree_g1x2_centralized`, and `action_type=joint` are filled
automatically. Compact aliases such as `doorpassage`, `framehang`, and
`handovereasy` are accepted and resolve to `handover`. The original XPolicyLab interface remains
available for existing jobs:

```bash
bash process_data.sh <bench> <ckpt> <env_cfg> joint [task] [max_demos]
```

The training launchers use the same task-first form. Seed and GPU default to 0:

```bash
bash train_gaussian.sh handover 0 0 --finetune-mode heads
bash extract_gaussian_features.sh handover 0 0
bash train.sh handover 0 0 --epochs 30
```

Resolution order is:

1. `MHBENCH_DATASET_PATH`, when set;
2. `datasets/<task>` under the MHBench repository root;
3. `datasets/<task>/lerobot` for locally exported datasets.

This creates:

```text
data/mhbench-<task>-unitree_g1x2_centralized-joint.hdf5
```

State and action are read out of the source's `meta/modality.json` by name —
the same named slices GR00T's config uses — so a column that moves in a future
export moves here too rather than being silently mislabelled. Per robot, state
is `left_leg(6) right_leg(6) waist(3) left_arm(7) left_hand(7) right_arm(7)
right_hand(7)` = 43D, and action is `left_arm(7) right_arm(7) left_hand(7)
right_hand(7) waist(3) base_height(1) navigate(3)` = 35D. The file records
`action_type`, `schema_version` and both key orders as HDF5 attributes, and
training refuses a file whose attributes do not match.

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
checkpoints/mhbench-cocarry-unitree_g1x2_centralized-joint-0/gaussian/best.ckpt
checkpoints/mhbench-cocarry-unitree_g1x2_centralized-joint-0/gaussian/last.ckpt
```

To evaluate reconstruction on the GauDP validation split:

```bash
# Official pretrained NoPoSplat checkpoint
bash eval_gaussian.sh cocarry 0 0

# Fine-tuned checkpoint
bash eval_gaussian.sh cocarry 0 0 \
  checkpoints/mhbench-cocarry-unitree_g1x2_centralized-joint-0/gaussian/best.ckpt
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
checkpoints/mhbench-cocarry-unitree_g1x2_centralized-joint-0/gaussian/features.hdf5
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
checkpoints/mhbench-cocarry-unitree_g1x2_centralized-joint-0/policy/best.ckpt
checkpoints/mhbench-cocarry-unitree_g1x2_centralized-joint-0/policy/last.ckpt
checkpoints/mhbench-cocarry-unitree_g1x2_centralized-joint-0/policy/metrics.jsonl
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

Run the shared sharded evaluator from the MHBench repository root. This is the
recommended path for a 50-episode result:

```bash
baselines/scripts/eval_launch.sh GauDP handover 50 0
```

The arguments are `<policy> <task> <episodes> <seed>`. By default the launcher
submits five shards, so the command above runs `5 x 10` independent episodes,
then submits a CPU aggregation job with an `afterany` dependency. Each shard
starts its own GauDP server and runs a fresh Isaac Sim process per episode; it
does not use a different GauDP eval implementation.

Override the shard count when needed (the episode count must divide evenly):

```bash
EVAL_SHARDS=10 baselines/scripts/eval_launch.sh GauDP handover 50 0
```

To queue evaluation only after a policy training job succeeds:

```bash
EVAL_DEPENDENCY=afterok:<training-job-id> \
  baselines/scripts/eval_launch.sh GauDP handover 50 0
```

The launcher prints the shard-array job ID, aggregation job ID and final result
path. For this example the outputs are:

```text
eval_results/handover/GauDP-centralized-seed0/results.json
eval_results/handover/GauDP-centralized-seed0/videos/
```

Videos are enabled by default. Use `EVAL_VIDEO=0` to disable them, and inspect
an active array with `squeue -j <array-job-id>`.

For a manually configured, unsharded Slurm job, invoke the underlying runner
directly:

```bash
sbatch --partition=suma_rtx4090 --qos=base_qos --exclude=cs-gpu-01 \
  --export=ALL,MHBENCH_WT=$PWD,MHBENCH_SIF=<sif>,ISAAC_ASSETS=<mirror> \
  -J gaudp-eval-cocarry \
  baselines/scripts/eval_policy.sbatch GauDP cocarry 50 0
```

Replace the Slurm partition, container, and asset paths for the target cluster.

For a single-episode wiring check, run from the GauDP directory:

```bash
bash eval.sh \
  mhbench Isaac-CoCarry-G1x2-v0 cocarry unitree_g1x2_centralized joint 0 0 0 \
  gaudp_env mhbench_env
```

The ten arguments are:

```text
<bench> <task_name> <ckpt> <env_cfg> <action_type> <seed>
<policy_gpu> <env_gpu> <policy_conda_env> <eval_conda_env>
```

## Reusing pre-joint Gaussian artifacts

The NoPoSplat encoder and its offline feature cache are functions of the RGB
frames and the camera geometry alone — neither reads a state or an action — so
the move to the joint contract does not invalidate either. Both are expensive
(a ~26 h fine-tune, and ~94 GB of cache), so the launchers look for them in
this order and reuse whatever they find:

1. `GAUDP_GAUSSIAN_CKPT` / `GAUDP_GAUSSIAN_FEATURES`, when set;
2. this run: `checkpoints/mhbench-<task>-unitree_g1x2_centralized-joint-<seed>/gaussian/`;
3. the pre-joint runs: `checkpoints/mhbench-<task>-<task>-ee-<seed>/gaussian/`
   and `checkpoints/<task>-experiment-<task>-ee-<seed>/gaussian/`.

`train_gaussian.sh` exits without fine-tuning when it finds one
(`GAUDP_FORCE_GAUSSIAN=1` fine-tunes anyway), and
`extract_gaussian_features.sh` reuses a complete cache in place rather than
writing a second copy.

Reuse is *checked*, not assumed. `extract_gaussian_features.py` re-validates
the frame count, camera order and encoder checkpoint before it calls a cache
complete, and `gaudp/dataset.py` additionally proves that the cache's recorded
source dataset and the joint dataset came from the **same LeRobot export**,
with the same episode boundaries and the same cameras. A cache from a different
export, a different episode selection, or a different number of views fails
loudly at the start of policy training instead of training on frames that do
not line up with the actions.

Changing the camera count (`GAUDP_USE_SCENE=1`) is the one case that does
require re-extraction, and a Gaussian checkpoint built for that many views.

The trainable parts — the observation encoder, the Gaussian fusion CNN and the
diffusion U-Net — do change shape with the contract, so the policy itself is
trained from scratch.

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
