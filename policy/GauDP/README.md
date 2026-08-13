# GauDP for MHBench (standalone XPolicyLab policy)

This directory contains the complete GauDP runtime. It does **not** import,
install, symlink, or add any external Policy-Lightning checkout to the module
search path. The needed diffusion/vision utilities are copied under
`gaudp/core/`, and the official NoPoSplat encoder/renderer/geometry code is
vendored under `gaudp/third_party/noposplat/`. See [NOTICE.md](NOTICE.md) for
source commits and licenses.

## Data contract

Images remain RGB throughout; no RGB/BGR swap is applied. The default views
are `ego_a` and `ego_b`. Set `GAUDP_USE_SCENE=1` consistently for conversion,
training, and evaluation to add `scene` as a third view.

The centralized state is real proprioception, never a copy of action:

```text
robot_a 21 = pelvis xyz+quat_xyzw (7) + left EEF (7) + right EEF (7)
robot_b 21 = pelvis xyz+quat_xyzw (7) + left EEF (7) + right EEF (7)
state       = robot_a + robot_b = 42D
```

Each raw MHBench robot action has 32 values. Its 14 hand joints are compressed
to `[left index, left middle, right index, right middle]`, giving:

```text
robot 22 = left EEF pose (7) + right EEF pose (7) + hands (4)
           + base velocity (3) + height (1)
action   = robot_a + robot_b = 44D
```

State and action min/max statistics are fitted and stored independently. At
evaluation, `mhbench_state` arrives in XPolicyLab's `wxyz` order and is
explicitly converted to training's `xyzw`. The 44D output is split into two
22D dictionaries under `mhbench_raw_action`; the MHBench environment expands
the four hand signals back to its 64D simulator action. Standard XPolicyLab EE
keys are also populated from robot_a for protocol compatibility.

## Install

Like LatentToM, GauDP installs into the currently active policy environment.
The default versions follow official NoPoSplat: Python 3.10/3.11, PyTorch
2.1.2, torchvision 0.16.2, and the CUDA 11.8 wheel.

```bash
conda create -n gaudp python=3.10 -y
conda activate gaudp
cd baselines/XPolicyLab/policy/GauDP
bash install.sh
```

`install.sh` installs pinned Python dependencies from `requirements.txt`
(including W&B),
XPolicyLab itself in editable mode, the vendored cuRoPE extension, and
NoPoSplat's CUDA Gaussian rasterizer. It verifies the resulting imports at the
end and never installs a local Policy-Lightning project. A system CUDA toolkit
with `nvcc` matching the PyTorch CUDA **major** version is required for the two
extensions; a driver alone is insufficient.

For a CUDA 12 host, override the wheel set with matching versions/index, for
example:

```bash
GAUDP_TORCH_VERSION=2.4.1 \
GAUDP_TORCHVISION_VERSION=0.19.1 \
GAUDP_TORCHAUDIO_VERSION=2.4.1 \
GAUDP_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
bash install.sh
```

Set `GAUDP_SKIP_TORCH_INSTALL=1` to retain an already compatible PyTorch.
`GAUDP_SKIP_CUDA_EXTENSIONS=1 bash install.sh` is available only for
data/schema tooling; Gaussian training and normal policy execution still need
the extensions. If multiple CUDA toolkits are installed, select one with
`CUDA_HOME=/path/to/cuda` or `GAUDP_NVCC=/path/to/nvcc`. Limit build
parallelism with `MAX_JOBS` (default `4`). Re-run `python verify_install.py`
at any time to audit the environment.

Download the public NoPoSplat checkpoint described by the upstream project and
export its path:

```bash
export GAUDP_NOPOSPLAT_CKPT=/absolute/path/to/noposplat.ckpt
```

Official checkpoint payloads with `state_dict` keys prefixed by `encoder.` and
GauDP's own `encoder_state` checkpoints are both accepted.

## 1. Convert MHBench demonstrations

`MHBENCH_DATASET_PATH` may name one raw `.hdf5`, a shard directory, or an
MHBench dataset root understood by `scripts/_dataset.py`.

```bash
export MHBENCH_DATASET_PATH=/path/to/raw/mhbench/data
bash process_data.sh cocarry experiment cocarry ee 100
```

This writes `data/cocarry-experiment-cocarry-ee.hdf5`, including 42D state,
44D action, RGB, depth, normalized-ready intrinsics, camera poses, and episode
boundaries. The optional fifth argument limits demonstration count.

## 2. Fine-tune Gaussian reconstruction

The six positional arguments are `<bench> <ckpt> <env_cfg> <action_type>
<seed> <gpu>`. Any remaining arguments are forwarded to
`train_gaussian.py`.

```bash
# Use GPU 0, seed 0, and the default Gaussian hyperparameters.
bash train_gaussian.sh cocarry experiment cocarry ee 0 0
```

This optimizes the locally vendored NoPoSplat encoder using RGB MSE plus masked
valid-depth L1. Depth, intrinsics, and camera poses are consumed only in this
stage. The defaults are 30 epochs, batch size 1, learning rate `1e-5`, depth
weight `0.1`, and four data workers. A complete override example is:

```bash
bash train_gaussian.sh cocarry experiment cocarry ee 0 0 \
  --epochs 50 \
  --batch-size 1 \
  --num-workers 8 \
  --lr 1e-5 \
  --depth-weight 0.1 \
  --wandb-run-name cocarry-gaussian-seed0 \
  --wandb-tags gaussian,cocarry,seed-0
```

Run a single train and validation batch before a full experiment to validate
the data, checkpoint, CUDA rasterizer, and logger setup:

```bash
bash train_gaussian.sh cocarry experiment cocarry ee 0 0 --debug
```

Outputs are:

```text
checkpoints/cocarry-experiment-cocarry-ee-0/gaussian/best.ckpt
checkpoints/cocarry-experiment-cocarry-ee-0/gaussian/last.ckpt
checkpoints/cocarry-experiment-cocarry-ee-0/gaussian/metrics.jsonl
checkpoints/cocarry-experiment-cocarry-ee-0/gaussian/wandb/
```

`best.ckpt` minimizes `val/loss`. Each JSONL/W&B record contains `lr`,
`train/loss`, `train/rgb_loss`, `train/depth_loss`, `train/psnr`, and the
corresponding `val/*` metrics. PSNR should increase while the other validation
metrics decrease; compare them against the unfine-tuned checkpoint on a fixed
validation split rather than relying on training loss alone.

## 3. Train the policy

```bash
# Uses the matching Gaussian best.ckpt, GPU 0, and seed 0.
bash train.sh cocarry experiment cocarry ee 0 0
```

The matching Gaussian `best.ckpt` is mandatory. Its reconstruction encoder is
kept in `eval()` with every parameter `requires_grad=False`; only the
Gaussian-image fusion CNN, shared ResNet observation encoder, and centralized
DDPM are trained. Defaults are horizon 8, 3 observation steps, 6 returned
execution steps, and 100 diffusion steps. Outputs are under the sibling
`policy/{best,last}.ckpt`. The policy defaults are 300 epochs, batch size 8,
learning rate `1e-4`, horizon 8, three observation steps, six action steps,
and 100 DDPM inference steps. For example:

```bash
bash train.sh cocarry experiment cocarry ee 0 0 \
  --epochs 300 \
  --batch-size 8 \
  --num-workers 8 \
  --lr 1e-4 \
  --horizon 8 \
  --obs-steps 3 \
  --action-steps 6 \
  --inference-steps 100 \
  --wandb-run-name cocarry-policy-seed0 \
  --wandb-tags policy,cocarry,seed-0
```

Use `--debug` for a one-batch smoke test. Override the Gaussian artifact when
running a policy ablation or a checkpoint from another run:

```bash
GAUDP_GAUSSIAN_CKPT=/absolute/path/to/gaussian/best.ckpt \
bash train.sh cocarry experiment cocarry ee 0 0 --debug
```

The policy stage writes `policy/{best.ckpt,last.ckpt,metrics.jsonl}` and a
`policy/wandb/` directory. Its metrics are `lr`, `train/loss`, and `val/loss`;
`best.ckpt` minimizes `val/loss`.

## Logging: JSONL and Weights & Biases

Both training stages always append one JSON object per epoch to
`<stage-output>/metrics.jsonl`. W&B is enabled in `offline` mode by default,
matching the XPolicyLab LatentToM launcher: it requires no login or network
connection and stores a locally syncable run under `<stage-output>/wandb/`.

Inspect the latest local metric without W&B:

```bash
tail -n 1 checkpoints/cocarry-experiment-cocarry-ee-0/gaussian/metrics.jsonl \
  | python -m json.tool
tail -n 1 checkpoints/cocarry-experiment-cocarry-ee-0/policy/metrics.jsonl \
  | python -m json.tool
```

To upload metrics directly, log in once and select online mode. All logger
arguments after the GPU are forwarded unchanged:

```bash
wandb login

bash train_gaussian.sh cocarry experiment cocarry ee 0 0 \
  --wandb-mode online \
  --wandb-project MHBench-GauDP \
  --wandb-entity YOUR_ENTITY \
  --wandb-group cocarry-seed0

bash train.sh cocarry experiment cocarry ee 0 0 \
  --wandb-mode online \
  --wandb-project MHBench-GauDP \
  --wandb-entity YOUR_ENTITY \
  --wandb-group cocarry-seed0
```

An offline run can be uploaded later with the path printed by W&B, for example:

```bash
wandb sync checkpoints/cocarry-experiment-cocarry-ee-0/gaussian/wandb/offline-run-*
wandb sync checkpoints/cocarry-experiment-cocarry-ee-0/policy/wandb/offline-run-*
```

For JSONL-only logging, disable W&B explicitly:

```bash
bash train_gaussian.sh cocarry experiment cocarry ee 0 0 --wandb-mode disabled
bash train.sh cocarry experiment cocarry ee 0 0 --wandb-mode disabled
```

The following environment variables provide convenient defaults for both
stages; explicit command-line arguments take precedence:

```bash
export GAUDP_WANDB_MODE=offline       # online, offline, or disabled
export WANDB_PROJECT=MHBench-GauDP
export WANDB_ENTITY=YOUR_ENTITY       # optional in offline mode
```

Available W&B arguments are `--wandb-mode`, `--wandb-project`,
`--wandb-entity`, `--wandb-run-name`, `--wandb-group`, and the comma-separated
`--wandb-tags`. Reusing an output directory appends to its `metrics.jsonl`;
use a different checkpoint name or seed for a clean experiment history.

## Evaluation

Use the same ten-argument XPolicyLab launcher convention as other policies:

```bash
bash eval.sh \
  cocarry Isaac-CoCarry-G1x2-v0 experiment cocarry ee 0 0 0 \
  gaudp_env mhbench_env
```

The resolver searches
`checkpoints/<bench>-<ckpt>-<env_cfg>-<action_type>-<seed>/`. `deploy.yml` can
instead provide a path in `ckpt_name` or one of XPolicyLab's explicit path
keys. Only `action_type=ee` is supported. Both single and batched observations,
per-environment history padding, six-action chunks, and `reset()` are handled.

## Direct entry points and diagnostics

Every Python path is based on `__file__`, so these work from another directory:

```bash
cd /tmp
python /path/to/GauDP/process_data.py --help
python /path/to/GauDP/train_gaussian.py --help
python /path/to/GauDP/train_policy.py --help
```

Common failures:

- `CUDA rasterizer is unavailable`: rerun `install.sh` in the same environment
  and ensure its CUDA toolkit matches PyTorch.
- cuRoPE import/build error: verify `nvcc` is available and rebuild the vendored
  `gaudp/third_party/noposplat/model/encoder/backbone/croco/curope` package.
- view-count mismatch: use `GAUDP_USE_SCENE=1` (or omit it) consistently across
  all three phases and set `use_scene` equivalently in `deploy.yml`.
- missing `mhbench_state`: use `scripts/mhbench_xpolicylab_env.py`; GauDP
  intentionally refuses the lossy generic-state/action fallback.
- missing policy checkpoint: run the Gaussian stage first, then policy training;
  both stages must share the five-part run name and seed.
