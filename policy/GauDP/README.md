# GauDP for MHBench (standalone XPolicyLab policy)

This directory contains the complete GauDP runtime. It does **not** import,
install, symlink, or add any external Policy-Lightning checkout to the module
search path. The needed diffusion/vision utilities are copied under
`gaudp/core/`, and the official NoPoSplat encoder/renderer/geometry code is
vendored under `gaudp/third_party/noposplat/`. See [NOTICE.md](NOTICE.md) for
source commits and licenses.

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

W&B logging is online by default for both training stages. Authenticate once
in the GauDP environment before launching training; `wandb login` prompts for
an API key from your W&B account and stores it for later runs:

```bash
wandb login
```

GauDP pins `wandb==0.22.3` because current W&B accounts issue API keys longer
than the legacy 40-character format rejected by older SDKs. If an existing
environment reports an API-key length error, rerun
`install.sh` or upgrade it with `python -m pip install --upgrade wandb==0.22.3`
before logging in again.

`install.sh` installs pinned Python dependencies from `requirements.txt`
(including W&B),
XPolicyLab itself in editable mode, the vendored cuRoPE extension, and
NoPoSplat's CUDA Gaussian rasterizer. It verifies the resulting imports at the
end and never installs a local Policy-Lightning project. A system CUDA toolkit
with `nvcc` matching the PyTorch CUDA **major** version is required for the two
extensions; a driver alone is insufficient. When run in an active conda
environment and no usable `nvcc` is found, the installer automatically installs
the CUDA toolkit matching `torch.version.cuda` from its version-specific NVIDIA
conda channel into that environment. Set `GAUDP_AUTO_INSTALL_CUDA_TOOLKIT=0` to
disable this behavior, or set `GAUDP_CUDA_TOOLKIT_CHANNEL` to override the conda
channel label.

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

`train_gaussian.sh` automatically downloads the official public NoPoSplat
checkpoint from `botaoye/NoPoSplat` on Hugging Face when it is first needed.
It selects `re10k.ckpt` for the default two-view dataset and
`re10k_3views.ckpt` for a dataset converted with `GAUDP_USE_SCENE=1`, stores it
under `weights/`, and reuses it on later runs. These checkpoints are about
2.45 GB each, so the first Gaussian training launch requires network access and
enough free disk space.

To use an existing or custom checkpoint and skip the download, export its path:

```bash
export GAUDP_NOPOSPLAT_CKPT=/absolute/path/to/noposplat.ckpt
```

The automatic source and destination can also be overridden with
`GAUDP_NOPOSPLAT_REPO`, `GAUDP_NOPOSPLAT_FILENAME`, and
`GAUDP_NOPOSPLAT_DIR`. Interrupted Hugging Face downloads are resumed by the
Hub client when the command is run again.

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
# Use GPU 0, seed 0, and the default Gaussian hyperparameters. The official
# NoPoSplat checkpoint is downloaded automatically on the first run. Full
# fine-tuning is the default and updates the complete ViT-L encoder.
bash train_gaussian.sh cocarry experiment cocarry ee 0 0 \
  --finetune-mode full
```

Full fine-tuning has about 625M trainable parameters and normally requires a
GPU with substantially more than 12 GB of VRAM. For local workstation runs,
freeze the pretrained ViT-L backbone and fine-tune only the depth and Gaussian
parameter heads (about 94M trainable parameters):

```bash
bash train_gaussian.sh cocarry experiment cocarry ee 0 0 \
  --finetune-mode heads \
  --batch-size 1 \
  --num-workers 2
```

Both modes save the complete encoder state, so their resulting
`gaussian/{best,last}.ckpt` files are consumed identically by policy training
and evaluation. The checkpoint records `finetune_mode` for reproducibility.
Both training scripts print the first batch, every 50 batches, and the final
batch by default. Each progress record includes loss, percentage, throughput,
ETA, and allocated/reserved/peak GPU memory, and Gaussian training additionally
reports RGB loss, depth loss, and PSNR to W&B. Change the cadence with
`--log-every N` (`--log-every 0` disables batch progress records).

This optimizes the locally vendored NoPoSplat encoder using RGB MSE plus masked
valid-depth L1. Depth, intrinsics, and camera poses are consumed only in this
stage. The defaults are 30 epochs, batch size 1, learning rate `1e-5`, depth
weight `0.1`, and four data workers. A complete override example is:

```bash
bash train_gaussian.sh cocarry experiment cocarry ee 0 0 \
  --finetune-mode full \
  --epochs 50 \
  --batch-size 1 \
  --num-workers 8 \
  --lr 1e-5 \
  --depth-weight 0.1 \
  --wandb-run-name cocarry-gaussian-seed0 \
  --wandb-tags gaussian,cocarry,seed-0
```

Run a single train and validation batch before a full experiment to validate
the data, checkpoint, CUDA rasterizer, and logger setup. The head-only smoke
test is suitable for a 12 GB workstation GPU; use `--finetune-mode full` here
only on a GPU with enough memory for full fine-tuning:

```bash
bash train_gaussian.sh cocarry experiment cocarry ee 0 0 \
  --finetune-mode heads \
  --batch-size 1 \
  --num-workers 2 \
  --debug
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

### Evaluate a Gaussian checkpoint without training

Evaluate the official pretrained NoPoSplat checkpoint on the same held-out
validation episodes without creating an optimizer or computing gradients. With
no seventh positional argument, the launcher selects (and downloads if needed)
`re10k.ckpt` for two views or `re10k_3views.ckpt` for more than two views:

```bash
bash eval_gaussian.sh cocarry experiment cocarry ee 0 0
```

To evaluate a fine-tuned GauDP checkpoint, pass its path as the optional seventh
positional argument:

```bash
bash eval_gaussian.sh cocarry experiment cocarry ee 0 0 \
  checkpoints/cocarry-experiment-cocarry-ee-0/gaussian/best.ckpt
```

Additional options are forwarded to `eval_gaussian.py`. For example, this runs
one validation batch as a smoke test without W&B and uses two data workers:

```bash
bash eval_gaussian.sh cocarry experiment cocarry ee 0 0 \
  --debug --wandb-mode disabled --num-workers 2
```

Evaluation defaults to online W&B logging and reports validation RGB MSE,
masked depth L1, total loss, PSNR, throughput, ETA, and GPU memory. Results are
written under `checkpoints/<run>/gaussian_eval/noposplat/` for the official
checkpoint or `gaussian_eval/<checkpoint-stem>/` for an explicitly supplied
checkpoint. This is same-view reconstruction evaluation: the camera images
used to construct the Gaussians are also the rendering targets, so these
numbers do not measure held-out novel-view synthesis.

## 3. Extract Gaussian features offline

Policy training uses Policy-Lightning's offline feature workflow: run the
fine-tuned, frozen NoPoSplat encoder once over every converted RGB frame, then
read its pixel-aligned 13-channel features from HDF5 during every policy epoch.
The default command uses the matching
`checkpoints/<run>/gaussian/best.ckpt`:

```bash
bash extract_gaussian_features.sh cocarry experiment cocarry ee 0 0
```

Pass a Gaussian encoder checkpoint as the optional seventh positional
argument to select it explicitly. For example:

```bash
bash extract_gaussian_features.sh cocarry experiment cocarry ee 0 0 \
  checkpoints/cocarry-experiment-cocarry-ee-0/gaussian/best.ckpt \
  --batch-size 4 \
  --num-workers 8
```

Alternatively, configure both the encoder and feature-cache paths with
environment variables:

```bash
export GAUDP_GAUSSIAN_CKPT=/absolute/path/to/gaussian/best.ckpt
export GAUDP_GAUSSIAN_FEATURES=/fast/local/nvme/cocarry-gaussian-features.hdf5
bash extract_gaussian_features.sh cocarry experiment cocarry ee 0 0
```

The default output is `checkpoints/<run>/gaussian/features.hdf5`. Extraction
uses FP16 storage by default, supports interruption/resume, and refuses to use
an incomplete cache for policy training. `--dtype float32` preserves features
at full precision; `--compression lzf` or `--compression gzip` trades extraction
and loading speed for disk space. For the current 23,701-frame, two-camera
dataset, the uncompressed cache is approximately 88 GiB in FP16 or 176 GiB in
FP32. Put `GAUDP_GAUSSIAN_FEATURES` on server-local NVMe when possible.

The cache records the absolute Gaussian checkpoint path. Policy training
checks that it matches the requested checkpoint, preventing accidental use of
features produced by a different encoder. To intentionally replace an existing
cache, pass `--overwrite`. A one-batch extraction test can be run with
`--debug`; rerunning without `--debug` resumes and completes that cache.

## 4. Train the policy

```bash
# Uses the matching Gaussian best.ckpt and offline features, GPU 0, and seed 0.
bash train.sh cocarry experiment cocarry ee 0 0
```

The matching Gaussian `best.ckpt` and its completed `features.hdf5` are
mandatory. NoPoSplat is not constructed or executed during policy training;
only the Gaussian-image fusion CNN, shared ResNet observation encoder, and
centralized DDPM are trained. Defaults are horizon 8, 3 observation steps, 6
returned execution steps, and 100 diffusion steps. Outputs are under the
sibling `policy/{best,last}.ckpt`. The policy defaults are 300 epochs, batch
size 8, learning rate `1e-4`, horizon 8, three observation steps, six action
steps, and 100 DDPM inference steps. For example:

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

Use `--debug` for a one-batch smoke test. Override both matching artifacts when
running a policy ablation or using files stored on server-local NVMe:

```bash
GAUDP_GAUSSIAN_CKPT=/absolute/path/to/gaussian/best.ckpt \
GAUDP_GAUSSIAN_FEATURES=/fast/local/nvme/cocarry-gaussian-features.hdf5 \
bash train.sh cocarry experiment cocarry ee 0 0 --debug
```

Offline features are a training optimization only. During MHBench evaluation,
`model.py` still loads the selected Gaussian encoder checkpoint and extracts
features online from each new RGB observation; the deployment/inference path
does not read `features.hdf5`.

The policy stage writes `policy/{best.ckpt,last.ckpt,metrics.jsonl}` and a
`policy/wandb/` directory. `best.ckpt` minimizes `val/loss`, which is the DDPM
noise-prediction MSE used to train GauDP. Policy batch and epoch records also
include:

- `diffusion/noise_cosine`, predicted/target noise RMS, sampled timestep, and
  SNR to diagnose the denoising objective;
- clipped clean-action (`x0`) MSE/MAE in normalized action space;
- separate normalized MSE for robot A, robot B, EEF poses, hands, base
  velocity, and height, matching GauDP's centralized multi-agent action;
- pre-clipping gradient norm, gradient-clipping fraction, learning rate,
  throughput, ETA, epoch time, and GPU memory.

The GauDP paper uses environment rollout success rate as its primary policy
metric and evaluates 100 episodes; offline action errors are diagnostics rather
than substitutes for success rate. Consequently, training does not label any
dataset-only proxy as success. Use the benchmark evaluation launcher to measure
task success after checkpoints are produced. The paper also reports training
time and inference FPS; this implementation records training throughput and
duration continuously, while inference FPS belongs to the separate evaluation
run.

## Logging: JSONL and Weights & Biases

Both training stages append periodic `record_type=batch` progress records and
one `record_type=epoch` summary per epoch to `<stage-output>/metrics.jsonl`.
The same records are sent to W&B. W&B is enabled in `online` mode by default,
so log in once with `wandb login` before training; no explicit
`--wandb-mode online` argument is needed.

Inspect the latest local metric without W&B:

```bash
rg '"record_type": "epoch"' \
  checkpoints/cocarry-experiment-cocarry-ee-0/gaussian/metrics.jsonl | tail -n 1 \
  | python -m json.tool
rg '"record_type": "epoch"' \
  checkpoints/cocarry-experiment-cocarry-ee-0/policy/metrics.jsonl | tail -n 1 \
  | python -m json.tool
```

All logger arguments after the GPU are forwarded unchanged. The following
commands use the default online mode:

```bash
wandb login

bash train_gaussian.sh cocarry experiment cocarry ee 0 0 \
  --wandb-project MHBench-GauDP \
  --wandb-entity YOUR_ENTITY \
  --wandb-group cocarry-seed0

bash train.sh cocarry experiment cocarry ee 0 0 \
  --wandb-project MHBench-GauDP \
  --wandb-entity YOUR_ENTITY \
  --wandb-group cocarry-seed0
```

For a machine without network access, select offline mode explicitly. Such a
run can be uploaded later with the path printed by W&B, for example:

```bash
bash train_gaussian.sh cocarry experiment cocarry ee 0 0 --wandb-mode offline
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
export GAUDP_WANDB_MODE=online        # default; offline or disabled are opt-in
export WANDB_PROJECT=MHBench-GauDP
export WANDB_ENTITY=YOUR_ENTITY       # optional; useful for a team account
```

Available W&B arguments are `--wandb-mode`, `--wandb-project`,
`--wandb-entity`, `--wandb-run-name`, `--wandb-group`, and the comma-separated
`--wandb-tags`. Reusing an output directory appends to its `metrics.jsonl`;
use a different checkpoint name or seed for a clean experiment history.

The GauDP shell launchers set `PYTHONNOUSERSITE=1` automatically. This prevents
packages under `~/.local/lib/python*/site-packages` from shadowing the pinned
packages in the active GauDP environment. In particular, `diffusers==0.27.2`
expects the environment's `huggingface-hub==0.25.2`; a newer user-site
`huggingface_hub` can otherwise fail with `cannot import name
'cached_download'`. When invoking a Python entry point directly, use:

```bash
PYTHONNOUSERSITE=1 python train_policy.py --help
```

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
- `No module named 'pkg_resources'` while building a CUDA extension: rerun the
  updated `install.sh`, which keeps `setuptools<81` for PyTorch 2.1 compatibility.
- cuRoPE import/build error: verify `nvcc` is available and rebuild the vendored
  `gaudp/third_party/noposplat/model/encoder/backbone/croco/curope` package.
- view-count mismatch: use `GAUDP_USE_SCENE=1` (or omit it) consistently across
  all three phases and set `use_scene` equivalently in `deploy.yml`.
- missing `mhbench_state`: use `scripts/mhbench_xpolicylab_env.py`; GauDP
  intentionally refuses the lossy generic-state/action fallback.
- missing policy checkpoint: run the Gaussian stage first, then policy training;
  both stages must share the five-part run name and seed.
