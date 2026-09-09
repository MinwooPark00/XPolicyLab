# Psi0

Ψ₀ (`physical-superintelligence-lab/Psi0`, [arXiv 2603.12263](https://arxiv.org/abs/2603.12263))
is a vision-language-action foundation model for **Unitree G1 whole-body
loco-manipulation**: a Qwen3-VL-2B backbone with a ~500M multi-modal DiT action
expert trained by flow matching, 2.5B parameters in total.

MHBench drives two Unitree G1s, so the fit is unusually close. One head camera
at 240x320 -- Ψ₀'s own post-training resolution and exactly what MHBench's
LeRobot export writes. An action holding the same physical quantities: fourteen
hand joints, fourteen arm joints, a torso orientation, a base height and a base
velocity. The adapter is therefore mostly plumbing plus one permutation.

* **Supported** `bench_name=mhbench`, `env_cfg_type=unitree_g1x2_decentralized`,
  `action_type=joint`, `mhbench_mode=decentralized`,
  `mhbench_decentralized_style=shared` (the benchmark's default) or `per_robot`.
* **Not supported** `mhbench_mode=centralized`. Ψ₀ reads one camera and answers
  with one robot's 36 dimensions; a 70-dimensional two-view policy is a
  different model, and `model.py` says so at load time rather than silently
  serving half a robot. Other benchmarks (RoboTwin, RoboDojo) are not wired up
  either: the only Ψ₀ checkpoints here are MHBench ones.

## Install

```bash
bash policy/Psi0/install.sh          # builds psi0/.venv (Python 3.11, torch 2.7.0+cu128)
```

Two departures from upstream's README, both explained in `install.sh`:

* **torch cu128, not the PyPI default (cu126).** cu128 is the only 2.7.0 build
  carrying `sm_120` kernels, so it is the one wheel that runs on every card this
  benchmark might use -- A100 (sm_80), A6000 (86), 4090 (89) and RTX PRO 6000
  (120). `PSI0_TORCH_BACKEND` overrides it.
* **flash-attn comes from a release wheel, and is not required.** PyPI ships
  only an sdist, so `pip install flash_attn` means an hour of compiling;
  `install.sh` takes the release wheel matching this interpreter, torch build and
  C++ ABI instead. Nothing breaks without it: `src/psi/models/psi0.py` already
  falls back to `sdpa`, and the training path does now too (`psi0/UPSTREAM.md`)
  -- by launching one tiny attention rather than trusting the import, because a
  wheel built before a GPU architecture existed imports cleanly and then dies at
  the first launch. `PSI0_FLASH_ATTN=0` skips it; `PSI0_FLASH_ATTN_VERSION` picks
  another.

The pretrained pieces are two directories from `USC-PSI-Lab/psi-model`, named by
`PSI0_BASE_MODEL` and `PSI0_ACTION_HEADER` in `baselines/scripts/site.env`:

| | HF path | size |
|---|---|---|
| VLM | `psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k` | 4.28 GB |
| action expert | `psi0/postpre.1by1.pad36.2601131206.ckpt.he30k` | 1.99 GB |

## Data

```bash
bash baselines/scripts/prepare_multitask_data.sh psi0
```

builds `baselines/data/psi0/multitask_psi0/{lerobot,lerobot_val}` from the
flattened all-task dataset and links them here as
`data/mhbench-multitask-unitree_g1x2_decentralized-joint{,_val}`.

Two directories rather than one with a split, because Ψ₀ separates train from
validation by *repo* (`LerobotDataConfig.val_repo_ids`) and falls back to the
training repo when none is given -- which would report a validation loss
measured on the training set. `train.sh` refuses to start without the held-out
one.

`scripts/build_psi0_lerobot.py` writes them: v2.0 with an aggregate
`meta/stats.json` (which is what Ψ₀'s LeRobot wrapper says it reads), plus the
`meta/stats_psi0.json` its configs name, MP4s symlinked, and the columns
permuted into Ψ₀'s own order.

### The permutation

`configs/psi0/psi0_layout.py` in the MHBench workspace is the single definition,
read by the dataset builder and by `model.py`. Ψ₀'s pretrained action expert has
a per-dimension input and output projection, so dimension *i* carries learned
meaning; handing it MHBench's order would give those weights the wrong joint on
every dimension.

| Ψ₀ action | from MHBench `action` |
|---|---|
| 0:14 hands (left, right) | 14:28 |
| 14:28 arms (left, right) | 0:14 |
| 28:31 torso roll, pitch, yaw | 29, 30, 28 (MHBench stores waist as yaw, roll, pitch) |
| 31 base height | 31 |
| 32:35 vx, vy, vyaw | 32:35 |
| 35 target_yaw | none -- `pad_action_dim=36` writes 0 |

| Ψ₀ state | from MHBench `observation.state` |
|---|---|
| 0:14 hands | 22:29, 36:43 |
| 14:28 arms | 15:22, 29:36 |
| 28:31 torso roll, pitch, yaw | 13, 14, 12 |
| 31 torso height | the last commanded height (0.72 in every recorded frame) |
| 32:35 | padding |

The twelve leg joints are dropped: Ψ₀'s own state carries none either, its RL
lower-body controller owns them, and `odim=36` has no room for them.

## Train

```bash
bash baselines/scripts/train_launch.sh Psi0        # the benchmark's run
```

The recipe is the paper's, unchanged except for the batch size --
`scripts/train/psi0/finetune-real-psi0.sh` upstream:

| | |
|---|---|
| what trains | the ~500M action expert. **Full fine-tuning, no LoRA**; `--model.no-tune-vlm` freezes the Qwen3-VL backbone. The paper uses no LoRA and neither does any script upstream ships. |
| optimiser | lr 1e-4, cosine, 1000 warmup steps, weight decay 1e-6, betas 0.95/0.999, grad clip 1.0, bf16 |
| budget | **global batch 32 x 40,000 steps** -- the benchmark's shared budget. The paper's stage 3 is also 40,000 steps, at global batch 128; the batch is the one number this changes. |
| chunk | 30 steps predicted and executed (0.6 s at 50 Hz), observation horizon 1 |
| dims | action 36, state 36 (`pad_*_dim=36`, `odim=36`) |
| images | 240x320 resize + center crop (both no-ops on MHBench frames), colour jitter and noise on |
| RTC | training-time real-time-chunking masks, `max_delay=8` |

Checkpoints land at
`checkpoints/mhbench-multitask-unitree_g1x2_decentralized-joint-0[-<CKPT_TAG>]/finetune/<run>/checkpoints/ckpt_{10000,20000,30000,40000}`.
The nesting is Ψ₀'s: it composes its run directory as
`<train.output_dir>/<train.name>/<run_name>`, and `run_name` carries the batch
size, the GPU count and a timestamp. `train.sh` fixes the timestamp so the
directory is the same on every launch and a requeued job resumes rather than
starting over; the serving hook and `model.py` find the run inside by its
`run_config.json`.

Knobs: `PSI0_TIER` (pro6000 | a6000 | a100), `PSI0_MICRO_BATCH` (per device;
the accumulation that makes the global batch is derived), `MAX_STEPS`,
`SAVE_STEPS`, `PSI0_LR`, `PSI0_WARMUP_STEPS`, `PSI0_EVAL_STEPS`,
`PSI0_EVAL_BATCHES`, `PSI0_RESUME`, `PSI0_TIMESTAMP`, `PSI0_EXTRA`, `CKPT_TAG`.

`PSI0_PRINT_ONLY=1 bash train.sh mhbench multitask unitree_g1x2_decentralized joint 0 0`
prints the exact `torchrun` line without launching it.

## Eval

```bash
bash baselines/scripts/eval_launch.sh Psi0 <cocarry|handover|frame_hang|door_passage>
```

Results go to `eval_results/<scene>/Psi0-decentralized-seed0/`.

Serving mirrors `psi0/src/psi/deploy/psi0_serve_simple.py`: the run directory's
own `argv.txt` and `run_config.json` rebuild the LaunchConfig, the same
resize/center-crop pair is applied to the frame, the state is padded and
normalised with the checkpoint's own bounds, and the chunk is denormalised and
trimmed to `exec_horizon`. Ten flow steps per call, as upstream's server
hardcodes.

**The instruction is lowercased before it reaches the model.**
`RealRepackTransform.__call__` ends with `data[instruction_key].lower()`, so
every sentence the checkpoint saw was lowercase; serving the capitalised wire
text would condition it on strings it never trained on.

`deploy.yml` knobs: `exec_horizon` (30), `num_inference_steps` (10),
`checkpoint_num` (`last`), `ckpt_tag`, `model_dir` / `model_dir_robot_{a,b}`,
`prompt_robot_{a,b}`.

## Known limitations

* Centralized mode and non-MHBench benchmarks raise rather than serve.
* Ψ₀'s state slot for torso height is filled with the last commanded height.
  MHBench's LeRobot export has no pelvis-height column, and every recorded frame
  commands 0.72, so the slot is constant -- exact at training time and
  reproducible at serving time, but carrying no information either.
* Inference is synchronous. The checkpoint is trained with RTC masking (which is
  the paper's recipe and makes it robust to executing a stale chunk), but the
  MHBench client blocks on each call, so the adapter uses the plain
  `predict_action` path rather than `predict_action_with_training_rtc_flow`.
* One vendored change to upstream: `scripts/train.py` honours an explicit
  `--wandb.name` instead of overwriting it (`WandbConfig.name` was otherwise a
  dead field), so every baseline's wandb run is named after its checkpoint
  directory.
