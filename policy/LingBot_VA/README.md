# LingBot_VA

**Contributor:** RoboDojo Team | **Paper:** LingBot-VA technical report | **arXiv:** TBD | **Original code:** See vendored `lingbot_va/`.

`LingBot_VA` adapts the LingBot-VA policy to XPolicyLab/RoboDojo; it trains on precomputed Wan2.2 VAE latents rather than raw LeRobot parquet/video. Integration scripts live at this directory level; the vendored upstream implementation lives in `lingbot_va/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/LingBot_VA
bash install.sh
conda activate <policy_env>  # e.g. lingbot_va
```

## Data Processing

`process_data.sh` takes a standard RoboDojo LeRobot **v2.1** dataset (parquet + per-camera mp4) and runs every upstream step to produce a training-ready latent dataset: it maps actions into the 30-dim layout — left/right arm EEF (7+7), left/right arm joints (7+7), left/right gripper (1+1), with missing dimensions zero-padded — adds an `action_config` segment to each `meta/episodes.jsonl` line, encodes Wan2.2 VAE video latents into a `latents/` tree (videos resized to ~256×256 and downsampled to 5–15 fps, matching `va_robotwin30_train_cfg`), and writes `empty_emb.pt` (the Wan2.2 text embedding of an empty string, used when classifier-free guidance drops language conditioning) at the dataset root. The base model supplies the VAE and text encoder, so `LINGBOT_VA_BASE_MODEL_PATH` must be set.

```bash
cd XPolicyLab/policy/LingBot_VA
export LEROBOT_DATA_ROOT=<parent_dir_of_lerobot_datasets>      # or LINGBOT_VA_SOURCE_DATASET=<full_dataset_path>
export LEROBOT_DATASET_REPO_ID=<source_dataset_folder_name>
export LINGBOT_VA_BASE_MODEL_PATH=<path_to_lingbot-va-base>

bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]

# Example: build the latent dataset (optional trailing episode count; omit or 0 for all)
bash process_data.sh RoboDojo cotrain arx_x5 joint
```

- The output (default `data/<bench>-<ckpt>-<env>-<action>/`, override with `LINGBOT_VA_DATASET_PATH`) mirrors `videos/` under `latents/`, with one latent file per camera in the config's `obs_cam_keys` (`cam_high`, `cam_left_wrist`, `cam_right_wrist` for RoboDojo); the trainer skips any segment whose latent files are missing for a camera.
- Latent files are named `episode_{index}_{start_frame}_{end_frame}.pth`, matching the `action_config` segments in `episodes.jsonl` (e.g. segment `0–450` → `episode_000000_0_450.pth`). `process_data.sh` writes one segment per episode (`start_frame=0`, `end_frame=length`, `action_text` = the episode task); for multiple sub-tasks per episode, use one entry (and one latent file per camera) per segment, and regenerate the matching latent files if you edit segments manually.
- The latent `.pth` field schema and full upstream data-prep details are in `lingbot_va/README.md` § Custom Dataset Preparation and § Post-Training LingBot-VA.

## Training

FSDP post-training through the upstream trainer. It requires a latent-prepared dataset from Data Processing (containing `latents/`, `empty_emb.pt`, and `action_config` in `meta/episodes.jsonl`) and the base model linked at `.merged_ckpt`:

```bash
cd XPolicyLab/policy/LingBot_VA
conda activate <policy_env>
export LINGBOT_VA_DATASET_PATH=<path_to_prepared_dataset>   # or set LEROBOT_DATA_ROOT + LEROBOT_DATASET_REPO_ID
export LINGBOT_VA_BASE_MODEL_PATH=<path_to_lingbot-va-base>
ln -sfn "${LINGBOT_VA_BASE_MODEL_PATH}" .merged_ckpt

bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on eight GPUs (FSDP-sharded transformer; a single GPU is typically insufficient)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0,1,2,3,4,5,6,7
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name, the full run-directory name, or a path to a checkpoint directory. Before training, ensure the base model's `transformer/config.json` has `"attn_mode": "flex"`; switch it to `"torch"` or `"flashattn"` before inference/eval (upstream README § Important: `attn_mode` Configuration).

## Evaluation

```bash
cd XPolicyLab/policy/LingBot_VA
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

`eval_batch` is `false` and must stay that way: `wan_va_server` keeps one global KV/VAE/`frame_st_id` cache, so one process can serve only one env. `get_action_batch` raises `NotImplementedError`.

## Configuration

`deploy.yml` keys to check before evaluation: `env_cfg`, `protocol`, `config_name`, `checkpoint_path`, `base_model_path`, `rollout_mode`, `result_dir`, `va_server_host`, `va_server_port`, `obs_transform_pipeline`.

Environment variables used by the adapter scripts:

| Variable | Notes |
|---|---|
| `LEROBOT_DATA_ROOT` | Parent directory that holds LeRobot dataset folders. |
| `LEROBOT_DATASET_REPO_ID` | Source dataset folder name under `LEROBOT_DATA_ROOT`; ignored when `LINGBOT_VA_DATASET_PATH` is set. |
| `LINGBOT_VA_SOURCE_DATASET` | Full path to the source LeRobot dataset; overrides the two variables above. |
| `LINGBOT_VA_DATASET_PATH` | Output path for the prepared latent dataset (default `data/<tag>`); also the dataset consumed by training. |
| `LINGBOT_VA_BASE_MODEL_PATH` | **Required:** `lingbot-va-base` weights (VAE + text encoder); also used at eval when `deploy.yml` `base_model_path` is null. |
| `LINGBOT_VA_PROCESS_GPU` | GPU id used for latent encoding (default `0`). |
| `LINGBOT_VA_TARGET_FPS` | Target sampling fps for latents (default `10`). |
| `LINGBOT_VA_VA_HOST` / `LINGBOT_VA_VA_PORT` | Point evaluation at an already-running wan_va backend instead of auto-launching one. |

## MHBench

MHBench (`bench_name=mhbench`) drives this adapter through its own runners
rather than the scripts above; what differs is collected here, and the full
recipe with its one-off setup is in the benchmark's
`baselines/scripts/README.md`, section *LingBot_VA*.

- **Action.** 35D -- 31 joint targets, the pelvis height and the base velocity
  -- not the 30D dual-arm layout. Both projections are
  reinitialised from a base prepared once by
  `baselines/scripts/prepare_lingbot_init_ckpt.py`; everything else transfers.
- **Cameras.** One ego view per agent (`observation.images.ego`) at 224x320,
  not three RoboTwin views at 256 square. Both sides of the canvas must be a
  multiple of 32 so the VAE's stride-16 grid stays even under 2x2 patching.
- **Cadence.** The recording is 50 fps and latents are sampled at 10, so a
  latent frame spans 20 actions (`LINGBOT_ACTION_PER_FRAME=20`).
- **Normalisation.** `meta/lingbot_norm_stat.json`, computed from the training
  split by `process_data.py --layout mhbench` and copied next to every
  checkpoint. The config refuses to run against the placeholder.
- **Budget.** Global batch 32 (the benchmark's) but 10,000 steps rather than
  its 40,000: a sample here is a whole episode rather than an action chunk, so
  the shared budget would be 1.28M episode-samples -- 67 days on one 3090,
  measured. 32 x 10,000 is upstream's own post-training budget.
- **Arm.** LoRA r32 on every block projection with the two action projections
  trained in full (`LINGBOT_LORA_RANK`, 0 for upstream's full fine-tune). The
  adapters are folded into the base weights on save, so a checkpoint is keyed
  exactly like a full fine-tune and the server needs no LoRA support.
- **Two agents, one policy.** `wan_va_server` keeps one rollout *session* per
  agent -- its own KV cache, VAE streaming cache, video history and
  instruction -- selected by a `session` field in the request. A client that
  sends none gets the previous single-rollout behaviour unchanged.
- **Resuming.** `save_checkpoint` also writes `training_state.pt` (optimizer
  moments, step counter, adapters), which upstream left commented out; without
  it a requeued job restarts at step 0.

## Notes

- **Latent pipeline is mandatory:** do not point training at a plain LeRobot dataset without `latents/` and `empty_emb.pt`; run Data Processing first.
- `eval.sh` auto-launches the wan_va backend (which holds the real weights). It requires a base model path: set `LINGBOT_VA_BASE_MODEL_PATH` or `base_model_path` in `deploy.yml`.
