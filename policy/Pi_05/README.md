# Pi_05

**Contributor:** RoboDojo Team | **Paper:** Pi0.5 technical report | **arXiv:** TBD | **Original code:** https://github.com/Physical-Intelligence/openpi

`Pi_05` adapts Physical Intelligence's π0.5 policy to XPolicyLab/RoboDojo through the uv-managed OpenPI stack. Integration scripts live at this directory level; the vendored upstream implementation lives in `openpi/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## MHBench

MHBench drives **two** Unitree G1 humanoids, and neither its observation nor its
action fits XPolicyLab's generic single-bimanual-robot schema -- so this adapter
has an `bench_name=mhbench` branch, as ACT, DP and GR00T_N17 do. Three targets
per task:

| target | `env_cfg_type` | state | action | cameras | instruction |
|---|---|---|---|---|---|
| centralized | `unitree_g1x2_centralized` | 86 (both robots' joints) | 70 | `ego_a`, `ego_b` | the pair's |
| robot_a / robot_b | `unitree_g1x2_decentralized` | 43 | 35 | that robot's ego view | that robot's |

Five tasks x three targets = fifteen runs. The column layout is derived from
MHBench's own `configs/gr00t/mhbench_keys.py`, so ACT, DP, GR00T and pi0.5 all
consume the same numbers in the same order and a difference in score is a
difference between methods.

### Data

There is nothing to convert. pi0.5 reads MHBench's LeRobot export directly and
the centralized/decentralized slicing happens in the transforms
(`openpi/src/openpi/policies/mhbench_policy.py`), the way GR00T slices it with
`meta/modality.json`. The one wrinkle is the format version: MHBench exports
**v2.1**, because that is what GR00T reads, and openpi pins `lerobot==0.4.4`,
which rejects v2.1 outright. So `process_data.sh` builds a **v3.0 view** --
generated metadata beside symlinks to the original parquet and mp4 files:

```bash
export HF_LEROBOT_HOME=/path/to/lerobot/datasets  # required
bash process_data.sh mhbench <task> unitree_g1x2_centralized joint
```

It costs a few hundred kilobytes and a couple of seconds per task, never writes
into the source, and carries over only the two ego cameras -- `LeRobotDataset`
decodes *every* video key on every sample, and the export also holds a room
camera and two depth streams that no pi0.5 target reads.

`process_data.sh` also computes normalization statistics, which training
requires. `baselines/scripts/norm_stats_pi05.sbatch` does all fifteen on CPU.

### Training

```bash
# centralized
bash train.sh mhbench <task> unitree_g1x2_centralized joint <seed> <gpu_id>
# decentralized, once per robot
bash train.sh mhbench <task>_robot_a unitree_g1x2_decentralized joint <seed> <gpu_id>
```

The TrainConfig is **derived** from the task and the target
(`pi05_mhbench_<task>_<centralized|robot_a|robot_b>`), not taken from
`OPENPI_TRAIN_CONFIG_NAME`, so a run cannot be trained under one target's config
and served under another's -- which would silently apply the wrong action width,
instruction and normalization. For the same reason `--data.repo-id` is not
passed on this path: it also sets `asset_id`, and the norm stats would stop
being found.

LoRA (`gemma_2b_lora` + `gemma_300m_lora`), at the GR00T_N17 baseline's budget:
**batch 32, 20 000 steps, checkpoint every 2 000** -- the same numbers
`train_groot_*.sbatch` exports, so the two baselines differ in method and not in
how much training they got. That batch does *not* fit the 24 GB card the
`pi05_rby1` precedent ran batch 16 on, so these want a 48 GB one
(`asus_6000ada`, `suma_a6000`, `gigabyte_a6000`). On SLURM use
`baselines/scripts/train_pi05_{centralized,decentralized}.sbatch`.

### Evaluation

```bash
sbatch --partition=suma_rtx4090 --qos=base_qos --exclude=cs-gpu-01 \
  --export=ALL,MHBENCH_WT=$PWD,MHBENCH_SIF=<sif>,ISAAC_ASSETS=<mirror>,MHBENCH_MODE=centralized \
  -J pi05-eval-<task> baselines/scripts/eval_pi05.sbatch <task> 50 0
```

`baselines/scripts/eval_pi05.sbatch` is a fork of `eval_groot.sbatch` that keeps
the episode protocol, step limit and aggregation byte-for-byte -- those are what
make two baselines comparable -- and changes only the serving side. One server
holds either the single centralized policy or both decentralized halves
(`MHBENCH_MODE`); `ckpt_name` is the task, and the per-robot run directories are
derived from it.

`deploy.yml`'s `train_config_name` and `repo_id` are ignored on this path.
Norm stats come from the checkpoint's own `assets/`, written there at save time.

### Things that would train fine and be wrong

Each of these is defended in code; they are listed because none of them fails
loudly.

- **`discrete_state_input=False`.** With `pi05=True` there is no `state_proj`
  (`pi0.Pi0.__init__`), so the state reaches the model only as prompt text.
  Turning the flag off drops the state from the model entirely while the run
  trains normally. Upstream's `pi05_libero` uses exactly that combination, so
  copying it as a template is the way in.
- **`max_token_len`.** pi0.5 spells the state out as digits in the prompt and
  `PaligemmaTokenizer` truncates past the limit with only a `logging.warning`.
  Measured worst case is 384 tokens for the 86-dim pair and 217 for one robot's
  43; the pi0.5 default of 200 would cut both. Hence 400 / 256.
- **Degenerate quantile statistics.** Six to twenty-one of the seventy action
  dimensions are zero in over 99% of frames -- hand and locomotion commands --
  so `q01` and `q99` land on the same value and quantile normalization scales
  their rare real samples by up to 1e6. `RunningStats._widen_degenerate_quantiles`
  rescales those dimensions to their observed range.
- **The held-out split.** openpi's loader takes every episode in the dataset;
  MHBench holds out `50:60`. `LeRobotMHBenchDataConfig` reads the split from
  `meta/info.json` and passes it through.
- **The shared instruction on a per-robot policy.** In CoCarry one robot
  side-steps right and the other left, so a decentralized policy given the
  pair's sentence is told to do both.

## Installation

```bash
cd XPolicyLab/policy/Pi_05
bash install.sh
source openpi/.venv/bin/activate  # OpenPI is uv-managed; there is no policy conda env
```

`eval.sh` arg 9 is not a conda env: pass `uv` (uses `deploy.yml` `policy_uv_env_path`) or an explicit OpenPI project path.

## Data Processing

Converts RoboDojo demonstrations into the LeRobot repo consumed by training. The optional `expert_data_num` caps episodes for data conversion only (it is not part of checkpoint naming); the optional `raw_task_dirs` is a source task directory or comma-separated task list under `data/<bench_name>/` (defaults to `ckpt_name`). `raw_task_dirs` may also be passed directly as the 5th argument to write a differently named dataset from all of a task's demos, e.g. `bash process_data.sh RoboDojo stack_bowls_ablation arx_x5 joint stack_bowls`.

```bash
cd XPolicyLab/policy/Pi_05
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]

# Example: convert stack_bowls demos for arx_x5 joint control
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: create a 50-episode ablation while reading from the original task data
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint 50 stack_bowls
```

## Training

```bash
cd XPolicyLab/policy/Pi_05
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0 (comma-separated gpu_id for multi-GPU)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name (auto-combined into that directory name), the full run-directory name, or a path to a checkpoint directory. By default training reads the LeRobot repo produced by `process_data.sh` (`<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`); override with `OPENPI_LEROBOT_REPO_ID` when reusing an existing dataset. `train.sh` sets `fsdp_devices=1` for one visible GPU and `2` for multi-GPU by default (override with `OPENPI_FSDP_DEVICES`).

## Evaluation

```bash
cd XPolicyLab/policy/Pi_05
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_uv_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 uv <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `checkpoint_num`, `result_dir`, `obs_transform_pipeline`, `policy_uv_env_path`, `train_config_name` (must match the config used by `train.sh`), `repo_id`.

MHBench adds five keys of its own, ignored on every other bench:

| Key | Notes |
|---|---|
| `mhbench_mode` | `centralized` (one policy, 70 actions) or `decentralized` (both halves in one process). |
| `model_dir` | Overrides the centralized run directory; `null` derives it from the task. |
| `model_dir_robot_a` / `model_dir_robot_b` | The same, per half. |
| `exec_horizon` | How much of the 50-step chunk to execute before re-observing; `null` runs all of it. |

On this path `train_config_name` and `repo_id` are ignored: the TrainConfig is
derived from the task and the target, and normalization statistics are read from
the checkpoint's own `assets/`.

Environment variables used by the adapter scripts:

| Variable | Notes |
|---|---|
| `HF_LEROBOT_HOME` | LeRobot dataset root, where `process_data.sh` writes the v3.0 view (required on the MHBench path). |
| `MHBENCH_DATASETS` | MHBench's `datasets/` root, the source the view points at; defaults to the checkout two levels above `baselines/`. |
| `MHBENCH_CONFIG_DIR` | Where `mhbench_keys.py` lives; defaults to `configs/gr00t` in the MHBench checkout. |
| `OPENPI_LEROBOT_REPO_ID` | Overrides the LeRobot repo id used by `train.sh`; defaults to `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`. |
| `OPENPI_FSDP_DEVICES` | Overrides the FSDP device count passed to OpenPI training. |
| `OPENPI_TRAIN_CONFIG_NAME` | Overrides the training config; defaults to `pi05_base_aloha_full_sim_arx-x5_seed_0`. |
| `OPENPI_DATA_MODE` | Data-processing mode passed to `openpi/scripts/process_data.py`; defaults to `image`. |
| `OPENPI_LOCAL_CACHE_ROOT` | Per-host local cache root for the HF datasets / JAX compilation caches; defaults to `/tmp/openpi-cache-$(hostname)`. |

`OPENPI_ROOT` and `OPENPI_SRC` are additional overrides consumed by the local scripts.
