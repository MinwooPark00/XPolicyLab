# LatentToM

**Contributor:** MHBench (staged adapter, not yet an official XPolicyLab submission) | **Paper:** [LatentToM (StanfordMSL, CoRL 2025)](https://github.com/StanfordMSL/LatentToM) | **Original code:** `external/LatentToM` (this checkout's source, branch `v1`)

`LatentToM` is a decentralized diffusion policy for two-arm cooperative manipulation: one
`DiffusionSheafSplitPolicy` per arm, each with a `SheafObsEncoder` splitting observations into a
shared (third-person) embedding and a private (own-camera + own-pose) embedding. Both arms train
and checkpoint independently (`arm1_latest.ckpt` / `arm2_latest.ckpt`) and run as two independent
policy instances at inference time.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment,
`EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md) and
[AGENTS.md](../../AGENTS.md).

## Installation

```bash
conda create -n <policy_env> python=3.10 -y  # e.g. latenttom
conda activate <policy_env>
cd baselines/XPolicyLab/policy/LatentToM
bash install.sh
```

## Data Processing

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [rotation_rep]
```

`convert_to_replay_buffer.py` reads MHBench's raw per-episode trajectory HDF5 directly (a single
`.hdf5`, a shard directory, or a dataset root) and writes upstream LatentToM's own on-disk
format — `data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<rotation_rep>/
{replay_buffer.zarr, videos/<episode>/<camera>.mp4}`. `rotation_rep` defaults to `quat` (22D/arm
action); `rot6d` (26D/arm) is the other option.

The source defaults to `datasets/<env_cfg_type>` — the task directory of the same name, which is
both where `record_demos.py` writes and where `baselines/README.md`'s download step lands. Set
`MHBENCH_DATASET_PATH` for anything else (a single shard, or a dataset outside the repo).

## Training

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [rotation_rep]
```

Writes `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/checkpoints/
{arm1,arm2}_latest.ckpt` — one file per arm, kept as LatentToM's own two-checkpoint layout rather
than flattened to one file the way single-policy adapters like DP do. `rotation_rep` must match
what `process_data.sh` converted the data with.

## Evaluation

```bash
cd XPolicyLab/policy/LatentToM
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>
```

`deploy.py`/`eval.sh`/`setup_eval_*.sh` are unmodified copies of `policy/demo_policy`'s — only
`model.py` differs.

## Quick end-to-end check

Data conversion → training → evaluation, in order, using consistent naming (especially
`action_type`) throughout so checkpoint resolution doesn't need a manual alias:

```bash
# 0) Install (skip if already done)
conda create -n <policy_env> python=3.10 -y  # e.g. latenttom
conda activate <policy_env>
cd baselines/XPolicyLab/policy/LatentToM
bash install.sh

# 1) Convert data. Reads datasets/cocarry -- env_cfg_type names the task directory too.
#    Set MHBENCH_DATASET_PATH to read from anywhere else.
bash process_data.sh mhbench verify cocarry ee "" quat
# -> data/mhbench-verify-cocarry-ee-quat/{replay_buffer.zarr,videos/}

# 2) Train. training.debug=true is a fast plumbing check.
bash train.sh mhbench verify cocarry ee 0 0 quat training.debug=true
# -> checkpoints/mhbench-verify-cocarry-ee-0/checkpoints/{arm1,arm2}_latest.ckpt

# 3) Evaluate, no simulator (fastest wiring check)
export EVAL_ENV_TYPE=debug
bash eval.sh mhbench verify verify cocarry ee 0 0 0 <policy_env> <policy_env>

# 4) Evaluate against the real Isaac Sim env (heavier)
export EVAL_ENV_TYPE=sim   # or: unset EVAL_ENV_TYPE
bash eval.sh mhbench verify verify cocarry ee 0 0 0 <policy_env> <policy_env>
```

## Model contract details

- `action_type`: only `ee` is implemented — `model.py` raises `NotImplementedError` on `joint`.
- **Proprio** (`arm{1,2}_proprio`, 21D each): pelvis pose(7) + left eef pose(7) + right eef
  pose(7).
- **Action**: 22D quat-native (`rotation_rep: quat`, default) or 26D rot6d-native
  (`rotation_rep: rot6d`) per robot — `[left pos(3)+rot(4|6), right pos(3)+rot(4|6), hands(4),
  base_vel(3), height(1)]`. `deploy.yml`'s `rotation_rep` must match what the loaded checkpoint
  was trained with; `Model.__init__` checks this against the checkpoint's own `action_dim` and
  raises on mismatch rather than silently misinterpreting the vector.
- **XPolicyLab's dual-arm `ee` action dict doesn't fit MHBench's shape.** It has one
  `left_ee_pose`/`right_ee_pose` pair for a *single* two-armed robot; MHBench is *two* robots, each
  with two wrists, plus base velocity and height. `_pack_dual_arm_action` fills the standard keys
  from robot_a's own two wrists only (documented stand-in) and puts everything — both robots'
  wrists, hands, base_vel, height — under a non-standard `mhbench_raw_action` key so nothing is
  silently dropped.
- Camera mapping (`_encode_arm_obs` in `model.py`): `cam_left_wrist`→`camera_1` (arm1/robot_a
  private), `cam_head`→`camera_3` (shared), `cam_right_wrist`→`camera_4` (arm2/robot_b private).
