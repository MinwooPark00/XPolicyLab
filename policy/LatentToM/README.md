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
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
```

`convert_to_replay_buffer.py` reads MHBench's shared LeRobot v2.1 export (`scripts/
export_lerobot.py`'s output in the main repo, `datasets/<task>/lerobot/`) — the same export ACT,
Diffusion Policy and GR00T all train on — and writes upstream LatentToM's own on-disk format —
`data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/{replay_buffer.zarr,
videos/<episode>/<camera>.mp4}`.

The source defaults to `datasets/<task>/lerobot`, where `<task>` is `env_cfg_type` without its
underscore — `door_passage` reads `datasets/doorpassage/lerobot`, the two spellings the rest of the
bench already accepts. Set `MHBENCH_DATASET_PATH` for an export kept anywhere else.

## Training

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
```

Writes `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/checkpoints/
{arm1,arm2}_latest.ckpt` — one file per arm, kept as LatentToM's own two-checkpoint layout rather
than flattened to one file the way single-policy adapters like DP do.

## Evaluation

```bash
cd XPolicyLab/policy/LatentToM
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>
```

`eval.sh`/`setup_eval_*.sh` are `policy/demo_policy`'s. `deploy.py` binds the shared rollout loop
at `OBS_STRIDE = 1` (`utils/rollout.py`): both arms condition on an `n_obs_steps: 2` window, so the
per-step observations inside a chunk are that window and must not be skipped.

On MHBench, `baselines/scripts/eval_policy.sbatch LatentToM <task>` is the shared runner — one
server, per-episode client processes, sharding and result accounting — and
`baselines/scripts/serve/LatentToM.sh` is where this policy's part of it lives.

## Quick end-to-end check

Data conversion → training → evaluation, in order, using consistent naming (especially
`action_type`) throughout so checkpoint resolution doesn't need a manual alias:

```bash
# 0) Install (skip if already done)
conda create -n <policy_env> python=3.10 -y  # e.g. latenttom
conda activate <policy_env>
cd baselines/XPolicyLab/policy/LatentToM
bash install.sh

# 1) Convert data. Reads datasets/cocarry/lerobot -- env_cfg_type names the task directory too.
#    Set MHBENCH_DATASET_PATH to read from anywhere else. Run scripts/export_lerobot.py
#    in the main repo first if that directory doesn't exist yet.
bash process_data.sh mhbench cocarry cocarry joint
# -> data/mhbench-verify-cocarry-joint/{replay_buffer.zarr,videos/}

# 2) Train. training.debug=true is a fast plumbing check.
bash train.sh mhbench cocarry cocarry joint 0 0 training.debug=true
# -> checkpoints/mhbench-cocarry-cocarry-joint-0/checkpoints/{arm1,arm2}_latest.ckpt
#    which is the name baselines/scripts/serve/LatentToM.sh serves from.

# 3) Evaluate, no simulator (fastest wiring check)
export EVAL_ENV_TYPE=debug
bash eval.sh mhbench cocarry cocarry cocarry joint 0 0 0 <policy_env> <policy_env>

# 4) Evaluate against the real Isaac Sim env. On MHBench prefer the shared runner,
#    which owns the step limit, per-episode processes, sharding and scoring:
#      sbatch ... baselines/scripts/eval_policy.sbatch LatentToM cocarry 50 0
```

## Model contract details

- **Proprio** (`arm{1,2}_proprio`, 43D each): joint angles, 7 URDF groups (legs, waist, arms,
  hands) — `mhbench_keys.JOINT_GROUPS`'s exact key order, read via each dataset's own
  `meta/modality.json` (never hardcoded offsets).
- **Action** (`arm{1,2}_action`, 35D each): 31 joint targets (arm+hand+waist groups) + base_height(1) +
  navigate_command(3) — `mhbench_keys.action_keys(robot)`'s exact key order. Joint-space
  throughout; no rotation representation to pick.
- Camera mapping (`_encode_arm_obs` in `model.py`): `cam_left_wrist`→`camera_1` (arm1/robot_a
  private), `cam_head`→`camera_3` (shared), `cam_right_wrist`→`camera_4` (arm2/robot_b private).
- `EVAL_ENV_TYPE=sim`: joint-space end to end. `MHBenchTaskEnv` runs with
  `upper_body_mode="joint"` — `mhbench.g1.actions.joint_target_action_cfg`, Pink IK bypassed — and
  reports proprio via `get_obs()`'s `mhbench_state.<robot>.joint_pos`. `serve/LatentToM.sh` sets
  `ACTION_TYPE=joint`, which is what selects that mode. Consistent with
  `convert_to_replay_buffer.py`'s training-side layout by construction, but not yet run against a
  trained checkpoint — the first run is the one that confirms it.
