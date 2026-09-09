# SIMPLE Quick Start

This guide assumes you already followed `psi.md`.

## Goal

Use SIMPLE as an installed Python package:

- import `simple` directly from Python
- run datagen through `simple.cli.datagen`
- run evaluation through `simple.cli.eval`

Do not add `third_party/SIMPLE/src` to `PYTHONPATH`.

## Runtime

From the repo root:

```bash
cd /path/to/Psi0
env -u LD_LIBRARY_PATH nix --extra-experimental-features "nix-command flakes" develop
source .venv-psi/bin/activate
```

The repo-level `uv sync` installs `simple` as an editable package, so the
library and CLI entrypoints should already be importable.

Verify the install:

```bash
python -c "import simple; print(simple.__version__)"
```

## Import SIMPLE as a library

Minimal import check:

```bash
python - <<'PY'
import simple
from simple.tasks.registry import TaskRegistry

print(simple.__version__)
task = TaskRegistry.make(
    "franka_tabletop_grasp",
    robot_uid="franka_fr3",
    controller_uid="pd_joint_pos",
    target_object="graspnet1b:63",
)
print(task.uid)
PY
```

Programmatic evaluation uses `simple.evals.api`:

```python
from simple.evals.api import EvalConfig, EvalRunner

config = EvalConfig(
    env_id="simple/G1WholebodyXMoveAndPickMP-v0",
    policy="hrdt",
    split="train",
    host="localhost",
    port=8010,
    data_format="lerobot",
    sim_mode="mujoco_isaac",
    headless=True,
    eval_dir="third_party/SIMPLE/data/evals",
    max_episode_steps=360,
    num_episodes=10,
    data_dir="third_party/SIMPLE/data/G1WholebodyXMoveAndPickMP-v0",
    success_criteria=0.9,
    num_workers=1,
)

result = EvalRunner(config).run()
print(result.success_rate)
print(result.log_path)
```

## Run datagen

SIMPLE exposes datagen as a package CLI:

```bash
python -m simple.cli.datagen \
  simple/G1WholebodyXMoveAndPickMP-v0 \
  --sim-mode mujoco_isaac \
  --headless \
  --data-format lerobot \
  --save-dir $PWD/third_party/SIMPLE/data/datagen \
  --num-episodes 1 \
  --shard-size 1 \
  --dr-level 0
```

Useful options from `simple.cli.datagen`:

- `--scene-uid` selects the scene instance.
- `--target-object` overrides the sampled target object.
- `--eval` generates eval environment configs only, without running policy data generation.
- `--save-dir` chooses the dataset output root.

Example eval-config generation only:

```bash
python -m simple.cli.datagen \
  simple/G1WholebodyXMoveAndPickMP-v0 \
  --sim-mode mujoco_isaac \
  --headless \
  --data-format lerobot \
  --save-dir $PWD/third_party/SIMPLE/data/datagen_eval \
  --num-episodes 10 \
  --eval
```

## Run eval

SIMPLE also exposes evaluation as a package CLI:

```bash
python -m simple.cli.eval \
  simple/G1WholebodyXMoveAndPickMP-v0 \
  hrdt \
  train \
  --host localhost \
  --port 8010 \
  --data-format lerobot \
  --sim-mode mujoco_isaac \
  --headless \
  --eval-dir $PWD/third_party/SIMPLE/data/evals \
  --max-episode-steps 360 \
  --num-episodes 10 \
  --data-dir "$(git rev-parse --show-toplevel)/third_party/SIMPLE/data/G1WholebodyXMoveAndPickMP-v0" \
  --success-criteria 0.9
```

For H-RDT, use this runtime env before `python -m simple.cli.eval`:

```bash
export CC=$(command -v gcc)
export CXX=$(command -v g++)
export CUDAHOSTCXX=$(command -v g++)
export TORCH_EXTENSIONS_DIR=$PWD/third_party/SIMPLE/.torch_extensions_flake
export TORCH_CUDA_ARCH_LIST=8.0+PTX
export NVCC_PREPEND_FLAGS="-ccbin $(command -v g++)"
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
```

Example EgoVLA eval:

```bash
python -m simple.cli.eval \
  simple/G1WholebodyXMoveAndPickMP-v0 \
  egovla \
  train \
  --host localhost \
  --port 8009 \
  --data-format lerobot \
  --sim-mode mujoco_isaac \
  --headless True \
  --eval-dir $PWD/third_party/SIMPLE/data/evals \
  --max-episode-steps 360 \
  --num-episodes 10 \
  --data-dir "$(git rev-parse --show-toplevel)/third_party/SIMPLE/data/G1WholebodyXMoveAndPickMP-v0" \
  --success-criteria 0.9
```

## Outputs

Datagen outputs are typically written under:

```bash
/path/to/Psi0/third_party/SIMPLE/data/datagen
```

Eval artifacts are written under:

```bash
/path/to/Psi0/third_party/SIMPLE/data/evals
```
