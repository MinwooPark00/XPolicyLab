## DreamZero

### Set-up Envrionment
> ℹ️ The following commands assume the variables eg., `PSI_HOME` are loaded by running `cd /path/to/psi0 && set -a; source .env; set +a `.

Set up seperate environment for DreamZero 

> ℹ️ We manage the $\Psi_0$ environment and all the baselines through `uv` and they all share the same `src/` code.  See [Environment Management](../README.md) for more details.

```
uv venv .venv-dreamzero --python 3.11
source .venv-dreamzero/bin/activate
VIRTUAL_ENV=.venv-dreamzero uv pip install -e .
VIRTUAL_ENV=.venv-dreamzero GIT_LFS_SKIP_SMUDGE=1 uv pip install -r baselines/dreamzero/requirements-dreamzero.txt
```

Download checkpoints

```bash
hf download GEAR-Dreams/DreamZero-DROID --repo-type model --local-dir $PSI_HOME/cache/checkpoints/DreamZero-DROID 
hf download GEAR-Dreams/DreamZero-AgiBot --repo-type model --local-dir $PSI_HOME/cache/checkpoints/DreamZero-AgiBot
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir $PSI_HOME/cache/checkpoints/Wan2.1-I2V-14B-480P
hf download google/umt5-xxl --local-dir $PSI_HOME/cache/checkpoints/umt5-xxl
```

Boostrap pre-downloaed `Wan2.1-I2V-14B-480P`

>  Due to dreamzero hard-coded some ckpt paths

```
python baselines/dreamzero/populate_hf_cache.py
```

Test inference
```bash
# start server
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 baselines/dreamzero/socket_test_optimized_AR.py \
	--port 5000 --enable-dit-cache --model-path $PSI_HOME/cache/checkpoints/DreamZero-DROID 

# run client
python baselines/dreamzero/test_client_AR.py --host 172.19.0.102 --port 5000
```

### Download SIMPLE training data

> SIMPLE-benchmarks offers 6 Tasks:
> `G1WholebodyXMovePickTeleop-v0`, 
> `G1WholebodyBendPickMP-v0`,
> `G1WholebodyHandoverTeleop-v0`,
> `G1WholebodyLocomotionPickBetweenTablesTeleop-v0`,
> `G1WholebodyTabletopGraspMP-v0`,
> `G1WholebodyXMoveBendPickTeleop-v0`

```bash
export task=G1WholebodyXMovePickTeleop-v0
```

```bash
hf download USC-PSI-Lab/psi-data simple/$task.zip --local-dir=$PSI_HOME/data --repo-type=dataset
unzip "$PSI_HOME/data/simple/$task.zip" -d "$PSI_HOME/data/simple"
```

### Fine-Tuning

Convert psi0 to dreamzero dataset format (for single task fine-tuning):
```
python baselines/dreamzero/convert_psi0_to_dreamzero.py \
	--input-paths $PSI_HOME/data/simple/$task \
	--output-path $PSI_HOME/data/simple/dreamzero/$task
```

You can also merge all data together
```
python baselines/dreamzero/convert_psi0_to_dreamzero.py \
	--input-paths \
	$PSI_HOME/data/simple/G1WholebodyXMovePickTeleop-v0 \
	$PSI_HOME/data/simple/G1WholebodyBendPickMP-v0 \
	$PSI_HOME/data/simple/G1WholebodyHandoverTeleop-v0 \
	$PSI_HOME/data/simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
	$PSI_HOME/data/simple/G1WholebodyTabletopGraspMP-v0 \
	$PSI_HOME/data/simple/G1WholebodyXMoveBendPickTeleop-v0 \
	--output-path $PSI_HOME/data/simple/dreamzero/all-in-one
```

```bash
export task=G1WholebodyLocomotionPickBetweenTablesTeleop-v0
```

Launch `LoRA` fine-tuning
```
./baselines/dreamzero/simple_training_lora.sh $task
```

### Fine-Tuning on PSI-X G1 Neck Data

Use `data/g1_neck_0617/g1` with PSI-X-compatible state/action semantics:
- state: `45D` `state.joint_positions`
- action: `80D` = `action.hand_joints` + `action.body_token` + `action.neck`

```bash
./baselines/dreamzero/g1_neck_training_full.sh g1_neck_0617
```

Serve the PSI-X G1 neck policy over the canonical openpi/msgpack websocket
interface (same flat wire format as `g1_sonic_client.py --include-neck`). The
server validates flat keys `observation/head | hand_joints | qpos | neck`, runs one
startup warmup before listening, and replies with a `(24, 80)` chunk in
`hand14 + neck2 + token64` layout. Default port is `48014`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --standalone --nproc_per_node=1 baselines/dreamzero/socket_test_g1_neck.py \
	--port 48014 --enable-dit-cache --model-path .runs/dreamzero/g1_neck_0617
```

### Serving & Evaluation

Start the Policy Server
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 baselines/dreamzero/serve_dreamzero_g1_simple.py \
  --port 8014 --host=0.0.0.0 \
  --model-path .runs/dreamzero/G1WholebodyLocomotionPickBetweenTablesTeleop-v0-lora/checkpoint-30000 \
  --pretrained-base .cache/checkpoints/DreamZero-AgiBot
```

> `--pretrained-base` is required for LoRA version


