## Diffusion Policy (DP)


### Set-up Envrionment
```
uv venv .venv-dp --python 3.10
source .venv-dp/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --group serve --group viz --active --frozen
VIRTUAL_ENV=.venv-dp uv pip install -e .
VIRTUAL_ENV=.venv-dp uv pip install -r baselines/dp/requirements-dp.txt
cp src/lerobot_patch/common/datasets/lerobot_dataset.py \
  .venv-dp/lib/python3.10/site-packages/lerobot/common/datasets/lerobot_dataset.py
```


### Download Psi0 Task Data

Download the task data, for example
```
export task=G1WholebodyXMovePick-v0
hf download USC-PSI-Lab/psi-data simple/$task.zip --local-dir=$PSI_HOME/data --repo-type=dataset
unzip "$PSI_HOME/data/simple/$task.zip" -d "$PSI_HOME/data/simple"
```


### Train $DP$
Launch the training script
```
bash baselines/dp/train_dp_g1_real.sh $task  # for train in real experiments
bash baselines/dp/train_dp_g1_simple.sh $task #  for train in simple
```


### Eval $DP$
```
export RUN_DIR=xxxx
export CKPT_STEP=40000
bash baselines/dp/serve_dp_g1_real.sh $RUN_DIR $CKPT_STEP # for train in real experiments
bash baselines/dp/serve_dp_g1_simple.sh $RUN_DIR $CKPT_STEP # for train in simple
```


### Eval in SIMPLE

TODO: migrate following instructions using SIMPLE third_party

```
cd <project root of SIMPLE>
source .venv/bin/activate
```

```
export task=G1WholebodyXMovePick-v0
```

Download eval data and extract it:
```
hf download USC-PSI-Lab/psi-data \
	simple-eval/$task.zip \
	--local-dir=data/evals \
	--repo-type=dataset

unzip data/evals/simple-eval/$task.zip -d data/evals/simple-eval
```
Now start SIMPLE eval in the SIMPLE environment:

> We provide three domain randomization levels: `level-0`, `level-1`, `level-2` for each task

```
export dr=level-0
```
We use two different entrypoints for evaluating different tasks:

set entrypoint and agent to `eval_decoupled_wbc.py` and `dp_decoupled_wbc` if the evaluating task ends with `Teleop`, which means the task data is collected using teleoperation:
```
export entry=eval_decoupled_wbc.py
export agent=dp_decoupled_wbc
```

and set entrypoint and agent to `eval.py` and `dp_g1` if the evaluating task ends with `MP`, which means the task data is generated using CuRobo Motion planning:
```
export entry=eval.py
export entry=dp_g1
```

```
python src/simple/cli/$entry \
	simple/$task \
	$agent \
	$dr \
	--host=localhost \
	--port=22085 \
	--sim-mode=mujoco_isaac \
	--no-headless \
	--data-format=lerobot \
	--data-dir=data/evals/simple-eval/$task/$dr
```