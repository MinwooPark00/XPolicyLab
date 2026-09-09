# InternVLA-M1
## Env Setup
Install the env 
```bash
cd src/InternVLA-M1; uv sync --python 3.10
```

## On Real Data:
1. training
```bash
cd src/InternVLA-M1
bash scripts/train_internvla.sh # will generate stats_gr00t.json in $TASK/meta/ folder
```
2. serving a checkpoint
```bash
cd src/InternVLA-M1
bash scripts/deploy_internvla.sh
```

## On Sim Data:

1. download sim data
```bash
huggingface-cli download --repo-type dataset USC-PSI-Lab/psi-data --include simple/G1WholebodyBendPick-v0-psi0.zip --local-dir $DATA_DIR
```

2. train on sim data
```bash
cd src/InternVLA-M1
bash scripts/train_internvla.sh # will generate stats_gr00t.json in $TASK/meta/ folder
```

3. serving a checkpoint
```bash
cd src/InternVLA-M1
bash scripts/deploy_internvla_sim.sh
```

4. launch the eval client
follow the instructions in examples/simple/README.md and replace the `policy` with `internvla_m1` to launch the eval client