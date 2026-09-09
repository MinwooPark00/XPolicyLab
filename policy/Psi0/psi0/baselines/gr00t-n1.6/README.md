## GR00T Baselines

Use the canonical launchers in this directory instead of adding more one-off shell scripts.

Train or pretrain with a YAML preset:

```bash
python3 baselines/gr00t-n1.6/finetune_gr00t.py --preset finetune_simple
bash baselines/gr00t-n1.6/pretrain_gr00t.sh --preset pretrain_g1_ee --dry-run
```

Run SIMPLE eval through the Python API:

```bash
.venv/bin/python baselines/gr00t-n1.6/eval_simple.py --preset simple_local
.venv/bin/python baselines/gr00t-n1.6/eval_simple.py --preset simple_local --dry-run
```

Preset files live under:

- `baselines/gr00t-n1.6/presets/train`
- `baselines/gr00t-n1.6/presets/eval`
- `pretrain_gr00t.sh` for preset-based pretraining
- `sim_eval.sh` for SIMPLE eval
- `deploy_gr00t_simple.sh` for server-only deployment
