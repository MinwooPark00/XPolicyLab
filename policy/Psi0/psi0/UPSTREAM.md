# Vendored Psi0

    https://github.com/physical-superintelligence-lab/Psi0
    a7db4fbb9e3ed55bb46978e890224fa3ff723ad9  (fetched 2026-09-08, --depth 1)

Vendored as plain tracked files, not a submodule and not a nested checkout --
the same shape as `policy/Pi_05/openpi` and `policy/GR00T_N17/gr00t_n17`. The
clone's `.git` was removed so git sees files rather than a gitlink.

## Local changes

`scripts/train.py` -- `wandb_config["name"]` honours an explicit
`--wandb.name` instead of overwriting it unconditionally.
`WandbConfig.name` was otherwise a dead field: it is read out of `cfg.wandb`
and then replaced by the composed `<exp>.b<batch>.gpus<n>.<timestamp>`, so no
run could be named anything else. MHBench names every policy's wandb run after
its checkpoint directory so the baselines line up in one view;
`trainer.run_name` stays the fallback.

`src/psi/trainers/finetune.py` -- the VLM's `attn_implementation` is chosen
rather than hard-coded to `flash_attention_2`. As written, flash-attn was a
build-time requirement of *training* while `models/psi0.py`, serving the same
weights, already fell back to `sdpa`. The choice is made by launching one tiny
flash attention: `is_flash_attn_2_available()` answers only whether the package
imports, and 2.7.4.post1 -- the pinned version -- predates Blackwell, so on an
sm_120 card it would import and then fail at the first kernel launch.
`PSI0_ATTN_IMPLEMENTATION` forces either backend.

`src/psi/utils/utils.py` -- a commented-out `token="hf_..."` argument is
removed. Upstream committed a live Hugging Face access token there; GitHub's
push protection blocks the whole branch over it, and vendoring somebody's
credential is wrong whether or not a scanner objects. The call reads
`HF_TOKEN` from the environment like every other one.

`src/psi/utils/lora.py` (new), `src/psi/trainers/finetune.py::apply_lora`,
`src/psi/config/model_psi0.py` (`lora_*`, `tune_dit_full`) and
`src/psi/models/psi0.py::from_pretrained` (merge on load) -- LoRA in pi0.5's
shape for the MHBench comparison; off by default (rank 0), so the paper's
recipe is unchanged.

`src/psi/data/` and `scripts/data/` were missing from the vendored tree -- the
adapter's `.gitignore` said `data/` unanchored -- and are restored from the
commit above (2026-09-10).

Nothing else is modified. Everything MHBench-specific lives in the adapter one
directory up.

## What is not vendored

`real/assets/` (181 MB of STL meshes) and `real/teleop/` (61 MB) are excluded by
`policy/Psi0/.gitignore` -- the CAD and rig assets for building and driving a
physical G1. MHBench is a simulator; nothing in the adapter or the training path
opens them. Re-fetch them from the commit above if a real robot ever needs them.
