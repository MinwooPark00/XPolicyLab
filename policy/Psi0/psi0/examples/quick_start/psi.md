# Psi Quick Start

This guide assumes a fresh clone of the public repo at
`https://github.com/physical-superintelligence-lab/Psi0`.

The baseline quick starts under this directory use the xmove-pick SIMPLE dataset:

- `third_party/SIMPLE/data/G1WholebodyXMoveAndPickMP-v0`

Unless noted otherwise, commands assume you are already inside the top-level
Psi0 dev shell started from the repo root with `nix develop`.

## 1. Clone and initialize the repo

```bash
git clone git@github.com:physical-superintelligence-lab/Psi0.git Psi0
cd Psi0
git submodule update --init --recursive
```

If your Git setup blocks local `file` transport for submodules, use:

```bash
git -c protocol.file.allow=always submodule update --init --recursive
```

Pull Git LFS content inside SIMPLE:

```bash
cd third_party/SIMPLE
git submodule foreach --recursive 'git lfs install --local || true'
git submodule foreach --recursive 'git lfs pull || true'
cd ../..
```

## 2. Enter the repo dev shell

```bash
cd /path/to/Psi0
env -u LD_LIBRARY_PATH nix --extra-experimental-features "nix-command flakes" develop
```

This shell composes the PSI and SIMPLE runtime.

## 3. Create the repo Python environment

From the repo root:

```bash
uv venv .venv-psi --python 3.10
source .venv-psi/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups --index-strategy unsafe-best-match --active
cp .env.sample .env
```

The repo-level `uv sync` installs `psi` and `simple` as editable packages.

## 4. Prepare baseline-specific environments

H-RDT:

```bash
cd /path/to/Psi0/src/h_rdt
uv sync --frozen
```

EgoVLA:

```bash
cd /path/to/Psi0/src/egovla
uv sync --frozen
```

<!-- GR00T uses the repo-level environment and launchers under
`baselines/gr00t-n1.6`. -->

GR00T-N1.6

```bash
cd /path/to/Psi0/src/gr00t
uv sync --frozen
```

## 5. Download required assets

The baseline guides below use:

- Dataset: `third_party/SIMPLE/data/G1WholebodyXMoveAndPickMP-v0`
- H-RDT release weights downloaded under `src/h_rdt`
- EgoVLA base checkpoint downloaded under `src/egovla/checkpoints`

Download the H-RDT release weights:

```bash
cd /path/to/Psi0/src/h_rdt
huggingface-cli download --resume-download embodiedfoundation/H-RDT --local-dir ./
```

This download provides the DINO-SigLIP files under `bak/dino-siglip` and the
pretrained backbone under `checkpoints/pretrain-0618/.../pytorch_model.bin`.

Download the EgoVLA base checkpoint:

```bash
cd /path/to/Psi0/src/egovla
source .venv/bin/activate
huggingface-cli download rchal97/egovla_base_vlm --repo-type model --local-dir checkpoints
```

That download currently provides a zip file. Extract it and link it to the
path expected by `finetune.sh`:

```bash
cd /path/to/Psi0/src/egovla/checkpoints
unzip -q vila-qwen2-vl-1.5b-instruct-sft-20240830191953.zip
ln -sfn \
  vila-qwen2-vl-1.5b-instruct-sft-20240830191953 \
  mix4data-30hz-transv2update2-fingertip-20e-hdof5-3d200-rot5-lr1e-4-h5p30f1skip6-b16-4
```

## Related Guides

- `gr00t.md`
- `hrdt.md`
- `egovla.md`
- `simple.md`
