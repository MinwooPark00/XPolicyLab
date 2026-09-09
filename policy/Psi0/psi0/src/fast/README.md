# Fit FAST for Pre-Training Psi0

change to the project root
```
cd /path/to/psi0
```

## Pre-Process EgoDex dataset
The script is adapted from [H-RDT EgoDex Pre-Processing](https://github.com/HongzheBi/H_RDT?tab=readme-ov-file#data-preprocessing) to Pre-compute the `48 DoF EgoDex action`.

Change the paths  and run

> Tweak the `NUM_PROCESSES` if on a powerful server, i tried max 64.

```
source src/h_rdt/datasets/pretrain/setup_pretrain.sh
```
and then 

> Tweak the parameters in step 2 if needed, eg.,  
>    --use_delta_actions \
>    --upsample_rate 3

```
source .venv-psi/bin/activate
bash src/h_rdt/datasets/pretrain/run_pretrain_pipeline.sh
```

## Download Official FAST tokenizer

```
python src/fast/download.py
```

and patch the original FAST tokenizer to avoid decoding error, see the authors's discussion [here](https://huggingface.co/physical-intelligence/fast/discussions/4)
```
python src/fast/patch_pi_action_tokenizer.py
```

## Fit the FAST tokenizer
> Try to set a reasonable `num_action_chunks` to obtain a reasonable stats.

> Feel free to tweak the training paramters, and we prefer smaller reconstruction error above everything else.

The script will load the dataset configured from the `training script` and train `FAST` tokenizer

```
python scripts/fast.py 
    --scale 100 \
    --vocab-size 2024 \
    --num_action_chunks 500000 \
    --training_script scripts/train/psi0/pretrain-egodex-psi0-fast.sh
```

After training, you can find a new FAST tokenizer is stored under `src/fast/...`, which can be loaded later through
`--model.action_tokenizer.pretrained_checkpoint=...`

***That's it!***