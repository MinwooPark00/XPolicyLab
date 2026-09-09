# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# LoRA finetuning support, ported from the n1.5-release `gr00t/utils/peft.py`
# (the N1.7 codebase dropped it). Same recipe: adapters on the attention
# projections, default rank 32 / alpha 16 / dropout 0.1, action head only.
#
# Two N1.7 adaptations:
#   * The DiT uses diffusers' `Attention`, whose projections are named
#     `to_q`/`to_k`/`to_v` -- already in the n1.5 target list, kept as is.
#   * `modules_to_save` carries the embodiment-specific
#     state_encoder / action_encoder / action_decoder banks. A NEW_EMBODIMENT
#     run trains a fresh slot in those banks; freezing them (what a plain
#     `get_peft_model` does) would leave the new embodiment's encoders at
#     their random init and the policy could never work. This mirrors the
#     stock finetune recipe, where `tune_projector=True` keeps exactly these
#     three modules trainable while LoRA replaces full DiT tuning.

import os

import torch
from peft import LoraConfig, get_peft_model

# The embodiment-conditioned banks that hold the NEW_EMBODIMENT slot.
EMBODIMENT_MODULES = ["state_encoder", "action_encoder", "action_decoder"]

# Which projections carry an adapter.
#
# `attn` is the stock recipe ported from n1.5 -- query/key/value and nothing
# else. `attn+ffn` adds the attention *output* projection and the feed-forward,
# which is the shape openpi gives pi0.5: `gemma_2b_lora` and `gemma_300m_lora`
# both declare `lora_configs` for "attn" *and* "ffn". Select it with
# MHBENCH_LORA_TARGETS=attn+ffn when the point of a run is for the two
# baselines to differ in method rather than in how much of the model a gradient
# can reach at all.
#
# The vision tower is in neither set, and not by omission: its attention is one
# fused `qkv` Linear and its MLP is `linear_fc1`/`linear_fc2`
# (backbone.model.model.visual.blocks.*), so no name here can match it. pi0.5
# does not adapt SigLIP with LoRA either -- it trains it outright, and the
# equivalent here is `--tune_visual`.
LORA_TARGET_SETS = {
    "attn": ("q_proj", "k_proj", "v_proj", "to_q", "to_k", "to_v"),
    "attn+ffn": ("q_proj", "k_proj", "v_proj", "to_q", "to_k", "to_v",
                 "o_proj", "to_out", "gate_proj", "up_proj", "down_proj", "ff.net"),
}


def _wrap_forward(peft_model, base_model):
    """Route calls through the base model's own forward.

    peft's task wrappers (`PeftModelForCausalLM`) expect `input_ids`-style
    kwargs; Gr00tN1d7.forward takes a single `inputs` dict. The LoRA layers
    live inside the base model's modules, so the base bound method sees them.
    Same intent as the n1.5 `_wrap_forward`.
    """
    peft_model.forward = base_model.forward
    return peft_model


def trainable_backbone_modules(model, target_modules) -> list:
    """Backbone modules that `--tune_visual` / `--tune_llm` asked to train, named
    so peft will *save* them.

    Making a parameter `requires_grad=True` is only half of it. peft writes
    `adapter_model.safetensors` from the adapter plus `modules_to_save` and
    nothing else, so a merely-unfrozen parameter trains for the whole run and
    is then absent from the checkpoint. That is exactly what happened to the
    first `--tune_visual` run here: 786,218,112 parameters trainable, 12 h 42 m
    of training, and 379,261,056 written -- the vision tower silently dropped,
    and the merge would have folded LoRA onto the *pretrained* vision weights.

    Returned names are full module paths, which is what `modules_to_save`
    matches on (`key.endswith(target)`). Only maximal modules whose parameters
    are all trainable are listed, and never one that already contains a LoRA
    target -- an adapted submodule is handled by the adapter, and wrapping its
    parent would save the frozen base weights a second time.
    """
    adapted = set(target_modules)
    names = []
    for name, module in model.named_modules():          # pre-order: parents first
        if not name.startswith("backbone."):
            continue
        if any(name == a or a.startswith(name + ".") for a in adapted):
            continue
        params = list(module.parameters(recurse=True))
        if not params or not all(p.requires_grad for p in params):
            continue
        if any(name.startswith(f"{seen}.") for seen in names):
            continue
        names.append(name)
    return names


def get_lora_model(model, rank=32, lora_alpha=16, lora_dropout=0.1, action_head_only=True,
                   targets=None):
    targets = targets or os.environ.get("MHBENCH_LORA_TARGETS", "attn")
    if targets not in LORA_TARGET_SETS:
        raise ValueError(
            f"unknown LoRA target set {targets!r}; expected one of {sorted(LORA_TARGET_SETS)}"
        )
    suffixes = LORA_TARGET_SETS[targets]

    target_modules = []

    # Inspect model structure to find the correct paths
    for name, module in model.named_modules():
        if action_head_only and "action_head" not in name:
            continue

        # Look for linear layers in attention mechanisms
        if isinstance(module, torch.nn.Linear):
            if any(x in name for x in suffixes):
                target_modules.append(name)

    if not target_modules:
        raise RuntimeError(
            f"get_lora_model found no {targets} projections to adapt "
            f"(action_head_only={action_head_only}); the module naming has drifted."
        )
    print(f"[peft] LoRA targets={targets} action_head_only={action_head_only}: "
          f"{len(target_modules)} module(s), "
          f"{sum('action_head' in n for n in target_modules)} in the action head")

    # The backbone the model wants trained rides in `modules_to_save` beside the
    # embodiment banks, which is both what makes it trainable under peft and
    # what gets it written to the checkpoint.
    saved_backbone = trainable_backbone_modules(model, target_modules)
    if saved_backbone:
        print(f"[peft] modules_to_save also carries {len(saved_backbone)} backbone "
              f"module(s) the model asked to train: {', '.join(saved_backbone)}")

    # pi0.5 adapts its two Gemmas at different ranks: 16/16 on the 2B language
    # model, 32/32 on the 300M action expert (openpi gemma.py, gemma_2b_lora /
    # gemma_300m_lora). `rank`/`lora_alpha` above are the action head's; these
    # two put the backbone's adapters at their own numbers. peft matches the
    # pattern keys with re.fullmatch, so the full module names are the keys.
    rank_pattern, alpha_pattern = {}, {}
    backbone_rank = int(os.environ.get("MHBENCH_LORA_BACKBONE_RANK", "0") or 0)
    backbone_alpha = os.environ.get("MHBENCH_LORA_BACKBONE_ALPHA", "")
    backbone_targets = [n for n in target_modules if n.startswith("backbone.")]
    if backbone_targets and (backbone_rank or backbone_alpha):
        for name in backbone_targets:
            if backbone_rank:
                rank_pattern[name] = backbone_rank
            if backbone_alpha:
                alpha_pattern[name] = float(backbone_alpha)
        print(f"[peft] backbone adapters at rank {backbone_rank or rank} / alpha "
              f"{backbone_alpha or lora_alpha} ({len(backbone_targets)} modules); "
              f"action head at rank {rank} / alpha {lora_alpha}")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=list(EMBODIMENT_MODULES) + saved_backbone,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )

    base_model = model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model = _wrap_forward(model, base_model)

    return model