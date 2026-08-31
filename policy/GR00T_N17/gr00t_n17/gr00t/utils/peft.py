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

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=list(EMBODIMENT_MODULES),
    )

    base_model = model
    # What the model itself asked to train, before peft has an opinion.
    # `LoraModel._mark_only_adapters_as_trainable` freezes every parameter
    # whose name lacks the `lora_` prefix, and that silently includes whatever
    # `--tune_visual` / `--tune_llm` had just unfrozen: the flag would be set,
    # the wandb tag would read `vision-tuned`, and the vision tower would not
    # move all run. Restore them after wrapping, because "frozen backbone with
    # LoRA on it, vision tower trained outright" is exactly pi0.5's
    # arrangement and it is only expressible if the two settings compose.
    #
    # An adapted layer's own weight is *not* restored: peft renames it to
    # `<name>.base_layer.weight`, which matches nothing here, and a module in
    # `modules_to_save` becomes `<name>.modules_to_save.default.<param>` beside
    # a frozen `<name>.original_module.<param>` -- also no match. Only the
    # parameters peft left untouched come back.
    # Scoped to the backbone deliberately. The action head's own
    # tune_projector / tune_diffusion_model / tune_vlln all default to True and
    # `ActionHead.set_trainable_parameters` starts by unfreezing *everything*,
    # so an unscoped restore also brings back the DiT's adaLN modulation
    # (`model.transformer_blocks.*.norm1.linear`, 151 M of 162 M) -- which is
    # the full DiT tuning this recipe exists to replace, and on its own three
    # times pi0.5's entire adapter budget. The embodiment banks stay trainable
    # through `modules_to_save`, which is the part of tune_projector a
    # NEW_EMBODIMENT run actually needs.
    pretrained_trainable = {n for n, p in model.named_parameters()
                            if p.requires_grad and n.startswith("backbone.")}

    model = get_peft_model(model, lora_config)

    peft_prefix = "base_model.model."
    restored = 0
    for name, param in model.named_parameters():
        original = name[len(peft_prefix):] if name.startswith(peft_prefix) else name
        if not param.requires_grad and original in pretrained_trainable:
            param.requires_grad = True
            restored += param.numel()
    if restored:
        print(f"[peft] restored {restored:,} parameter(s) that the model had marked "
              f"trainable and peft froze (tune_visual / tune_llm)")
    model.print_trainable_parameters()

    model = _wrap_forward(model, base_model)

    return model