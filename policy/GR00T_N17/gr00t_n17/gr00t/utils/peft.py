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

import torch
from peft import LoraConfig, get_peft_model

# The embodiment-conditioned banks that hold the NEW_EMBODIMENT slot.
EMBODIMENT_MODULES = ["state_encoder", "action_encoder", "action_decoder"]


def _wrap_forward(peft_model, base_model):
    """Route calls through the base model's own forward.

    peft's task wrappers (`PeftModelForCausalLM`) expect `input_ids`-style
    kwargs; Gr00tN1d7.forward takes a single `inputs` dict. The LoRA layers
    live inside the base model's modules, so the base bound method sees them.
    Same intent as the n1.5 `_wrap_forward`.
    """
    peft_model.forward = base_model.forward
    return peft_model


def get_lora_model(model, rank=32, lora_alpha=16, lora_dropout=0.1, action_head_only=True):
    target_modules = []

    # Inspect model structure to find the correct paths
    for name, module in model.named_modules():
        if action_head_only and "action_head" not in name:
            continue

        # Look for linear layers in attention mechanisms
        if isinstance(module, torch.nn.Linear):
            if any(x in name for x in ["q_proj", "v_proj", "to_q", "to_v", "k_proj", "to_k"]):
                target_modules.append(name)

    if not target_modules:
        raise RuntimeError(
            "get_lora_model found no attention projections to adapt "
            f"(action_head_only={action_head_only}); the module naming has drifted."
        )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=list(EMBODIMENT_MODULES),
    )

    base_model = model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model = _wrap_forward(model, base_model)

    return model