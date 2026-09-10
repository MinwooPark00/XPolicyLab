# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""LoRA adapters for the shared transformer stack.

Not upstream. LingBot-VA post-trains the whole 5.3B model under FSDP, which
needs at least two 96 GB cards; every other world-action baseline in MHBench
(FastWAM) is trained as a rank-32 LoRA on one card, so this gives LingBot-VA
the same arm.

The video and action streams share `model.blocks`, so an adapter on a block is
seen by both. What is *not* shared -- `action_embedder` and `action_proj_out` --
is trained in full, because MHBench's 35D action does not fit the pretrained
30D projections and those two tensors are reinitialised anyway.

Adapters merge back into the base weights on save, so a checkpoint is keyed
exactly like a full fine-tune and the inference server never sees a LoRA.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# The ten projections in every WanTransformerBlock. Same set FastWAM adapts.
LORA_TARGETS = (
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
    "ffn.net.0.proj", "ffn.net.2",
)


def _get_module(root: nn.Module, path: str):
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _set_module(root: nn.Module, path: str, value: nn.Module):
    parts = path.split(".")
    parent = _get_module(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    if parts[-1].isdigit():
        parent[int(parts[-1])] = value
    else:
        setattr(parent, parts[-1], value)


class LoRALinear(nn.Module):
    """`base` frozen in its loaded dtype, adapters kept in fp32 and cast per call."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank = rank
        self.scaling = float(alpha) / float(rank)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.base(x)
        h = self.lora_dropout(x)
        h = F.linear(h, self.lora_A.to(x.dtype))
        h = F.linear(h, self.lora_B.to(x.dtype))
        return out + h * self.scaling

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """base.weight + BA*scaling, without touching the base."""
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return self.base.weight + delta.to(self.base.weight.dtype)


class FP32Linear(nn.Module):
    """A fully trainable Linear held in fp32 next to a bf16 base model."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base.float()

    def forward(self, x):
        bias = self.base.bias.to(x.dtype) if self.base.bias is not None else None
        return F.linear(x, self.base.weight.to(x.dtype), bias)


def apply_lora(model: nn.Module, rank: int, alpha: float, dropout: float,
               train_action_projections: bool = True,
               train_action_condition: bool = False):
    """Freeze the model, adapt every block projection, unfreeze the action head.

    Returns (trainable_parameters, lora_parameters).
    """
    model.requires_grad_(False)

    n_lora = 0
    for block in model.blocks:
        # apply_ac() wraps blocks in a checkpoint wrapper; reach through it.
        target = getattr(block, "_checkpoint_wrapped_module", block)
        for path in LORA_TARGETS:
            base = _get_module(target, path)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"LoRA target {path} is {type(base).__name__}, not Linear")
            wrapped = LoRALinear(base, rank, alpha, dropout)
            _set_module(target, path, wrapped)
            n_lora += wrapped.lora_A.numel() + wrapped.lora_B.numel()

    if train_action_projections:
        model.action_embedder = FP32Linear(model.action_embedder)
        model.action_proj_out = FP32Linear(model.action_proj_out)
        model.action_embedder.requires_grad_(True)
        model.action_proj_out.requires_grad_(True)

    if train_action_condition:
        # Left in the model's dtype on purpose: it runs on bf16 activations, so
        # upcasting it here would raise on the first matmul.
        model.condition_embedder_action.requires_grad_(True)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_train, n_lora


@torch.no_grad()
def merged_state_dict(model: nn.Module, state_dict: dict) -> dict:
    """Fold the adapters into a state dict keyed the way upstream saves.

    Works off the state dict alone, not the live parameters, so it is correct
    for an FSDP full_state_dict gather as well as for the single-card path.
    Non-destructive: the live model keeps its adapters, so training continues
    after a checkpoint and two saves do not apply the delta twice.
    """
    # Activation checkpointing wraps each block, so named_modules() carries a
    # `_checkpoint_wrapped_module` segment that the wrapper's state_dict hook
    # strips. Normalise, or every lookup below misses and the adapters are
    # silently dropped from the saved weights.
    scalings = {name.replace('._checkpoint_wrapped_module', ''): mod.scaling
                for name, mod in model.named_modules() if isinstance(mod, LoRALinear)}

    out = {}
    for key, value in state_dict.items():
        if key.endswith(".lora_A") or key.endswith(".lora_B"):
            continue
        if key.endswith(".base.weight"):
            owner = key[: -len(".base.weight")]
            if owner in scalings:
                a = state_dict[f"{owner}.lora_A"]
                b = state_dict[f"{owner}.lora_B"]
                delta = (b.float() @ a.float()) * scalings[owner]
                value = value + delta.to(value.dtype).to(value.device)
            key = f"{owner}.weight"
        elif key.endswith(".base.bias"):
            key = f"{key[: -len('.base.bias')]}.bias"
        out[key] = value
    return out
