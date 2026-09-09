"""LoRA for Ψ₀ in pi0.5's shape, and the merge that makes a checkpoint plain again.

MHBench compares Ψ₀ against pi0.5 and GR00T N1.7 under one adapter recipe --
openpi's ``gemma_2b_lora`` / ``gemma_300m_lora``: rank 16 / alpha 16 on the
language model's attention *and* feed-forward projections, rank 32 / alpha 32
on the action expert's, the vision tower and the small projections trained in
full. Here that is:

* ``LLM_TARGETS``  Qwen3-VL's text layers: q/k/v/o and gate/up/down.
* ``DIT_TARGETS``  the MM-DiT blocks: every attention projection (the action
  stream's ``to_q/k/v/out`` and the observation stream's ``add_*_proj`` /
  ``to_add_out``) and both feed-forwards. The adaLN modulation linears stay
  frozen -- they are 151 M of the 162 M block parameters and adapting them
  would be the full DiT tuning LoRA replaces (the same call GR00T's recipe
  makes).
* ``HEADER_FULL``  what the action header still trains outright when its
  blocks carry adapters: the observation projection (VLM tokens -> DiT width),
  the action in/out projections and the time embedding -- the pieces pi0.5
  also trains in full because they are new or tiny.

The adapters are injected in place (``peft.inject_adapter_in_model``), so the
module tree keeps its names and the trainer's optimizer groups, gradient
norms and checkpoints see ordinary parameters. A checkpoint therefore holds
``<module>.base_layer.weight`` + ``<module>.lora_A.default.weight`` +
``<module>.lora_B.default.weight`` where a plain run holds ``<module>.weight``;
``merge_lora_state_dict`` folds that back so serving loads a plain model.
"""

from __future__ import annotations

import re
from typing import Callable

import torch
import torch.nn as nn

LLM_TARGETS = re.compile(
    r"vlm_model\.model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
)
DIT_TARGETS = re.compile(
    r"action_header\.transformer_blocks\.\d+\."
    r"(attn\.(to_q|to_k|to_v|to_out\.0|add_q_proj|add_k_proj|add_v_proj|to_add_out)"
    r"|ff_act\.net\.(0\.proj|2)|ff_obs\.net\.(0\.proj|2))"
)
HEADER_FULL = (
    "action_header.obs_proj.",
    "action_header.action_proj_in.",
    "action_header.action_proj_out.",
    "action_header.time_ins_embed.",
)

_LORA_A = re.compile(r"^(?P<prefix>.+)\.lora_A\.(?P<adapter>[^.]+)\.weight$")


def lora_targets(model: nn.Module, llm: bool, dit: bool) -> tuple[list[str], list[str]]:
    """(language-model targets, action-expert targets): the nn.Linear names."""
    llm_names, dit_names = [], []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if llm and LLM_TARGETS.fullmatch(name):
            llm_names.append(name)
        elif dit and DIT_TARGETS.fullmatch(name):
            dit_names.append(name)
    return llm_names, dit_names


def inject_lora(model: nn.Module, *, llm_rank: int, llm_alpha: float, dit_rank: int,
                dit_alpha: float, dropout: float = 0.0) -> dict[str, int]:
    """Put adapters on the language model and/or the action expert, in place.

    Returns ``{"llm": n, "dit": n}``, how many Linear modules each got.
    peft freezes every non-adapter parameter as it injects; the caller decides
    what else trains afterwards (``FinetuneTrainer.apply_lora``).
    """
    from peft import LoraConfig, inject_adapter_in_model

    llm_names, dit_names = lora_targets(model, llm=llm_rank > 0, dit=dit_rank > 0)
    targets = llm_names + dit_names
    if not targets:
        raise RuntimeError("inject_lora found no target Linear modules -- the module naming drifted")

    # One config, two ranks: the pattern keys are matched by re.fullmatch, so
    # the full module names are the keys.
    rank_pattern = {n: dit_rank for n in dit_names}
    alpha_pattern = {n: float(dit_alpha) for n in dit_names}
    base_rank, base_alpha = (llm_rank, llm_alpha) if llm_names else (dit_rank, dit_alpha)
    config = LoraConfig(
        r=base_rank,
        lora_alpha=float(base_alpha),
        lora_dropout=float(dropout),
        target_modules=targets,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        bias="none",
        init_lora_weights=True,
    )
    inject_adapter_in_model(config, model)
    return {"llm": len(llm_names), "dit": len(dit_names)}


def is_lora_param(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name or ".lora_embedding_" in name


def merge_lora_state_dict(state_dict: dict[str, torch.Tensor],
                          alpha_of: Callable[[str], float]) -> dict[str, torch.Tensor]:
    """Fold ``base_layer + scaling * B @ A`` into plain ``weight`` keys.

    ``alpha_of(prefix)`` gives the adapter's alpha for a module (its rank is
    ``A.shape[0]``); scaling is alpha / rank, i.e. peft's default without
    rsLoRA. A state dict with no adapter keys comes back unchanged.
    """
    merged = dict(state_dict)
    touched = 0
    for key in list(state_dict):
        match = _LORA_A.match(key)
        if match is None:
            continue
        prefix, adapter = match.group("prefix"), match.group("adapter")
        a = state_dict[key]
        b_key = f"{prefix}.lora_B.{adapter}.weight"
        base_key = f"{prefix}.base_layer.weight"
        if b_key not in state_dict or base_key not in state_dict:
            raise KeyError(f"{key} has no matching {b_key} / {base_key}")
        b = state_dict[b_key]
        base = state_dict[base_key]
        scaling = float(alpha_of(prefix)) / a.shape[0]
        delta = (b.to(torch.float32) @ a.to(torch.float32)) * scaling
        merged[f"{prefix}.weight"] = (base.to(torch.float32) + delta).to(base.dtype)
        for k in (key, b_key, base_key):
            merged.pop(k, None)
        bias_key = f"{prefix}.base_layer.bias"
        if bias_key in merged:
            merged[f"{prefix}.bias"] = merged.pop(bias_key)
        touched += 1
    # Any leftover adapter bookkeeping (dropout has none; lora_magnitude for DoRA).
    for key in list(merged):
        if is_lora_param(key):
            raise KeyError(f"unmerged adapter tensor left behind: {key}")
    if touched:
        print(f"[lora] merged {touched} adapter(s) into plain weights")
    return merged


def alpha_for_config(model_cfg) -> Callable[[str], float]:
    """The alpha a run's model config gave each part, keyed by module prefix."""
    def alpha_of(prefix: str) -> float:
        if prefix.startswith("action_header."):
            return float(getattr(model_cfg, "lora_dit_alpha", 32.0))
        return float(getattr(model_cfg, "lora_llm_alpha", 16.0))
    return alpha_of
