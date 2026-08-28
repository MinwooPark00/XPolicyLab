"""LoRA adapters for the FastWAM MoT experts.

Upstream FastWAM has no parameter-efficient path: `train.sh` fine-tunes all
6.021B DiT parameters, which needs two 96 GB cards just to hold the optimizer
state. The MHBench baselines it is compared against are both PEFT -- GR00T_N17
runs LoRA r32 with the LLM and vision tower frozen, pi0.5 runs
`gemma_2b_lora` + `gemma_300m_lora` behind a freeze filter -- so this adds the
same lever here, on the video expert's linear projections.

Two properties matter downstream:

  * `lora_B` starts at zero, so an injected model is numerically identical to
    the checkpoint it was built from until the first optimizer step.
  * `merged_state_dict` folds the adapters back into the base weights under
    the ORIGINAL key names, so a saved checkpoint is indistinguishable from a
    full or freeze run's and the eval server needs no LoRA support at all.
"""

import math

import torch
from torch import nn

# One DiTBlock's linear projections, by attribute path inside the block.
DEFAULT_TARGETS = (
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    "ffn.0", "ffn.2",
)


class LoRALinear(nn.Module):
    """`base(x) + scaling * B @ A @ dropout(x)`, with `base` left frozen."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        weight = base.weight
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, base.in_features, device=weight.device, dtype=weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, self.rank, device=weight.device, dtype=weight.dtype)
        )
        # A as a normal linear init, B at zero: the delta is exactly 0 at step 0.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, x):
        delta = self.dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)
        return self.base(x) + self.scaling * delta

    def extra_repr(self):
        return f"rank={self.rank}, scaling={self.scaling:.4f}"


def inject_lora(root: nn.Module, rank: int, alpha: float, dropout: float,
                targets=DEFAULT_TARGETS) -> tuple[int, int]:
    """Wrap every `nn.Linear` under `root` whose name ends in one of `targets`.

    Returns (layers wrapped, LoRA parameters added). Collect first, mutate
    after: replacing modules while walking `named_modules` skips siblings.
    """
    victims = []
    for name, module in root.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(name == t or name.endswith("." + t) for t in targets):
            victims.append((name, module))

    added = 0
    for name, module in victims:
        parent_name, _, attr = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        wrapper = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr, wrapper)
        added += wrapper.lora_A.numel() + wrapper.lora_B.numel()
    return len(victims), added


def lora_parameters(root: nn.Module):
    for module in root.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_A
            yield module.lora_B


def has_lora(root: nn.Module) -> bool:
    return any(isinstance(m, LoRALinear) for m in root.modules())


def merged_state_dict(root: nn.Module) -> dict:
    """`root.state_dict()` with the adapters folded into the base weights.

    `LoRALinear` stores the wrapped layer at `<path>.base.*`, which no other
    run's checkpoint has. Folding restores `<path>.weight` / `<path>.bias`, so
    a LoRA run saves the same key set as a full or freeze run and the policy
    server keeps loading checkpoints the one way it always has.
    """
    wrapped = {name: m for name, m in root.named_modules() if isinstance(m, LoRALinear)}
    state = root.state_dict()
    if not wrapped:
        return state

    merged = {}
    for key, value in state.items():
        if key.endswith(".lora_A") or key.endswith(".lora_B"):
            continue
        prefix, sep, suffix = key.rpartition(".base.")
        if sep and prefix in wrapped:
            if suffix == "weight":
                module = wrapped[prefix]
                delta = module.lora_B.data.to(torch.float32) @ module.lora_A.data.to(torch.float32)
                value = (value.to(torch.float32) + module.scaling * delta).to(value.dtype)
            merged[f"{prefix}.{suffix}"] = value
            continue
        merged[key] = value
    return merged
