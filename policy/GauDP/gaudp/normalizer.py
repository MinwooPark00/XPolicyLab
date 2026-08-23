"""Separate state/action min-max normalization for GauDP."""

from __future__ import annotations

import torch
from torch import nn

from .schema import ACTION_DIM, PROPRIO_DIM


class GauDPNormalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state_min", torch.zeros(PROPRIO_DIM))
        self.register_buffer("state_max", torch.ones(PROPRIO_DIM))
        self.register_buffer("action_min", torch.zeros(ACTION_DIM))
        self.register_buffer("action_max", torch.ones(ACTION_DIM))

    def fit(self, state, action) -> None:
        state = torch.as_tensor(state, dtype=torch.float32)
        action = torch.as_tensor(action, dtype=torch.float32)
        if state.ndim != 2 or state.shape[-1] != PROPRIO_DIM:
            raise ValueError(f"state statistics require [N,{PROPRIO_DIM}], got {tuple(state.shape)}")
        if action.ndim != 2 or action.shape[-1] != ACTION_DIM:
            raise ValueError(f"action statistics require [N,{ACTION_DIM}], got {tuple(action.shape)}")
        if not torch.isfinite(state).all():
            raise ValueError("state statistics contain NaN or Inf")
        if not torch.isfinite(action).all():
            raise ValueError("action statistics contain NaN or Inf")
        self.state_min = state.amin(dim=0)
        self.state_max = state.amax(dim=0)
        self.action_min = action.amin(dim=0)
        self.action_max = action.amax(dim=0)

    @staticmethod
    def _normalize(value: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor) -> torch.Tensor:
        scale = (maximum - minimum).clamp_min(1e-6)
        return ((value - minimum) / scale) * 2.0 - 1.0

    @staticmethod
    def _unnormalize(value: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor) -> torch.Tensor:
        scale = (maximum - minimum).clamp_min(1e-6)
        return ((value + 1.0) * 0.5) * scale + minimum

    def normalize_state(self, value: torch.Tensor) -> torch.Tensor:
        return self._normalize(value, self.state_min, self.state_max)

    def normalize_action(self, value: torch.Tensor) -> torch.Tensor:
        return self._normalize(value, self.action_min, self.action_max)

    def unnormalize_action(self, value: torch.Tensor) -> torch.Tensor:
        return self._unnormalize(value, self.action_min, self.action_max)

    def range_diagnostics(self, state, action) -> dict[str, float]:
        """Measure held-out values outside the train-fitted min/max range."""
        state = torch.as_tensor(state, dtype=torch.float32, device=self.state_min.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.action_min.device)
        if not torch.isfinite(state).all() or not torch.isfinite(action).all():
            raise ValueError("normalization diagnostics received NaN or Inf")
        normalized_state = self.normalize_state(state)
        normalized_action = self.normalize_action(action)
        return {
            "normalization/val_state_out_of_range_fraction": float((normalized_state.abs() > 1).float().mean()),
            "normalization/val_action_out_of_range_fraction": float((normalized_action.abs() > 1).float().mean()),
            "normalization/val_state_max_abs": float(normalized_state.abs().amax()),
            "normalization/val_action_max_abs": float(normalized_action.abs().amax()),
        }
