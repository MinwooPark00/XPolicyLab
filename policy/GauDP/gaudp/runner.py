"""Per-environment observation history for online GauDP inference."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import torch


class GauDPRunner:
    def __init__(self, n_obs_steps: int) -> None:
        self.n_obs_steps = int(n_obs_steps)
        self._history = defaultdict(lambda: deque(maxlen=self.n_obs_steps))

    def update(self, env_idx: int, images: np.ndarray, state: np.ndarray) -> None:
        self._history[int(env_idx)].append((np.asarray(images, np.float32), np.asarray(state, np.float32)))

    def batch(self, env_indices: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        image_batch, state_batch = [], []
        for env_idx in env_indices:
            history = list(self._history[int(env_idx)])
            if not history:
                raise RuntimeError(f"environment {env_idx} has no observation")
            history = [history[0]] * (self.n_obs_steps - len(history)) + history
            image_batch.append(np.stack([item[0] for item in history]))
            state_batch.append(np.stack([item[1] for item in history]))
        return (
            torch.as_tensor(np.stack(image_batch), device=device),
            torch.as_tensor(np.stack(state_batch), device=device),
        )

    def reset(self) -> None:
        self._history.clear()
