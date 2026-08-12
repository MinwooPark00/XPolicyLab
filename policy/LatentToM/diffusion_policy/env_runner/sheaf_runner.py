"""Per-arm observation history buffer for online (server-side) inference.

LatentToM's own training/eval code always pulls a full ``n_obs_steps`` window
out of a pre-built batch (see ``build_arm_sub_batch`` in
``diffusion_policy.common.pytorch_util``); there is no equivalent for a
policy server that receives one observation at a time. This mirrors
``diffusion_policy.env_runner.dp_runner.DPRunner`` (XPolicyLab's DP adapter)
generalized from one shared obs dict to two arm-specific obs dicts, keyed by
env index so a batched evaluation client can drive several environments at
once.
"""

from collections import deque

import numpy as np
import torch

MAX_ENVS = 100  # matches DPRunner's own fixed pool size


class SheafRunner:
    def __init__(self, n_obs_steps=2):
        self.n_obs_steps = n_obs_steps
        self.arm1_history = [deque(maxlen=n_obs_steps) for _ in range(MAX_ENVS)]
        self.arm2_history = [deque(maxlen=n_obs_steps) for _ in range(MAX_ENVS)]

    def reset(self):
        for q in self.arm1_history:
            q.clear()
        for q in self.arm2_history:
            q.clear()

    def update_obs(self, arm1_obs_list, arm2_obs_list, env_idx_list):
        for env_idx, arm1_obs, arm2_obs in zip(env_idx_list, arm1_obs_list, arm2_obs_list):
            self.arm1_history[env_idx].append(arm1_obs)
            self.arm2_history[env_idx].append(arm2_obs)

    @staticmethod
    def _stack_last_n(values, n_steps):
        """Left-pad-by-repeat a list of same-shaped arrays out to ``n_steps``."""
        values = list(values)
        template = values[-1]
        result = np.zeros((n_steps,) + template.shape, dtype=template.dtype)
        start = -min(n_steps, len(values))
        result[start:] = np.stack(values[start:])
        if n_steps > len(values):
            result[:start] = result[start]
        return result

    def _batch(self, history, env_idx_list, device):
        batch = []
        for env_idx in env_idx_list:
            frames = history[env_idx]
            keys = frames[0].keys()
            stacked = {k: self._stack_last_n([f[k] for f in frames], self.n_obs_steps) for k in keys}
            batch.append(stacked)
        # dict[str, (n_obs_steps, ...)] per env -> dict[str, (B, n_obs_steps, ...)]
        obs_dict = {}
        for key in batch[0].keys():
            obs_dict[key] = torch.stack(
                [torch.from_numpy(sample[key]).to(device=device) for sample in batch], dim=0
            )
        return obs_dict

    def get_arm_obs_batch(self, env_idx_list, device):
        arm1_obs = self._batch(self.arm1_history, env_idx_list, device)
        arm2_obs = self._batch(self.arm2_history, env_idx_list, device)
        return arm1_obs, arm2_obs
