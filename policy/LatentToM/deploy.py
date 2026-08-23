"""Rollout loop for LatentToM.

`OBS_STRIDE = 1` is the shared loop unchanged -- observe after every executed
step -- and this policy needs it. Both arms condition on a window:
`sheaf_xarm_split_diffusion_workspace.yaml` sets `n_obs_steps: 2`, `SheafRunner`
holds that many frames per arm per environment, and `_stack_last_n` feeds the
whole window to each arm's `SheafObsEncoder`. The observations the loop collects
inside a chunk are that window, so skipping them would hand both arms the same
frame twice -- no motion between them -- for the twenty steps of a chunk.

`scripts/eval_policy_xpolicylab.py` reads this value to decide whether
`MHBenchTaskEnv.get_obs` may re-serve a cached render; 1 renders every step.
"""

from XPolicyLab.utils.rollout import bind

OBS_STRIDE = 1

eval_one_episode, eval_one_episode_batch = bind(OBS_STRIDE)
