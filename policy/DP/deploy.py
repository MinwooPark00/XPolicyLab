"""Rollout loop for Diffusion Policy.

`robot_dp.yaml` sets `n_obs_steps: 1`, and the trained checkpoints carry the
same value in their own cfg -- so `DPRunner.stack_last_n_obs` takes exactly one
frame, the last one appended before `get_action`. The shared loop's inner
`update_obs` calls append five more per six-step chunk (`n_action_steps: 6`),
each one three camera renders and ~690 KB over the websocket, and every one of
them is overwritten before it is read. Observe once per chunk instead.

Restore `OBS_STRIDE = 1` if this policy is ever retrained with `n_obs_steps > 1`
-- then the discarded frames are the observation window and dropping them
changes what the policy sees.
"""

from XPolicyLab.utils.rollout import bind

OBS_STRIDE = None

eval_one_episode, eval_one_episode_batch = bind(OBS_STRIDE)
