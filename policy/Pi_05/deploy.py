"""Rollout loop for pi0.5.

pi0.5 conditions on the current frame alone -- there is no observation window to
fill -- so it is observed once per action chunk and the whole chunk is then
executed. The shared loop called `update_obs` on every step inside the chunk as
well, which for an ACT or Diffusion Policy checkpoint is how the window gets its
history, but here shipped two camera frames over the websocket at every step for
a result that was thrown away.

`exec_horizon` in `deploy.yml` decides how much of the chunk is executed before
re-observing; the adapter trims the chunk, so the loop just runs what it is
given.
"""

from XPolicyLab.utils.rollout import bind

OBS_STRIDE = None

eval_one_episode, eval_one_episode_batch = bind(OBS_STRIDE)
