"""Rollout loop for Ψ₀.

Ψ₀ conditions on the current frame alone -- `observation_horizon` is 1 and
there is no window to fill -- so it is observed once per action chunk and the
chunk is then executed. `exec_horizon` in `deploy.yml` decides how much of the
30-step chunk (0.6 s at MHBench's 50 Hz) runs before re-observing; the adapter
trims, so the loop just runs what it is handed.
"""

from XPolicyLab.utils.rollout import bind

OBS_STRIDE = None

eval_one_episode, eval_one_episode_batch = bind(OBS_STRIDE)
