"""Rollout loop for GauDP.

`OBS_STRIDE = 1` is the shared loop unchanged -- observe after every executed
step -- and for this policy that is not a default left in place but a
requirement. `GauDPPolicy` now conditions on the current observation only
(`n_obs_steps=1`), while `GauDPRunner` keeps the checkpoint-configured number
of frames for compatibility with explicitly trained multi-observation models.
The rollout still observes after every executed action so each new prediction
uses a fresh image rather than a cached render.

The same holds in the environment: `scripts/eval_policy_xpolicylab.py` reads
this value to decide whether `MHBenchTaskEnv.get_obs` may re-serve a cached
render, and 1 keeps every step's render honest.
"""

from XPolicyLab.utils.rollout import bind

OBS_STRIDE = 1

eval_one_episode, eval_one_episode_batch = bind(OBS_STRIDE)
