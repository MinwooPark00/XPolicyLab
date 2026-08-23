"""Rollout loop for GauDP.

`OBS_STRIDE = 1` is the shared loop unchanged -- observe after every executed
step -- and for this policy that is not a default left in place but a
requirement. `GauDPPolicy` conditions on a window: `n_obs_steps` defaults to 3,
`GauDPRunner` keeps that many frames per environment in a `deque(maxlen=...)`,
and `batch()` stacks the whole window into the observation encoder. The frames
the loop collects inside a chunk are that window. Dropping them (`None`, or any
stride above 1) would leave the policy re-reading one frame three times and
seeing no motion at all, which is a different input distribution from the one it
was trained on rather than a saving.

The same holds in the environment: `scripts/eval_policy_xpolicylab.py` reads
this value to decide whether `MHBenchTaskEnv.get_obs` may re-serve a cached
render, and 1 keeps every step's render honest.
"""

from XPolicyLab.utils.rollout import bind

OBS_STRIDE = 1

eval_one_episode, eval_one_episode_batch = bind(OBS_STRIDE)
