"""Rollout loop for GR00T N1.7.

GR00T conditions on the current frame (`delta_indices=[0]` on video and state),
so `model.py` overwrites the stored observation on every `update_obs` and
`get_action` reads only the last -- 39 of every 40 the shared loop took were
rendered and shipped to be discarded. Measured worth of observing once per
chunk instead: **337 -> 119 ms per control step**, 2.9x.

This file used to carry its own copy of the loop to get that. The copy is gone;
`utils/rollout.py` is the same loop with the observation rate as a parameter,
and Pi_05 (which had forked it for the same reason) now shares it.
"""

from XPolicyLab.utils.rollout import bind

OBS_STRIDE = None

eval_one_episode, eval_one_episode_batch = bind(OBS_STRIDE)
