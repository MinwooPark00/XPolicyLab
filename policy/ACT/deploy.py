"""Rollout loop for ACT.

ACT reads a single `start_ts` frame -- `EpisodicDataset` never stacks a window
-- so the observations the shared loop takes inside an action chunk are stored
in `ACT.obs_cache` and overwritten unread. The network itself already runs only
every `query_frequency` steps (`= chunk_size` with temporal aggregation off);
what cost a full round trip per step was the loop around it, plus this
adapter's `encode_obs`, which upscales both 320x240 camera frames to 640x480.

With `temporal_agg` on, `query_frequency` is 1 and the ensemble is defined per
step: `model.py` then returns one action per call, the chunk is length 1, and
this loop observes every step regardless of the stride below. The return value
carries that decision, so there is nothing to keep in sync here.
"""

from XPolicyLab.utils.rollout import bind

OBS_STRIDE = None

eval_one_episode, eval_one_episode_batch = bind(OBS_STRIDE)
