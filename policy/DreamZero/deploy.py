"""Rollout loop for DreamZero.

DreamZero conditions on a WINDOW of frames (video_history=4, trained on
consecutive steps: eval_delta_indices [-3..0]), so the observation must be
refreshed after every executed step to keep the frame buffers contiguous --
obs_stride=1 is byte-for-byte the shared loop the old hand-written copy here
implemented.
"""

from XPolicyLab.utils.rollout import bind

OBS_STRIDE = 1

eval_one_episode, eval_one_episode_batch = bind(OBS_STRIDE)
