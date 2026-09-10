"""Rollout loop for LingBot_VA.

Unlike the current-frame policies, this one consumes every executed frame: the
closed loop feeds the frames the robot actually passed through back into the
server as the next chunk's KV cache (`_mhbench_commit`, the official RoboTwin
eval's compute_kv_cache step), so an observation skipped here is a gap in the
model's own video history. OBS_STRIDE = 1 keeps the shared loop rendering each
step -- the same choice DP and ACT make for their frame windows.
"""

from XPolicyLab.utils.rollout import bind

OBS_STRIDE = 1

eval_one_episode, eval_one_episode_batch = bind(OBS_STRIDE)
