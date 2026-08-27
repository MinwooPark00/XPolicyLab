"""Rollout loop for FastWAM.

FastWAM conditions on the current frame alone (its adapter overwrites the
stored observation on every `update_obs` and reads only the last one before
`get_action`), so the shared loop's per-step observations would be rendered,
shipped and discarded -- observe once per chunk instead, exactly the reason
GR00T_N17 and Pi_05 moved onto `utils/rollout.py`.
"""

from XPolicyLab.utils.rollout import bind

OBS_STRIDE = None

eval_one_episode, eval_one_episode_batch = bind(OBS_STRIDE)
