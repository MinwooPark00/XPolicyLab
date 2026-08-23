"""Rollout loop for pi0.5.

pi0.5 conditions on the current frame alone -- there is no observation window to
fill -- so it is observed once per action chunk and the whole chunk is then
executed. The shared XPolicyLab loop calls `update_obs` on *every* step inside
the chunk as well, which for an ACT or Diffusion Policy checkpoint is how the
window gets its history, but here ships two camera frames over the websocket at
every step for a result that is thrown away. `GR00T_N17/deploy.py` diverges for
the same reason (measured there at 337 -> 119 ms per control step).

`exec_horizon` in `deploy.yml` decides how much of the chunk is executed before
re-observing; the adapter trims the chunk, so this loop just runs what it is
given.
"""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        actions = model_client.call(func_name="get_action")

        for action in actions:
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list))
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)

        for action_idx in range(len(actions[0])):
            TASK_ENV.take_action_batch([env_actions[action_idx] for env_actions in actions], env_idx_list)
            if TASK_ENV.is_episode_end():
                break

            # Environments that finished mid-chunk drop out; the rest keep
            # executing the chunk they were already given.
            running = set(TASK_ENV.get_running_env_idx_list())
            keep = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]
            if len(keep) != len(env_idx_list):
                actions = [actions[i] for i in keep]
                env_idx_list = [env_idx_list[i] for i in keep]
