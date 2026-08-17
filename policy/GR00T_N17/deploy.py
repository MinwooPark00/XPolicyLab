"""GR00T N1.7's episode loop.

Deliberately NOT `policy/demo_policy/deploy.py`'s loop, in one respect: the
shared version fetches an observation and ships it to the server after *every*
executed step, but `model.py`'s `update_obs_batch` only stores the latest one
and `get_action` reads only that. With `exec_horizon` 40 that is 39 of every 40
observations rendered, serialised (three 320x240 RGB frames, ~690 KB) and sent
over the websocket to be overwritten unread. Reading the cameras is 85 ms of a
110 ms control step, so the waste is most of eval's wall clock: dropping it
takes a step to ~58 ms.

**This is safe only for a policy that conditions on the current frame.**
GR00T N1.7 here does (its video modality is a single frame, `delta_indices`
[0]). A policy that stacks past observations -- diffusion policy, ACT with an
observation window -- MUST keep the shared loop, because for it the
intermediate observations are inputs, not redundancy. Do not copy this file
into such an adapter.

The video is unaffected: `MHBenchTaskEnv` records one scene-camera frame per
executed step from inside `take_action`, not from `get_obs`, precisely so that
how often a policy observes cannot change what the recording looks like.
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
            running = set(TASK_ENV.get_running_env_idx_list())
            keep = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]
            if len(keep) != len(env_idx_list):
                actions = [actions[i] for i in keep]
                env_idx_list = [env_idx_list[i] for i in keep]
