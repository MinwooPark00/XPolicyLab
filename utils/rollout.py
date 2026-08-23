"""One rollout loop for every policy, with the observation rate as its only knob.

Every policy in this repo ships an `eval_one_episode` / `eval_one_episode_batch`
pair in its own `deploy.py`, and sixteen of them are byte-identical copies of
the same loop. Two are not: `policy/GR00T_N17/deploy.py` and
`policy/Pi_05/deploy.py` each forked it to make one change -- observe once per
action chunk instead of once per executed step -- and each wrote the same
reasoning into its own comment. Measured there: **337 -> 119 ms per control
step**, 2.9x, on GR00T.

The change is worth that much because the shared loop's inner `update_obs`
calls are not free and, for a large class of policies, not read either:

    while not end:
        obs = get_obs();  update_obs(obs);  actions = get_action()
        for i, action in enumerate(actions):
            take_action(action)
            if end or i + 1 == len(actions): break
            obs = get_obs();  update_obs(obs)      # <-- these
                                                   #     ^ 3 camera renders,
                                                   #       ~690 KB over the
                                                   #       websocket, and the
                                                   #       adapter's own
                                                   #       encode_obs, per step

A policy conditioned on a *window* of frames fills that window here, and must
keep them. A policy conditioned on the current frame alone -- Diffusion Policy
at `n_obs_steps=1`, ACT reading one `start_ts`, GR00T and pi0.5 with
`delta_indices=[0]` -- overwrites its single stored observation on each call
and reads only the last one before `get_action`. For those, every observation
but the last of each chunk is rendered, serialized, shipped and discarded.

So the loop is not duplicated once per policy; it is parameterized:

    obs_stride=1     observe after every executed step. Byte-for-byte the
                     shared loop this replaces, and the safe default: a
                     history-conditioned policy needs exactly this.
    obs_stride=None  observe once per chunk. What GR00T_N17 and Pi_05 forked
                     the file to get.
    obs_stride=k     observe every k steps inside the chunk.

Each `deploy.py` declares its own as a module-level `OBS_STRIDE`, which is also
what MHBench's `scripts/eval_policy_xpolicylab.py` reads to decide whether the
environment may serve a cached render (`MHBenchTaskEnv.get_obs`). The two
layers are complementary and neither is required: the loop skips the round
trip, the env skips the render, and a policy that declares nothing keeps the
behaviour it has today.

Note there is no separate "chunk length" setting, and there does not need to
be. The chunk *is* what `get_action` returned, so a policy whose adapter hands
back one action per call (ACT with temporal aggregation on, where the ensemble
is only defined per step) automatically observes every step even at
`obs_stride=None` -- the inner loop breaks immediately. The information is
carried by the return value, not by a number somebody has to keep in sync.
"""

from __future__ import annotations


def _observe(TASK_ENV, model_client) -> None:
    model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())


def _should_observe(index: int, obs_stride: int | None) -> bool:
    """Whether to re-observe after the `index`-th action of a chunk (0-based).

    `None` never re-observes inside a chunk; `k` does so every k steps, which
    at k=1 is after every one of them.
    """
    if not obs_stride:
        return False
    return (index + 1) % obs_stride == 0


def eval_one_episode(TASK_ENV, model_client, obs_stride: int | None = 1) -> None:
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        _observe(TASK_ENV, model_client)
        actions = model_client.call(func_name="get_action")

        for index, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if TASK_ENV.is_episode_end() or index + 1 == len(actions):
                break

            if _should_observe(index, obs_stride):
                _observe(TASK_ENV, model_client)


def eval_one_episode_batch(TASK_ENV, model_client, obs_stride: int | None = 1) -> None:
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list))
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)

        chunk_size = len(actions[0])
        for index in range(chunk_size):
            TASK_ENV.take_action_batch(
                [env_actions[index] for env_actions in actions], env_idx_list
            )

            if TASK_ENV.is_episode_end() or index + 1 == chunk_size:
                break

            # Environments that finished mid-chunk drop out; the rest keep
            # executing the chunk they were already given.
            running = set(TASK_ENV.get_running_env_idx_list())
            keep = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]
            if len(keep) != len(env_idx_list):
                actions = [actions[i] for i in keep]
                env_idx_list = [env_idx_list[i] for i in keep]

            if _should_observe(index, obs_stride):
                model_client.call(
                    func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list)
                )


def bind(obs_stride: int | None):
    """The two loop entry points a `deploy.py` exports, bound to one stride.

    `deploy.py` is imported by name and called with keywords
    (`eval_module.eval_one_episode(TASK_ENV=..., model_client=...)`), so what
    it exports has to be a two-argument callable -- not this module's
    three-argument one.
    """

    def eval_one_episode_bound(TASK_ENV, model_client):
        return eval_one_episode(TASK_ENV, model_client, obs_stride=obs_stride)

    def eval_one_episode_batch_bound(TASK_ENV, model_client):
        return eval_one_episode_batch(TASK_ENV, model_client, obs_stride=obs_stride)

    return eval_one_episode_bound, eval_one_episode_batch_bound
