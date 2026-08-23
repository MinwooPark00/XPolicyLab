import torch
import yaml
import cv2
import numpy as np
import hydra
import dill
import sys, os
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)

from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.env_runner.dp_runner import DPRunner
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state, get_robot_action_dim_info
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root, build_run_dir_name

# MHBenchTaskEnv.get_obs (scripts/mhbench_xpolicylab_env.py) maps each robot's
# own ego camera onto these XPolicyLab-generic slot names -- same mapping
# ACT/model.py and GR00T_N17/model.py's MHBENCH_CAMERA_SLOT use, since it
# comes from the env, not the policy.
MHBENCH_CAMERA_SLOT = {"robot_a": "cam_left_wrist", "robot_b": "cam_right_wrist"}


def _prep_camera(color: np.ndarray) -> np.ndarray:
    """XPolicyLab vision obs (HWC uint8) -> DP's CHW float32 input, 320x240."""
    img = np.moveaxis(color, -1, 0) / 255.0
    return np.transpose(
        cv2.resize(np.transpose(img, (1, 2, 0)), (320, 240), interpolation=cv2.INTER_AREA), (2, 0, 1)
    )


DDPM_ONLY_CONFIG_KEYS = ("variance_type",)
"""DDPM settings that have no DDIM counterpart and must not be forwarded."""


def configure_sampler(policy, scheduler_name, num_inference_steps):
    """Choose the denoiser used at *evaluation*, leaving the weights alone.

    The trained checkpoints sample with `DDPMScheduler` at
    `num_inference_steps: 100` -- read out of `600.ckpt`'s own cfg, not just
    `robot_dp.yaml` -- which is 100 U-Net forwards for every `n_action_steps`
    (6) environment steps, i.e. ~16,700 per 1000-step episode. DDPM cannot be
    shortened: its reverse process is defined on the same grid it was trained
    on. DDIM can, by subsampling that grid, and it is what the Diffusion Policy
    release itself uses for fast inference -- with the identical betas
    (`squaredcos_cap_v2`, 1e-4..0.02, 100 train steps), `prediction_type` and
    `clip_sample`, so the model is unchanged and only the solver differs.

    That is still a different sampler, and eta=0 DDIM is deterministic where
    DDPM's `variance_type: fixed_small` is not -- so this is opt-in and has to
    be justified by a success-rate A/B on real episodes, not by the speedup.

    `policy.kwargs` is forwarded to `scheduler.step` by
    `DiffusionUnetImagePolicy.conditional_sample`, and DDIM's `step` takes no
    `**kwargs` (diffusers 0.11.1) where DDPM's does -- so anything the DDPM
    path tolerated is filtered here rather than raising inside the rollout.
    """
    if scheduler_name is None and num_inference_steps is None:
        return policy

    name = (scheduler_name or "ddpm").lower()
    if name not in ("ddpm", "ddim"):
        raise ValueError(f"inference_scheduler must be 'ddpm' or 'ddim', got {scheduler_name!r}")

    if name == "ddim" and not isinstance(policy.noise_scheduler, DDIMScheduler):
        import inspect

        trained = dict(policy.noise_scheduler.config)
        carried = {
            key: value
            for key, value in trained.items()
            if key not in DDPM_ONLY_CONFIG_KEYS and not key.startswith("_")
        }
        # The Diffusion Policy release's own DDIM settings; neither exists on
        # the DDPM config this is derived from.
        carried.setdefault("set_alpha_to_one", True)
        carried.setdefault("steps_offset", 0)
        policy.noise_scheduler = DDIMScheduler(**carried)

        allowed = set(inspect.signature(DDIMScheduler.step).parameters)
        dropped = {k: v for k, v in policy.kwargs.items() if k not in allowed}
        if dropped:
            print(f"[DP][sampler] dropping step kwargs DDIM does not take: {sorted(dropped)}")
            policy.kwargs = {k: v for k, v in policy.kwargs.items() if k in allowed}

    if num_inference_steps is not None:
        policy.num_inference_steps = int(num_inference_steps)

    print(
        f"[DP][sampler] {type(policy.noise_scheduler).__name__} "
        f"num_inference_steps={policy.num_inference_steps} "
        f"(trained: {policy.noise_scheduler.config.num_train_timesteps} steps)"
    )
    return policy


def _pack_single_robot_action(flat_action: np.ndarray) -> dict:
    """One robot's 35D DP action -> MHBenchTaskEnv.take_action's
    {joint_targets, base_vel, height}. Same mhbench_keys.ACTION_KEYS layout
    ACT's model.py packs (scripts/data_convertion.py's `keys_for` is the same
    function behind both --format act and --format diffusion_policy, so the
    two adapters see the same 35D order)."""
    assert flat_action.shape[-1] == 35, f"expected 35D per-robot action, got {flat_action.shape}"
    return {
        "joint_targets": flat_action[0:31],
        "height": flat_action[31:32],
        "base_vel": flat_action[32:35],
    }


def _pack_dual_arm_action(flat_action: np.ndarray) -> dict:
    """This adapter's 70D flat action -> one dict per robot."""
    assert flat_action.shape[-1] == 70, f"expected 70D dual-robot action, got {flat_action.shape}"
    return {
        robot: _pack_single_robot_action(flat_action[i * 35 : (i + 1) * 35])
        for i, robot in enumerate(("robot_a", "robot_b"))
    }


def fitted_obs_bounds(policy):
    """The per-dimension [min, max] `agent_pos` the checkpoint's normalizer was
    fitted on.

    `clip_sample` bounds what comes *out* of the sampler; nothing bounds what
    goes *in*, and `mode: limits` divides each observation dimension by the
    range the demonstrations happened to show it over. A joint the operator
    never moved has almost no range -- in MHBench's framehang set, `robot_a`'s
    right hand and `robot_b`'s left hand are idle, and one of their finger
    joints spans 0.0007 rad across all 28,549 recorded steps. That is above
    `range_eps` (1e-4), so it is not treated as constant; it is given a gain of
    2852 normalized units per radian, and a 0.2 rad move hands the encoder 570
    where it was trained on [-1, 1].

    That is the return half of a loop, not its start. Measured on framehang's
    centralized rollout, the commanded action leaves its own range first and
    the observation follows: normalized `agent_pos` reaches 1.7 by step 6 and
    9.7 by step 30 -- the step the action explodes -- then 2463 by step 42,
    once a 1.72 m hip-height command has actually moved the robot. Clamping
    the action instead keeps the observation inside the fitted range for a
    whole 1000-step episode (max 1.01). So either clamp cuts the loop, and
    both were measured to: 7 falls in 10 episodes become 0 with `action_clip`
    alone, and 0 with `obs_clip` alone.
    """
    stats = policy.normalizer.params_dict["agent_pos"]["input_stats"]
    return (
        stats["min"].detach().cpu().numpy().astype(np.float32),
        stats["max"].detach().cpu().numpy().astype(np.float32),
    )


def fitted_action_bounds(policy):
    """The per-dimension [min, max] the checkpoint's own normalizer was fitted
    on, as numpy.

    Why an evaluation would want them: `LinearNormalizer` in `limits` mode
    gives a dimension whose training range is zero (`base_height_command` is a
    constant 0.72 m in every MHBench demonstration, and one hand per robot
    never moves) `scale = 1`, `offset = -min` -- an identity. The sampler's
    `clip_sample` then bounds that dimension's *raw* command to `value +/- 1.0`,
    so an out-of-distribution sample commands 1.72 m of hip height, or a
    full-scale base velocity, in units the robot takes literally. Those are
    the dimensions the balance controller reads.
    """
    stats = policy.normalizer.params_dict["action"]["input_stats"]
    return (
        stats["min"].detach().cpu().numpy().astype(np.float32),
        stats["max"].detach().cpu().numpy().astype(np.float32),
    )


class Model(ModelTemplate):

    def __init__(self, model_cfg):
        # Fallback only. The authority on how many frames a policy conditions
        # on, and how many it emits, is the served checkpoint -- `_load_policy`
        # overwrites both from `policy.n_obs_steps` / `policy.n_action_steps`.
        # Reading `robot_dp.yaml` alone would serve every previously trained
        # checkpoint with whatever window the *current* config happens to say,
        # which is a silent train/eval mismatch the moment the file is edited
        # (it was: n_obs_steps went 3 -> 1 in 0db9c1c).
        load_config_path = os.path.join(parent_dir, 'diffusion_policy/config/robot_dp.yaml')
        with open(load_config_path, "r", encoding="utf-8") as f:
            model_training_config = yaml.safe_load(f)
        self.n_obs_steps = model_training_config['n_obs_steps']
        self.n_action_steps = model_training_config['n_action_steps']
        self.action_type = model_cfg['action_type']
        self._dump_calls = 0
        self._dump_episode = -1
        self._dump_last_obs = None
        # Evaluation-time sampler. Both default to None = whatever the
        # checkpoint was trained and saved with (see configure_sampler).
        self._inference_scheduler = model_cfg.get('inference_scheduler')
        self._num_inference_steps = model_cfg.get('num_inference_steps')
        # 'fitted' clamps every commanded dimension to the range the
        # checkpoint's normalizer was fitted on (see
        # fitted_action_bounds). Default None = today's behaviour.
        self._action_clip = model_cfg.get('action_clip')
        # 'fitted' clamps agent_pos to the range the checkpoint's normalizer
        # was fitted on, before the policy sees it (see fitted_obs_bounds).
        self._obs_clip = model_cfg.get('obs_clip')
        self._obs_bounds_of = {}
        # Keyed by policy, not stored once: the two decentralized checkpoints
        # fit their own 35D ranges and they are NOT the same -- robot_a's idle
        # hand is the right one, robot_b's the left -- so one robot's bounds
        # would clamp the wrong seven columns of the other's action.
        self._bounds_of = {}

        self._mhbench_decentralized = (
            str(model_cfg.get("bench_name") or "") == "mhbench"
            and model_cfg.get("env_cfg_type") == "unitree_g1x2_decentralized"
        )
        if self._mhbench_decentralized:
            self._init_mhbench_decentralized(model_cfg)
            return

        self._mhbench_dual_robot = False  # set per-batch in update_obs_batch
        self.model = self.get_model(model_cfg=model_cfg)
        self.runner = DPRunner(n_obs_steps=self.n_obs_steps, n_action_steps=self.n_action_steps)
        try:
            self.robot_action_dim_info = get_robot_action_dim_info(model_cfg['env_cfg_type'])
        except FileNotFoundError:
            # MHBench env_cfg_types (unitree_g1x2_centralized, mhbench_cocarry,
            # ...) have no env_cfg/<type>.yml -- their scene lives in MHBench's
            # own Isaac Lab env_cfg tree, not XPolicyLab's. Fine here: the
            # mhbench dual-robot path never touches this, since it packs
            # agent_pos from mhbench_state and the action from
            # mhbench_raw_action directly instead of going through
            # pack_robot_state/unpack_robot_state.
            self.robot_action_dim_info = None
        self._latest_env_idx_list = None

    def _init_mhbench_decentralized(self, model_cfg):
        """Two single-robot DP checkpoints served from one process.

        Trained separately as mhbench-<task>_robot_a/-unitree_g1x2_decentralized
        and ..._robot_b (baselines/README.md's DP Train section), each with
        one camera stream (train.sh's num_cameras=1 for this env_cfg_type).
        Mirrors ACT's/GR00T_N17's mhbench decentralized mode: one server, one
        policy instance per robot, combined into the same `mhbench_raw_action`
        shape `MHBenchTaskEnv.take_action` expects. `ckpt_name` here is the
        task (e.g. "cocarry"), not a run dir.
        """
        task = str(model_cfg.get('ckpt_name') or '').strip()
        if not task:
            raise ValueError("mhbench decentralized eval needs ckpt_name=<task> (e.g. cocarry)")

        checkpoints_dir = os.path.join(parent_dir, "checkpoints")
        self._sub_policies = {}
        self._sub_runners = {}
        for robot, camera_name in MHBENCH_CAMERA_SLOT.items():
            explicit = model_cfg.get(f'model_dir_{robot}')
            if explicit:
                ckpt_dir = explicit
            else:
                run_cfg = dict(model_cfg)
                run_cfg['ckpt_name'] = f"{task}_{robot}"
                run_name = build_run_dir_name(run_cfg)
                if run_name is None:
                    raise ValueError("bench_name/ckpt_name/env_cfg_type/action_type/seed required to name the run dir")
                ckpt_dir = os.path.join(checkpoints_dir, run_name)
            if not os.path.isdir(ckpt_dir):
                raise FileNotFoundError(f"{robot} checkpoint not found: {ckpt_dir}")

            self._sub_policies[robot] = self._load_policy(ckpt_dir, model_cfg.get('checkpoint_num', 'latest'))
            # After _load_policy, so the runner is sized by the checkpoint.
            self._sub_runners[robot] = DPRunner(n_obs_steps=self.n_obs_steps, n_action_steps=self.n_action_steps)
            print(f"[DP][mhbench] {robot}: {ckpt_dir} (camera={camera_name})")

    def _load_policy(self, ckpt_dir, checkpoint_num):
        ckpt_file = self._resolve_checkpoint_file(ckpt_dir, checkpoint_num)

        # load checkpoint and workspace
        payload = torch.load(open(ckpt_file, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=None)
        workspace: RobotWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        configure_sampler(policy, self._inference_scheduler, self._num_inference_steps)

        # The checkpoint decides the observation window and the chunk length.
        # `DPRunner` stacks exactly `n_obs_steps` frames, so a mismatch here
        # feeds the policy a window it was never trained on -- and at
        # n_obs_steps > 1 it also means the rollout loop must observe every
        # step (deploy.py's OBS_STRIDE, or --obs_stride 1), or the stack is
        # padded with copies of one frame instead of a real history.
        served_obs = int(getattr(policy, "n_obs_steps", self.n_obs_steps))
        served_action = int(getattr(policy, "n_action_steps", self.n_action_steps))
        if (served_obs, served_action) != (self.n_obs_steps, self.n_action_steps):
            print(f"[DP] checkpoint was trained with n_obs_steps={served_obs} "
                  f"n_action_steps={served_action}; robot_dp.yaml says "
                  f"{self.n_obs_steps}/{self.n_action_steps} -- serving the checkpoint's")
        self.n_obs_steps, self.n_action_steps = served_obs, served_action
        if served_obs > 1:
            print(f"[DP] n_obs_steps={served_obs}: the rollout loop must observe every step "
                  f"(EVAL_OBS_STRIDE=1), or the observation window is padded, not real")

        device = torch.device("cuda:0")
        policy.to(device)
        policy.eval()

        if self._action_clip is not None:
            if str(self._action_clip).lower() not in ("fitted", "true", "1"):
                raise ValueError(f"action_clip must be 'fitted' or null, got {self._action_clip!r}")
            low, high = fitted_action_bounds(policy)
            print(f"[DP][action_clip] clamping to the fitted action range, "
                  f"{int(((high - low) < 1e-4).sum())} of {low.size} dims have zero range")
            self._bounds_of[id(policy)] = (low, high)

        if self._obs_clip is not None:
            if str(self._obs_clip).lower() not in ("fitted", "true", "1"):
                raise ValueError(f"obs_clip must be 'fitted' or null, got {self._obs_clip!r}")
            low, high = fitted_obs_bounds(policy)
            gain = 2.0 / np.maximum(high - low, 1e-9)
            print(f"[DP][obs_clip] clamping agent_pos to the fitted range; "
                  f"largest normalization gain {gain.max():.0f} per unit on dim {int(gain.argmax())}")
            self._obs_bounds_of[id(policy)] = (low, high)

        return policy

    def get_model(self, model_cfg):
        ckpt_dir = resolve_checkpoint_root(
            model_cfg,
            os.path.join(parent_dir, "checkpoints"),
            policy_dir=parent_dir,
            must_exist=False,
        )
        return self._load_policy(ckpt_dir, model_cfg.get('checkpoint_num', 'latest'))

    def _resolve_checkpoint_file(self, ckpt_dir, checkpoint_num):
        ckpt_dir = os.fspath(ckpt_dir)
        checkpoint_num = "latest" if checkpoint_num is None else str(checkpoint_num)

        if checkpoint_num.lower() not in {"", "latest", "none"}:
            ckpt_file = os.path.join(ckpt_dir, f"{checkpoint_num}.ckpt")
            if not os.path.isfile(ckpt_file):
                raise FileNotFoundError(f"DP checkpoint not found: {ckpt_file}")
            return ckpt_file

        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"DP checkpoint directory not found: {ckpt_dir}")

        candidates = []
        for name in os.listdir(ckpt_dir):
            if not name.endswith(".ckpt"):
                continue
            stem = name[:-5]
            if stem.isdigit():
                candidates.append((int(stem), os.path.join(ckpt_dir, name)))

        if not candidates:
            raise FileNotFoundError(f"No numeric DP checkpoints found under: {ckpt_dir}")

        return max(candidates, key=lambda item: item[0])[1]

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        env_idx_list = [obs["env_idx"] for obs in obs_list]

        if self._mhbench_decentralized:
            merged = {}
            for robot, runner in self._sub_runners.items():
                encoded = [self._encode_mhbench_robot_obs(obs, robot) for obs in obs_list]
                for item in encoded:
                    item["agent_pos"] = self._clip_obs(item["agent_pos"], self._sub_policies[robot])
                if encoded:
                    merged.update({f"{robot}_{k}": v for k, v in encoded[0].items()})
                runner.update_obs(encoded, env_idx_list)
            self._dump_last_obs = merged or None
            self._latest_env_idx_list = env_idx_list
            return

        self._mhbench_dual_robot = bool(obs_list) and "mhbench_state" in obs_list[0]
        encoded_list = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info, self._mhbench_dual_robot)
            for obs in obs_list
        ]
        for encoded in encoded_list:
            encoded["agent_pos"] = self._clip_obs(encoded["agent_pos"], self.model)
        self._dump_last_obs = encoded_list[0] if encoded_list else None
        self.runner.update_obs(encoded_list, env_idx_list)
        self._latest_env_idx_list = env_idx_list

    def _encode_mhbench_robot_obs(self, observation, robot):
        """One robot's single-camera decentralized obs: head_cam is that
        robot's own ego view -- no left_cam/right_cam, matching what its
        checkpoint was trained with (train.sh's num_cameras=1)."""
        camera_name = MHBENCH_CAMERA_SLOT[robot]
        head_img = _prep_camera(observation["vision"][camera_name]["color"])
        agent_pos = np.asarray(observation["mhbench_state"][robot]["joint_pos"], dtype=np.float32)
        return dict(head_cam=head_img, agent_pos=agent_pos)

    def get_action(self):
        if not self._latest_env_idx_list:
            raise RuntimeError("get_action() called before update_obs().")

        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]])

        return action_list[0]

    def _clip_obs(self, agent_pos, policy):
        """Clamp agent_pos to the range `policy`'s normalizer was fitted on.

        The return half of the loop `_clip_action` cuts on the way out; see
        `fitted_obs_bounds`. Either one alone stops the rollout diverging, so
        turning both on is belt and braces rather than two fixes.
        """
        bounds = self._obs_bounds_of.get(id(policy))
        if bounds is None:
            return agent_pos
        low, high = bounds
        if low.size != agent_pos.shape[-1]:
            raise ValueError(
                f"obs_clip bounds are {low.size}D but agent_pos is {agent_pos.shape[-1]}D"
            )
        return np.clip(agent_pos, low, high)

    def _clip_action(self, actions, policy):
        """Clamp a (..., D) action to the range `policy`'s own normalizer was
        fitted on, when `action_clip` asks for it."""
        bounds = self._bounds_of.get(id(policy))
        if bounds is None:
            return actions
        low, high = bounds
        if low.size != actions.shape[-1]:
            raise ValueError(
                f"action_clip bounds are {low.size}D but the action is {actions.shape[-1]}D"
            )
        return np.clip(actions, low, high)

    # ------------------------------------------------------------------
    # Diagnostics. MHBENCH_DP_DUMP=<dir> makes the server write, per
    # get_action call, exactly what it was handed and exactly what it
    # answered -- the numbers a rollout failure has to be explained by and
    # that no log holds today. Off unless the variable is set; nothing on
    # the normal path reads it. MHBENCH_DP_DUMP_LIMIT caps the files per
    # episode (default 400 calls = 2400 control steps).
    # ------------------------------------------------------------------
    def _dump_dir(self):
        root = os.environ.get("MHBENCH_DP_DUMP")
        if not root:
            return None
        os.makedirs(root, exist_ok=True)
        return root

    def _dump(self, obs_encoded, actions, action_pred=None):
        root = self._dump_dir()
        if root is None:
            return
        limit = int(os.environ.get("MHBENCH_DP_DUMP_LIMIT", "400"))
        if self._dump_calls >= limit:
            return
        payload = {"call": np.int64(self._dump_calls), "episode": np.int64(self._dump_episode)}
        for key, value in obs_encoded.items():
            arr = np.asarray(value)
            if arr.ndim == 3 and arr.shape[0] == 3:           # CHW float image
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            payload[f"obs_{key}"] = arr
        payload["action"] = np.asarray(actions, dtype=np.float32)
        if action_pred is not None:
            payload["action_pred"] = np.asarray(action_pred, dtype=np.float32)
        np.savez_compressed(
            os.path.join(root, f"ep{self._dump_episode:03d}_call{self._dump_calls:04d}.npz"),
            **payload,
        )
        self._dump_calls += 1

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        if not env_idx_list:
            raise RuntimeError("get_action_batch() called before update_obs_batch().")

        if self._mhbench_decentralized:
            per_robot = {
                robot: self._clip_action(
                    runner.get_action(self._sub_policies[robot], env_idx_list),
                    self._sub_policies[robot],
                )
                for robot, runner in self._sub_runners.items()
            }  # each: (len(env_idx_list), n_action_steps, 35)
            if self._dump_last_obs is not None:
                self._dump(
                    self._dump_last_obs,
                    np.concatenate([per_robot["robot_a"][0], per_robot["robot_b"][0]], axis=-1),
                )
            steps = min(arr.shape[1] for arr in per_robot.values())
            return [
                [
                    {
                        "mhbench_raw_action": {
                            robot: _pack_single_robot_action(per_robot[robot][env_i, t])
                            for robot in per_robot
                        }
                    }
                    for t in range(steps)
                ]
                for env_i in range(len(env_idx_list))
            ]

        actions = self.runner.get_action(self.model, env_idx_list)  # (len(env_idx_list), n_action_steps, action_dim)
        actions = self._clip_action(actions, self.model)
        if self._dump_last_obs is not None:
            self._dump(self._dump_last_obs, actions[0])

        if self._mhbench_dual_robot:
            return [
                [{"mhbench_raw_action": _pack_dual_arm_action(actions[i, t])} for t in range(actions.shape[1])]
                for i in range(len(env_idx_list))
            ]

        return [
            unpack_robot_state(actions[i], self.action_type, self.robot_action_dim_info, source_type='obs')
            for i in range(len(env_idx_list))
        ]

    def reset(self):
        self._dump_episode += 1
        self._dump_calls = 0
        if self._mhbench_decentralized:
            for runner in self._sub_runners.values():
                runner.reset_obs()
            self._latest_env_idx_list = None
            return
        self.runner.reset_obs()
        self._latest_env_idx_list = None

def encode_obs(observation, action_type, robot_action_dim_info, mhbench_dual_robot=False):
    left_cam = _prep_camera(observation["vision"]["cam_left_wrist"]["color"])
    right_cam = _prep_camera(observation["vision"]["cam_right_wrist"]["color"])

    if mhbench_dual_robot:
        # MHBench's two-full-humanoid state doesn't fit XPolicyLab's generic
        # single-bimanual-robot obs['state'] schema (pack_robot_state only
        # knows arm_dim+ee_dim=70, but this robot's qpos is 86) -- same reason
        # ACT/model.py's encode_obs branches here. Only the two ego cameras
        # are used (num_cameras=2, no head_cam) -- cam_head is the scene
        # camera, never in the training zarr or shape_meta for this robot;
        # see RobotImageDataset's camera_map-derived cam_obs_names.
        mhbench_state = observation["mhbench_state"]
        agent_pos = np.concatenate([
            np.asarray(mhbench_state["robot_a"]["joint_pos"], dtype=np.float32),
            np.asarray(mhbench_state["robot_b"]["joint_pos"], dtype=np.float32),
        ])
        return dict(left_cam=left_cam, right_cam=right_cam, agent_pos=agent_pos)

    head_img = _prep_camera(observation["vision"]["cam_head"]["color"])
    agent_pos = pack_robot_state(observation, action_type, robot_action_dim_info, source_type='obs')
    return dict(head_cam=head_img, left_cam=left_cam, right_cam=right_cam, agent_pos=agent_pos)
