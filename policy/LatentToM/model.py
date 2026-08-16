"""XPolicyLab adapter for LatentToM (StanfordMSL, CoRL 2025).

LatentToM is not one policy but two: a `DiffusionSheafSplitPolicy` per arm,
each with its own `SheafObsEncoder` that splits observations into a shared
(third-person) embedding and a private (own-camera + own-pose) embedding.
`arm1` sees `camera_1` (private) + `camera_3` (shared); `arm2` sees `camera_3`
(shared) + `camera_4` (private) -- see
`diffusion_policy/model/vision/sheaf_obs_encoder.py`. Both arms are loaded
from separate checkpoints (`arm1_latest.ckpt` / `arm2_latest.ckpt`, LatentToM's
own save convention) and driven independently at inference time; the only
information that crosses between them is whatever `--type decentralized`-style
training already baked into each arm's weights.

Proprio (43D/arm, 7 URDF joint groups) and action (35D/arm, arm+hand+waist
joint targets + base height + navigate velocity) are joint-space throughout,
matching MHBench's GR00T contract (`configs/gr00t/mhbench_keys.py`) exactly
-- `convert_to_replay_buffer.py` is the training-side source of truth for
this layout. Eval against a real env needs `scripts/mhbench_xpolicylab_env.py`'s
`MHBenchTaskEnv`, which reads proprio from `get_obs()`'s
`mhbench_state.<robot>.joint_pos` and applies joint actions via
`mhbench.g1.actions.joint_target_action_cfg` (Pink IK bypassed).

Status: structurally complete and internally consistent, but not yet run
against a real MHBench checkpoint or a registered `env_cfg_type` -- see
policy/LatentToM/README.md's "What's still open" section before treating this
as verified. Built ahead of the XPolicyLab-format port into the main MHBench
repo, per the architecture decision recorded there.
"""

import sys
from pathlib import Path

import dill
import hydra
import numpy as np
import torch

# Importable root is the parent of the XPolicyLab checkout -- the server
# imports `XPolicyLab.policy.LatentToM.model` -- per XPolicyLab/AGENTS.md.
# parents[1] is this checkout itself, parents[3] is unrelated; both are bugs.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_POLICY_DIR = Path(__file__).resolve().parent
if str(_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(_POLICY_DIR))

from diffusion_policy.env_runner.sheaf_runner import SheafRunner  # noqa: E402
from diffusion_policy.policy.diffusion_sheaf_split_policy import DiffusionSheafSplitPolicy  # noqa: E402
from XPolicyLab.model_template import ModelTemplate  # noqa: E402
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root  # noqa: E402
from XPolicyLab.utils.process_data import (  # noqa: E402
    get_action_dim,
    get_batch_size,
    get_robot_action_dim_info,
)

IMAGE_SIZE = (240, 320)  # (H, W) -- LatentToM's trained resolution.

_PROPRIO_DIM = 43  # mhbench_keys.JOINTS_PER_ROBOT

# One robot's 35D action, GR00T's own column order (mhbench_keys.ACTION_KEYS):
# left_arm(7) right_arm(7) left_hand(7) right_hand(7) waist(3) base_height(1)
# navigate(3). _JOINT_TARGETS matches mhbench.g1.actions.gr00t_joint_names()
# exactly, so no reordering happens before it reaches the articulation.
_ROBOT_ACTION_DIM = 35
_JOINT_TARGETS = slice(0, 31)
_BASE_HEIGHT = slice(31, 32)
_NAVIGATE = slice(32, 35)


def _load_arm_policy(ckpt_path: Path, arm_id: int, device: torch.device, use_ema: bool) -> DiffusionSheafSplitPolicy:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"LatentToM arm{arm_id} checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location=device)
    cfg = payload["cfg"]
    policy: DiffusionSheafSplitPolicy = hydra.utils.instantiate(cfg.policy)

    state_dicts = payload["state_dicts"]
    key_order = [f"arm{arm_id}_ema_model", f"arm{arm_id}_model"] if use_ema else [f"arm{arm_id}_model"]
    for key in key_order:
        if key in state_dicts:
            policy.load_state_dict(state_dicts[key])
            break
    else:
        raise KeyError(f"{ckpt_path} has none of {key_order} -- got {sorted(state_dicts)}")

    policy.to(device)
    policy.eval()
    return policy


def _predict_arm_action(policy: DiffusionSheafSplitPolicy, obs_dict: dict) -> torch.Tensor:
    """Two-stage predict_action, reproducing what
    train_diffusion_sheaf_split_workspace.py's sample-logging block does
    (`process_encoded_obs`).

    `DiffusionSheafSplitPolicy.predict_action()` does NOT fold the sheaf
    embedding into `global_cond` on its own when called in one shot --
    `compute_loss()` (training) does the concat inline, but the inference
    path returns it un-concatenated unless the caller does the same concat
    itself via a second call with `encoded_obs=`. This is a real gap in the
    upstream inference path, not a design choice -- skipping it silently
    trains and evaluates on different input distributions.
    """
    encoded = policy.predict_action(obs_dict, return_embedding_only=True, encoded_obs=None)
    encoded = dict(encoded)
    encoded["global_cond"] = torch.cat([encoded["global_cond"], encoded["sheaf_embedding"]], dim=-1)
    result = policy.predict_action(obs_dict=None, return_embedding_only=False, encoded_obs=encoded)
    return result["action"]  # (B, n_action_steps, _ROBOT_ACTION_DIM)


def _to_chw_float(color: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> (3, h, w) float32 in [0, 1], resized to IMAGE_SIZE.

    `color` arrives already decoded (server-side, see XPolicyLab/AGENTS.md) and
    already RGB -- no BGR conversion belongs here.
    """
    import cv2

    resized = cv2.resize(color, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_AREA)
    return np.transpose(resized, (2, 0, 1)).astype(np.float32) / 255.0


def _robot_proprio_joint43(observation: dict, robot: str) -> np.ndarray:
    """This robot's 43 joint angles [rad], `mhbench.g1.actions.
    mhbench_state_joint_names()` order -- matches convert_to_replay_buffer.py's
    STATE_KEY_NAMES, what a joint-space checkpoint's proprio was trained on.

    Only source is the `mhbench_state` side-channel
    (`scripts/mhbench_xpolicylab_env.py`'s `MHBenchTaskEnv.get_obs()`);
    XPolicyLab's standard obs contract has no joint-angle slot. Zero-fills
    when absent (e.g. `EVAL_ENV_TYPE=debug`'s `TestEnv`) so debug-mode wiring
    checks keep running.
    """
    mhbench_state = observation.get("mhbench_state")
    if mhbench_state is not None and robot in mhbench_state:
        joint_pos = mhbench_state[robot].get("joint_pos")
        if joint_pos is not None:
            return np.asarray(joint_pos, dtype=np.float32)[:_PROPRIO_DIM]
    return np.zeros(_PROPRIO_DIM, dtype=np.float32)


def _encode_arm_obs(observation: dict, arm: int) -> dict:
    """XPolicyLab's standard obs dict -> one arm's LatentToM obs_dict.

    Camera mapping: `cam_left_wrist` -> `camera_1` (arm1/robot_a private),
    `cam_head` -> `camera_3` (shared), `cam_right_wrist` -> `camera_4`
    (arm2/robot_b private). Proprio (`arm{N}_proprio`, 43D) is that robot's
    own joint angles (`_robot_proprio_joint43`), joint-space throughout.
    """
    vision = observation["vision"]
    robot = "robot_a" if arm == 1 else "robot_b"
    proprio = _robot_proprio_joint43(observation, robot)

    obs = {}
    if arm == 1:
        obs["camera_1"] = _to_chw_float(vision["cam_left_wrist"]["color"])
        obs["camera_3"] = _to_chw_float(vision["cam_head"]["color"])
        obs["arm1_proprio"] = proprio
    else:
        obs["camera_3"] = _to_chw_float(vision["cam_head"]["color"])
        obs["camera_4"] = _to_chw_float(vision["cam_right_wrist"]["color"])
        obs["arm2_proprio"] = proprio
    return obs


def _unpack_robot_action(action: np.ndarray) -> dict:
    """One robot's raw 35D model output -> `mhbench_xpolicylab_env.py`'s
    `_robot_action_35d` input: `{joint_targets(31), base_vel(3), height(1)}`.
    Column order is `_JOINT_TARGETS`/`_BASE_HEIGHT`/`_NAVIGATE` (module top);
    `joint_targets` is columns 0:31 as-is, already `gr00t_joint_names()` order.
    """
    return {
        "joint_targets": np.asarray(action[_JOINT_TARGETS], dtype=np.float32),
        "base_vel": np.asarray(action[_NAVIGATE], dtype=np.float32),
        "height": np.asarray(action[_BASE_HEIGHT], dtype=np.float32),
    }


def _pack_dual_arm_action(arm1_action: np.ndarray, arm2_action: np.ndarray) -> dict:
    """Repack one timestep's arm1(=robot_a)/arm2(=robot_b) model output.

    Joint-space, so nothing maps onto XPolicyLab's dual-arm `ee` contract
    (`left_ee_pose`/etc.) even as a stand-in -- a joint-target vector isn't a
    pose without forward kinematics, and solving one just to fill an unused
    key would put a solver back between the prediction and the joint it
    drives. Everything lives under `mhbench_raw_action`;
    `MHBenchTaskEnv.take_action` is the only reader.
    """
    return {
        "mhbench_raw_action": {
            "robot_a": _unpack_robot_action(arm1_action),
            "robot_b": _unpack_robot_action(arm2_action),
        }
    }


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]
        if self.action_type != "joint":
            raise NotImplementedError(
                "LatentToM is joint-space only -- set deploy.yml's action_type: joint "
                f"(got {self.action_type!r}). See README.md's Model contract details."
            )

        self.action_dim = get_action_dim(self.env_cfg_type)
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.batch_size = get_batch_size(self.env_cfg_type)
        num_arms = len(self.robot_action_dim_info["arm_dim"])
        if num_arms != 2:
            raise NotImplementedError(
                f"LatentToM is a two-arm sheaf-split policy (arm1/arm2); env_cfg_type="
                f"{self.env_cfg_type!r} registers {num_arms} arm(s)."
            )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        policy_dir = Path(__file__).resolve().parent
        ckpt_root = resolve_checkpoint_root(
            model_cfg, policy_dir / "checkpoints", policy_dir=policy_dir, must_exist=True
        )
        use_ema = bool(model_cfg.get("use_ema", True))
        self.arm1_policy = _load_arm_policy(ckpt_root / "checkpoints" / "arm1_latest.ckpt", 1, self.device, use_ema)
        self.arm2_policy = _load_arm_policy(ckpt_root / "checkpoints" / "arm2_latest.ckpt", 2, self.device, use_ema)

        for arm_id, policy in ((1, self.arm1_policy), (2, self.arm2_policy)):
            if policy.action_dim != _ROBOT_ACTION_DIM:
                raise ValueError(
                    f"arm{arm_id} checkpoint has action_dim={policy.action_dim}, expected "
                    f"{_ROBOT_ACTION_DIM} -- this checkpoint wasn't trained on the "
                    "joint-space GR00T-convention data."
                )

        n_obs_steps = self.arm1_policy.n_obs_steps
        self.n_action_steps = self.arm1_policy.n_action_steps
        self.runner = SheafRunner(n_obs_steps=n_obs_steps)
        self._env_idx_list = None

        print(f"[LatentToM] loaded arm1/arm2 from {ckpt_root} (device={self.device}, use_ema={use_ema})")

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        env_idx_list = [obs["env_idx"] for obs in obs_list]
        arm1_obs_list = [_encode_arm_obs(obs, arm=1) for obs in obs_list]
        arm2_obs_list = [_encode_arm_obs(obs, arm=2) for obs in obs_list]
        self.runner.update_obs(arm1_obs_list, arm2_obs_list, env_idx_list)
        self._env_idx_list = env_idx_list

    def get_action(self):
        if not self._env_idx_list:
            raise RuntimeError("get_action() called before update_obs().")
        return self.get_action_batch(env_idx_list=[self._env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = self._env_idx_list
        if not env_idx_list:
            raise RuntimeError("get_action_batch() called before update_obs_batch().")

        arm1_obs, arm2_obs = self.runner.get_arm_obs_batch(env_idx_list, self.device)
        with torch.no_grad():
            arm1_action = _predict_arm_action(self.arm1_policy, arm1_obs).cpu().numpy()
            arm2_action = _predict_arm_action(self.arm2_policy, arm2_obs).cpu().numpy()

        num_envs, chunk_len = arm1_action.shape[0], arm1_action.shape[1]
        action_batch = []
        for b in range(num_envs):
            chunk = [_pack_dual_arm_action(arm1_action[b, t], arm2_action[b, t]) for t in range(chunk_len)]
            action_batch.append(chunk)
        return action_batch

    def reset(self):
        self.runner.reset()
        self._env_idx_list = None
