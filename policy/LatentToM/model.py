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

Proprio (21D/arm: pelvis pose + left eef pose + right eef pose) and action
(22D/arm quat-native, or 26D/arm if `rotation_rep: rot6d` -- see
`convert_to_replay_buffer.py`, the offline converter that's the
training-side source of truth for this layout) match what that converter
produces from MHBench's own data, not LatentToM's original xArm 10D
convention.

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

from diffusion_policy.common.rotation_conversion import action_dim_for, rot6d_to_quat_xyzw  # noqa: E402
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

# Compressed action column layout -- must match
# diffusion_policy/common/rotation_conversion.py's quat-native (22D) and
# rot6d-native (26D) layouts exactly, since a checkpoint trained under one
# is only decodable under the same one.
_QUAT_LEFT_POS, _QUAT_LEFT_QUAT = slice(0, 3), slice(3, 7)
_QUAT_RIGHT_POS, _QUAT_RIGHT_QUAT = slice(7, 10), slice(10, 14)
_QUAT_REST = slice(14, 22)  # hands(4) + base_vel(3) + height(1)

_R6_LEFT_POS, _R6_LEFT_ROT = slice(0, 3), slice(3, 9)
_R6_RIGHT_POS, _R6_RIGHT_ROT = slice(9, 12), slice(12, 18)
_R6_REST = slice(18, 26)


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
    return result["action"]  # (B, n_action_steps, action_dim_for(rotation_rep))


def _to_chw_float(color: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> (3, h, w) float32 in [0, 1], resized to IMAGE_SIZE.

    `color` arrives already decoded (server-side, see XPolicyLab/AGENTS.md) and
    already RGB -- no BGR conversion belongs here.
    """
    import cv2

    resized = cv2.resize(color, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_AREA)
    return np.transpose(resized, (2, 0, 1)).astype(np.float32) / 255.0


def _robot_proprio_7d3(observation: dict, robot: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This robot's own (pelvis, left eef, right eef) pose, each [x,y,z,qw,qx,qy,qz].

    Prefers the `mhbench_state` side-channel (`scripts/mhbench_xpolicylab_env.py`'s
    `MHBenchTaskEnv.get_obs()`) -- `{"robot_a": {...}, "robot_b": {...}}`, each with
    its own `pelvis_pose`/`left_eef_pose`/`right_eef_pose` -- since the standard
    XPolicyLab obs dict only has one `state.left_ee_pose`/`right_ee_pose` pair
    (it assumes one bimanual robot), which used to be fed to *both* arm1 and
    arm2 regardless of which robot the "arm" actually is -- a real bug: arm2
    (robot_b) got robot_a's wrist data as its own proprio, since MHBench is two
    robots, not one bimanual one. Falls back to the standard slots (both
    robots reading the same one) when `mhbench_state` is absent -- e.g.
    `EVAL_ENV_TYPE=debug`'s `TestEnv`, which never sets this key -- so that
    debug-mode plumbing checks keep working unchanged.
    """
    mhbench_state = observation.get("mhbench_state")
    if mhbench_state is not None and robot in mhbench_state:
        own = mhbench_state[robot]
        pelvis = np.asarray(own["pelvis_pose"], dtype=np.float32)[:7]
        left = np.asarray(own["left_eef_pose"], dtype=np.float32)[:7]
        right = np.asarray(own["right_eef_pose"], dtype=np.float32)[:7]
        return pelvis, left, right

    state = observation["state"]
    pelvis = _pelvis_pose_7d(observation)
    left = np.asarray(state["left_ee_pose"], dtype=np.float32)[:7]  # [x,y,z,qw,qx,qy,qz]
    right = np.asarray(state["right_ee_pose"], dtype=np.float32)[:7]
    return pelvis, left, right


def _pelvis_pose_7d(observation: dict) -> np.ndarray:
    """Best-effort pelvis pose [x,y,z,qw,qx,qy,qz] from XPolicyLab's generic
    obs contract. Only mobile-base robots carry `state.mobile.base_pose`
    (debug_env_client.py's demo obs), and even then it's XPolicyLab's own
    xyzw-position + xyzw-quat convention for a wheeled/tracked base, not
    something verified against a real MHBench-in-XPolicyLab obs yet -- zero
    when absent rather than guessing. Used only as `_robot_proprio_7d3`'s
    fallback when `mhbench_state` is absent (that side-channel carries each
    robot's own real pelvis pose instead).
    """
    mobile = observation.get("state", {}).get("mobile")
    if mobile is not None and "base_pose" in mobile:
        return np.asarray(mobile["base_pose"], dtype=np.float32)[:7]
    return np.zeros(7, dtype=np.float32)


def _encode_arm_obs(observation: dict, arm: int) -> dict:
    """XPolicyLab's standard obs dict -> one arm's LatentToM obs_dict.

    Camera mapping (see the module docstring): `cam_left_wrist` -> `camera_1`
    (arm1's private view), `cam_head` -> `camera_3` (shared, both arms),
    `cam_right_wrist` -> `camera_4` (arm2's private view). This mirrors
    XPolicyLab's generic bimanual-single-robot camera names, standing in for
    MHBench's two-humanoid ego views (robot_a's ego -> camera_1, a shared
    third-person view -> camera_3, robot_b's ego -> camera_4) until the real
    port assigns MHBench-specific camera keys.

    Proprio (`arm{N}_proprio`, 21D) = pelvis(7) + left eef pose(7) + right eef
    pose(7), matching convert_to_replay_buffer.py's PROPRIO_FIELDS -- both
    wrists ride along regardless of which camera is this arm's private one,
    since removing SheafObsEncoder's old "exactly 2 low-dim keys" limit means
    there's no need to pick a single representative wrist anymore. Each arm's
    proprio is *its own* robot's data (`_robot_proprio_7d3`), not always
    robot_a's.
    """
    vision = observation["vision"]
    robot = "robot_a" if arm == 1 else "robot_b"
    pelvis, left_pose, right_pose = _robot_proprio_7d3(observation, robot)
    proprio = np.concatenate([pelvis, left_pose, right_pose]).astype(np.float32)

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


def _unpack_robot_action(action: np.ndarray, rotation_rep: str) -> dict:
    """One robot's raw model output -> {left_pose, right_pose (pos3+quat_xyzw4
    each), hands(4), base_vel(3), height(1)}.

    Handles both training representations so eval matches whichever one the
    loaded checkpoint was actually trained with (`rotation_rep` in
    deploy.yml/model_cfg) -- this is the eval-side half of the quat<->rot6d
    toggle; rotation_conversion.repack_action() (applied once, offline, by
    convert_to_replay_buffer.py) is the training-side half.
    """
    if rotation_rep == "quat":
        left = np.concatenate([action[_QUAT_LEFT_POS], action[_QUAT_LEFT_QUAT]])
        right = np.concatenate([action[_QUAT_RIGHT_POS], action[_QUAT_RIGHT_QUAT]])
        rest = action[_QUAT_REST]
    else:
        left = np.concatenate([action[_R6_LEFT_POS], rot6d_to_quat_xyzw(action[_R6_LEFT_ROT])])
        right = np.concatenate([action[_R6_RIGHT_POS], rot6d_to_quat_xyzw(action[_R6_RIGHT_ROT])])
        rest = action[_R6_REST]
    return {
        "left_pose": left,  # pos3 + quat_xyzw4
        "right_pose": right,
        "hands": rest[:4],
        "base_vel": rest[4:7],
        "height": rest[7:8],
    }


def _xyzw_to_wxyz_pose(pos_quat_xyzw: np.ndarray) -> np.ndarray:
    pos, quat_xyzw = pos_quat_xyzw[:3], pos_quat_xyzw[3:7]
    return np.concatenate([pos, quat_xyzw[[3, 0, 1, 2]]]).astype(np.float32)


def _pack_dual_arm_action(arm1_action: np.ndarray, arm2_action: np.ndarray, rotation_rep: str, ee_dim: list) -> dict:
    """Repack one timestep's arm1(=robot_a)/arm2(=robot_b) model output.

    XPolicyLab's dual-arm `ee` contract (`left_ee_pose`/`right_ee_pose` as
    [x,y,z,qw,qx,qy,qz], `left_ee_joint_state`/`right_ee_joint_state`) assumes
    ONE robot with two arms -- it has no slot for MHBench's actual shape (TWO
    robots, each with two wrists, plus base velocity and height, neither of
    which LatentToM's original fixed-base xArm convention has at all). Rather
    than guess a mapping that silently drops three of the four wrists, this
    fills the standard keys from robot_a's own two wrists only (documented
    stand-in, matching "arm1=robot_a" elsewhere in this adapter) and puts
    everything -- both robots' wrists, hands, base_vel, height -- under
    `mhbench_raw_action` so nothing is lost. A real per-robot action key
    scheme is pending the XPolicyLab-format port defining one (see
    README.md's "What's still open").
    """
    robot_a = _unpack_robot_action(arm1_action, rotation_rep)
    robot_b = _unpack_robot_action(arm2_action, rotation_rep)

    result = {
        "left_ee_pose": _xyzw_to_wxyz_pose(robot_a["left_pose"]),
        "right_ee_pose": _xyzw_to_wxyz_pose(robot_a["right_pose"]),
        "left_ee_joint_state": np.full(ee_dim[0], robot_a["hands"][:2].mean(), dtype=np.float32),
        "right_ee_joint_state": np.full(ee_dim[1], robot_a["hands"][2:].mean(), dtype=np.float32),
        "mhbench_raw_action": {"robot_a": robot_a, "robot_b": robot_b},
    }
    return result


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]
        if self.action_type != "ee":
            raise NotImplementedError(
                "LatentToM only produces end-effector pose actions -- "
                f"action_type={self.action_type!r} isn't supported yet."
            )
        self.rotation_rep = model_cfg.get("rotation_rep", "quat")
        if self.rotation_rep not in ("quat", "rot6d"):
            raise ValueError(f"rotation_rep must be 'quat' or 'rot6d', got {self.rotation_rep!r}")

        self.action_dim = get_action_dim(self.env_cfg_type)
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.batch_size = get_batch_size(self.env_cfg_type)
        num_arms = len(self.robot_action_dim_info["arm_dim"])
        if num_arms != 2:
            raise NotImplementedError(
                f"LatentToM is a two-arm sheaf-split policy (arm1/arm2); env_cfg_type="
                f"{self.env_cfg_type!r} registers {num_arms} arm(s)."
            )
        self.ee_dim = self.robot_action_dim_info["ee_dim"]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        policy_dir = Path(__file__).resolve().parent
        ckpt_root = resolve_checkpoint_root(
            model_cfg, policy_dir / "checkpoints", policy_dir=policy_dir, must_exist=True
        )
        use_ema = bool(model_cfg.get("use_ema", True))
        self.arm1_policy = _load_arm_policy(ckpt_root / "checkpoints" / "arm1_latest.ckpt", 1, self.device, use_ema)
        self.arm2_policy = _load_arm_policy(ckpt_root / "checkpoints" / "arm2_latest.ckpt", 2, self.device, use_ema)

        expected_dim = action_dim_for(self.rotation_rep)
        for arm_id, policy in ((1, self.arm1_policy), (2, self.arm2_policy)):
            if policy.action_dim != expected_dim:
                raise ValueError(
                    f"arm{arm_id} checkpoint has action_dim={policy.action_dim}, but "
                    f"rotation_rep={self.rotation_rep!r} expects {expected_dim} -- deploy.yml's "
                    "rotation_rep must match what this checkpoint was trained with."
                )

        n_obs_steps = self.arm1_policy.n_obs_steps
        self.n_action_steps = self.arm1_policy.n_action_steps
        self.runner = SheafRunner(n_obs_steps=n_obs_steps)
        self._env_idx_list = None

        print(
            f"[LatentToM] loaded arm1/arm2 from {ckpt_root} "
            f"(device={self.device}, use_ema={use_ema}, rotation_rep={self.rotation_rep})"
        )

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
            chunk = [
                _pack_dual_arm_action(arm1_action[b, t], arm2_action[b, t], self.rotation_rep, self.ee_dim)
                for t in range(chunk_len)
            ]
            action_batch.append(chunk)
        return action_batch

    def reset(self):
        self.runner.reset()
        self._env_idx_list = None
