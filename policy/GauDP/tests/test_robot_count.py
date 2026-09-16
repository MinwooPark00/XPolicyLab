"""Two- and three-robot GauDP side by side.

The two-robot assertions are the backward-compat evidence for the round already
trained on 86D / 70D; the three-robot ones are the 129D / 105D path.
"""

import ast
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from torch import nn

from XPolicyLab.policy.GauDP.gaudp.dataset import IMAGE_SIZE, GauDPSequenceDataset, _LazyH5Dataset
from XPolicyLab.policy.GauDP.gaudp.gaussian import load_gaussian_checkpoint
from XPolicyLab.policy.GauDP.gaudp.policy import (
    GauDPPolicy,
    action_group_indices,
    policy_checkpoint_payload,
)
from XPolicyLab.policy.GauDP.gaudp.schema import (
    ACTION_SCHEMA,
    CAMERA_SLOT,
    STATE_SCHEMA,
    action_dim,
    flat_action_from_xpolicy,
    pack_xpolicy_action,
    proprio_dim,
    robot_count_from_action_dim,
    robot_count_from_state_dim,
    robot_names,
    robots_in_observation,
)
from XPolicyLab.policy.GauDP.model import _check_checkpoint_contract, encode_observation
from XPolicyLab.policy.GauDP.process_data import (
    _direct_modality,
    _robot_count_from_info,
    resolve_cameras,
)

REPO = Path(__file__).resolve().parents[5]
EGO3 = ["ego_a", "ego_b", "ego_c"]

# policy.config() of a real cohub2 checkpoint. train_policy.py compares `config`
# as a whole dict on resume, so one more key here refuses every run in flight.
CONFIG_KEYS = {
    "num_views", "horizon", "n_obs_steps", "n_action_steps", "num_train_timesteps",
    "num_inference_steps", "obs_feature_dim", "down_dims", "crop_shape", "image_norm",
    "group_norm_divisor",
}


class TinyGaussian(nn.Module):
    def __init__(self):
        super().__init__()
        self.unused = nn.Parameter(torch.ones(()))

    def forward(self, context, global_step=0, return_features=False):
        image = context["image"]
        features = torch.cat((image, image.repeat(1, 1, 3, 1, 1), image[:, :, :1]), dim=2)[:, :, :13]
        return (None, features) if return_features else None


class TinyObservation(nn.Module):
    feature_dim = 4

    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.output_dim = state_dim + 12  # room for 3 channels x up to 4 views

    def forward(self, images, state):
        pooled = images.mean(dim=(-1, -2)).flatten(1)
        padding = torch.zeros((images.shape[0], 12 - pooled.shape[1]), device=images.device)
        return torch.cat((pooled, padding, state), dim=-1)


def _policy(robot_count=2, views=2):
    return GauDPPolicy(
        num_views=views,
        robot_count=robot_count,
        num_inference_steps=1,
        down_dims=(32, 64, 128),
        gaussian_encoder=TinyGaussian(),
        observation_encoder=TinyObservation(proprio_dim(robot_count)),
    )


# --- schema -----------------------------------------------------------------

def test_robot_count_is_read_off_the_width():
    assert (robot_count_from_state_dim(86), robot_count_from_state_dim(129)) == (2, 3)
    assert (robot_count_from_action_dim(70), robot_count_from_action_dim(105)) == (2, 3)
    for width in (0, 43, 172):
        with pytest.raises(ValueError, match="70D"):
            robot_count_from_state_dim(width)
    with pytest.raises(ValueError):
        robot_names(4)


def test_a_105d_action_packs_into_three_robots_and_back():
    flat = np.arange(105, dtype=np.float32)
    raw = pack_xpolicy_action(flat)["mhbench_raw_action"]
    assert list(raw) == ["robot_a", "robot_b", "robot_c"]
    np.testing.assert_array_equal(raw["robot_c"]["joint_targets"], flat[70:101])
    np.testing.assert_array_equal(raw["robot_c"]["height"], flat[101:102])
    np.testing.assert_array_equal(raw["robot_c"]["base_vel"], flat[102:105])
    np.testing.assert_array_equal(flat_action_from_xpolicy({"mhbench_raw_action": raw}), flat)
    with pytest.raises(ValueError, match="70D for 2 robots"):
        pack_xpolicy_action(flat, robot_names(2))


def test_camera_slots_are_the_envs():
    source = (REPO / "scripts" / "mhbench_xpolicylab_env.py").read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_VISION_SLOT" for target in node.targets
        ):
            assert ast.literal_eval(node.value) == CAMERA_SLOT
            return
    pytest.fail("scripts/mhbench_xpolicylab_env.py no longer defines _VISION_SLOT")


# --- policy -----------------------------------------------------------------

def test_two_robot_metric_groups_are_the_historical_lists():
    def both(local):
        return [index + offset for offset in (0, 35) for index in local]

    assert action_group_indices(2) == {
        "robot_a": list(range(0, 35)),
        "robot_b": list(range(35, 70)),
        "arm": both(range(0, 14)),
        "hand": both(range(14, 28)),
        "waist": both(range(28, 31)),
        "height": both([31]),
        "navigation": both(range(32, 35)),
    }
    assert list(action_group_indices(2)) == [
        "robot_a", "robot_b", "arm", "hand", "waist", "height", "navigation"
    ]
    three = action_group_indices(3)
    assert three["robot_c"] == list(range(70, 105))
    assert three["height"] == [31, 66, 101]


def test_the_two_robot_checkpoint_contract_is_unchanged():
    policy = _policy()
    payload = policy_checkpoint_payload(policy)
    assert set(payload["config"]) == CONFIG_KEYS
    assert (payload["state_dim"], payload["action_dim"], payload["robot_count"]) == (86, 70, 2)
    assert tuple(policy.normalizer.state_min.shape) == (86,)
    assert tuple(policy.normalizer.action_min.shape) == (70,)


def _batch(robot_count, views, horizon=8):
    return {
        "images": torch.rand(1, 1, views, 3, 32, 32),
        "state": torch.randn(1, 1, proprio_dim(robot_count)),
        "action": torch.randn(1, horizon, action_dim(robot_count)),
    }


@pytest.mark.parametrize("robot_count", [2, 3])
def test_a_policy_is_as_wide_as_its_robots(robot_count):
    policy = _policy(robot_count, views=robot_count)
    policy.normalizer.fit(
        torch.randn(5, proprio_dim(robot_count)), torch.randn(5, action_dim(robot_count))
    )
    batch = _batch(robot_count, robot_count)
    loss, metrics = policy.compute_loss(batch, return_metrics=True)
    loss.backward()
    robot_keys = {key for key in metrics if key.startswith("action/robot_")}
    assert robot_keys == {f"action/{robot}_mse" for robot in robot_names(robot_count)}
    policy.eval()
    with torch.no_grad():
        action = policy.predict_action(batch["images"], batch["state"])
    assert action.shape == (1, 6, action_dim(robot_count))
    assert policy_checkpoint_payload(policy)["robot_count"] == robot_count
    assert set(policy_checkpoint_payload(policy)["config"]) == CONFIG_KEYS


def test_a_normalizer_refuses_statistics_of_another_width():
    policy = _policy(3, views=3)
    with pytest.raises(ValueError, match=r"\[N,129\]"):
        policy.normalizer.fit(torch.randn(5, 86), torch.randn(5, 105))


# --- serving ----------------------------------------------------------------

def _payload(**overrides):
    payload = {
        "format": "mhbench-gaudp-policy-v2",
        "state_dim": 86,
        "action_dim": 70,
        "state_schema": STATE_SCHEMA,
        "action_schema": ACTION_SCHEMA,
    }
    payload.update(overrides)
    return payload


def test_serving_reads_the_robot_count_off_the_checkpoint():
    path = Path("policy.ckpt")
    assert _check_checkpoint_contract(_payload(), path) == 2  # written before robot_count existed
    assert _check_checkpoint_contract(_payload(state_dim=129, action_dim=105, robot_count=3), path) == 3
    with pytest.raises(ValueError, match="robot_count=3"):
        _check_checkpoint_contract(_payload(robot_count=3), path)
    with pytest.raises(ValueError, match="expected 105D"):
        _check_checkpoint_contract(_payload(state_dim=129, action_dim=70), path)
    with pytest.raises(ValueError, match="86D / 70D"):
        _check_checkpoint_contract(_payload(state_dim=42, action_dim=44), path)


def _observation(robots, slots):
    return {
        "vision": {slot: {"color": np.zeros((240, 320, 3), np.uint8)} for slot in slots},
        "mhbench_state": {
            robot: {"joint_pos": np.full(43, index, np.float32)} for index, robot in enumerate(robots)
        },
    }


def test_serving_encodes_the_checkpoints_views_in_its_order():
    robots = robot_names(3)
    observation = _observation(robots, ["cam_left_wrist", "cam_right_wrist", "cam_third_view", "cam_head"])
    assert robots_in_observation(observation) == robots
    images, state = encode_observation(observation, EGO3, robots)
    assert images.shape == (3, 3, *IMAGE_SIZE)
    assert state.shape == (129,)
    np.testing.assert_array_equal(state[86:], np.full(43, 2, np.float32))

    missing = _observation(robots, ["cam_left_wrist", "cam_right_wrist"])
    with pytest.raises(KeyError, match="ego_c"):
        encode_observation(missing, EGO3, robots)


# --- conversion -------------------------------------------------------------

def _features(views, depth=None):
    features = {f"observation.images.{view}": {} for view in views}
    for view in depth if depth is not None else [view for view in views if view != "scene"]:
        features[f"observation.depth.{view}"] = {}
    return features


def test_ego_views_follow_the_robot_count():
    three = _features(EGO3 + ["scene"])
    assert resolve_cameras(three, 3, use_scene=False) == EGO3
    assert resolve_cameras(three, 3, use_scene=True) == EGO3 + ["scene"]
    assert resolve_cameras(_features(["ego_a", "ego_b", "scene"]), 2, use_scene=False) == ["ego_a", "ego_b"]
    assert resolve_cameras(three, 3, False, "ego_a,ego_b") == ["ego_a", "ego_b"]
    with pytest.raises(ValueError, match="one ego view per robot"):
        resolve_cameras(_features(["ego_a", "ego_b"]), 3, use_scene=False)
    for bad, message in (
        ("ego_a,ego_d", "unknown views"),
        ("ego_a,ego_a", "duplicate"),
        ("ego_a", "at least two"),
        ("ego_b,ego_a", "order"),
    ):
        with pytest.raises(ValueError, match=message):
            resolve_cameras(three, 3, False, bad)
    with pytest.raises(ValueError, match="use-scene"):
        resolve_cameras(three, 3, True, "ego_a,ego_b")


def _info(state, height, navigate):
    return {"features": {
        "observation.state": {"shape": [state]},
        "teleop.base_height_command": {"shape": [height]},
        "teleop.navigate_command": {"shape": [navigate]},
    }}


def test_the_robot_count_comes_from_the_exports_widths():
    assert _robot_count_from_info(_info(86, 2, 6)) == 2
    assert _robot_count_from_info(_info(129, 3, 9)) == 3
    with pytest.raises(ValueError, match="navigate_command"):
        _robot_count_from_info(_info(129, 3, 6))
    with pytest.raises(ValueError):
        _robot_count_from_info(_info(100, 3, 9))


def test_direct_slices_match_the_real_three_robot_export():
    modality_path = REPO / "datasets" / "bigtable" / "lerobot" / "meta" / "modality.json"
    if not modality_path.is_file():
        pytest.skip(f"{modality_path} is not available")
    exported = json.loads(modality_path.read_text())
    derived = _direct_modality(robot_names(3))
    for part in ("state", "action"):
        for key, value in derived[part].items():
            assert exported[part][key] == value, (part, key)


# --- datasets ---------------------------------------------------------------

def _write(path, robot_count, cameras, *, state_dim=None, action_dim_=None, frames=4):
    state_dim = proprio_dim(robot_count) if state_dim is None else state_dim
    action_dim_ = action_dim(robot_count) if action_dim_ is None else action_dim_
    with h5py.File(path, "w") as target:
        target.create_dataset("episode_ends", data=np.asarray([2, frames], np.int64))
        target.create_dataset("episode_ids", data=np.arange(2, dtype=np.int64))
        target.create_dataset("state", data=np.zeros((frames, proprio_dim(robot_count)), np.float32))
        target.create_dataset("action", data=np.zeros((frames, action_dim(robot_count)), np.float32))
        for index in range(len(cameras)):
            target.create_dataset(f"rgb_{index}", data=np.zeros((frames, 4, 4, 3), np.uint8))
        target.attrs["camera_order"] = json.dumps(cameras)
        target.attrs["source"] = str(path.parent / "lerobot")
        target.attrs["state_dim"] = state_dim
        target.attrs["action_dim"] = action_dim_
        target.attrs["robot_count"] = robot_count
        target.attrs["action_type"] = "joint"
        target.attrs["state_schema"] = json.dumps(STATE_SCHEMA)
        target.attrs["action_schema"] = json.dumps(ACTION_SCHEMA)
    return path


def _write_cache(path, data, cameras, frames=4):
    with h5py.File(path, "w") as target:
        target.create_dataset(
            "gaussian_features", data=np.zeros((frames, len(cameras), 13, *IMAGE_SIZE), np.float16)
        )
        target.attrs["camera_order"] = json.dumps(cameras)
        target.attrs["source_data"] = str(data)
        target.attrs["gaussian_checkpoint"] = "/somewhere/gaussian/best.ckpt"
        target.attrs["complete"] = True
    return path


def test_a_three_robot_dataset_trains_a_three_robot_policy(tmp_path):
    data = _write(tmp_path / "three.hdf5", 3, EGO3)
    cache = _write_cache(tmp_path / "features.hdf5", data, EGO3)
    dataset = GauDPSequenceDataset(data, True, horizon=8, n_obs_steps=1, gaussian_features=cache)
    assert (dataset.robot_count, dataset.state_dim, dataset.action_dim) == (3, 129, 105)
    sample = dataset[0]
    assert sample["state"].shape == (1, 129)
    assert sample["action"].shape == (8, 105)
    assert sample["gaussian_features"].shape[1] == 3

    policy = _policy(dataset.robot_count, views=len(dataset.camera_order))
    state, action = dataset.normalization_arrays()
    policy.normalizer.fit(state + np.random.rand(*state.shape), action + np.random.rand(*action.shape))
    batch = {
        "images": torch.rand(1, 1, 3, 3, 32, 32),
        "state": torch.as_tensor(sample["state"])[None],
        "action": torch.as_tensor(sample["action"])[None],
    }
    loss, metrics = policy.compute_loss(batch, return_metrics=True)
    assert torch.isfinite(loss) and "action/robot_c_mse" in metrics
    payload = policy_checkpoint_payload(policy, camera_order=dataset.camera_order)
    assert _check_checkpoint_contract(payload, Path("three.ckpt")) == 3


def test_a_two_robot_dataset_still_reads_as_two(tmp_path):
    data = _write(tmp_path / "two.hdf5", 2, ["ego_a", "ego_b"])
    dataset = _LazyH5Dataset(data)
    assert (dataset.robot_count, dataset.state_dim, dataset.action_dim) == (2, 86, 70)
    assert dataset.robot_names == ("robot_a", "robot_b")


def test_datasets_whose_widths_disagree_are_refused(tmp_path):
    with pytest.raises(ValueError, match="expected 105D"):
        _LazyH5Dataset(_write(tmp_path / "a.hdf5", 3, EGO3, action_dim_=70))
    with pytest.raises(ValueError, match="robot_count=3"):
        _LazyH5Dataset(_write(tmp_path / "b.hdf5", 3, EGO3, state_dim=86, action_dim_=70))
    # Without the redundant robot_count record, the array width still catches it.
    path = _write(tmp_path / "c.hdf5", 3, EGO3, state_dim=86, action_dim_=70)
    with h5py.File(path, "r+") as target:
        del target.attrs["robot_count"]
    with pytest.raises(ValueError, match="state array is 129 wide"):
        _LazyH5Dataset(path)
    path = _write(tmp_path / "d.hdf5", 3, EGO3)
    with h5py.File(path, "r+") as target:
        target.attrs["camera_order"] = json.dumps(["ego_a", "ego_b"])
    with pytest.raises(ValueError, match="more rgb_"):
        _LazyH5Dataset(path)


# --- the encoder ------------------------------------------------------------

def test_a_two_view_encoder_is_refused_for_three_views(tmp_path):
    path = tmp_path / "gaussian.ckpt"
    torch.save(
        {"format": "mhbench-gaudp-gaussian-v1", "num_views": 2, "encoder": "noposplat",
         "encoder_state": {"weight": torch.zeros(1)}},
        path,
    )
    with pytest.raises(ValueError, match="2-view"):
        load_gaussian_checkpoint(nn.Linear(1, 1), path, expect_views=3)
