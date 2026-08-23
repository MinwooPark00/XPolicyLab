"""The MHBench column layout, checked against the dataset that defines it.

:mod:`openpi.policies.mhbench_policy` derives its slices from
``configs/gr00t/mhbench_keys.py`` so that importing it needs no dataset. The
dataset states the same layout independently, in ``meta/modality.json``, which
``scripts/export_lerobot.py`` writes. These tests assert the two agree -- so a
layout change that reaches only one of them fails here rather than training a
model against shifted columns.

Run from the Pi_05 openpi checkout::

    uv run pytest src/openpi/policies/mhbench_policy_test.py -v

``MHBENCH_LEROBOT_ROOT`` points at one exported task; the tests skip if it is
absent rather than failing, so the suite still runs on a machine without the
data.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import mhbench_policy

_DEFAULT_ROOT = Path(__file__).resolve().parents[8] / "datasets" / "cocarry" / "lerobot"


def _lerobot_root() -> Path:
    root = Path(os.environ.get("MHBENCH_LEROBOT_ROOT", _DEFAULT_ROOT))
    if not (root / "meta" / "modality.json").exists():
        pytest.skip(f"no MHBench LeRobot export at {root}; set MHBENCH_LEROBOT_ROOT")
    return root


@pytest.fixture(scope="module")
def modality() -> dict:
    return json.loads((_lerobot_root() / "meta" / "modality.json").read_text())


@pytest.fixture(scope="module")
def keys():
    return mhbench_policy._mhbench_keys()


def _spans_from_modality(modality: dict, section: str, names: list[str], default_column: str):
    """``meta/modality.json``'s own answer: (column, start, end) per key."""
    out = []
    for name in names:
        entry = modality[section][name]
        out.append((entry.get("original_key", default_column), entry["start"], entry["end"]))
    return out


# -- state ------------------------------------------------------------------


def test_state_spans_match_modality_json(modality, keys):
    """Every (robot, joint group) block sits where the dataset says it does."""
    derived = mhbench_policy._joint_spans()
    for robot in keys.ROBOTS:
        for group in keys.JOINT_GROUPS:
            entry = modality["state"][f"{robot}_{group}"]
            assert derived[(robot, group)] == (entry["start"], entry["end"]), (
                f"{robot}_{group}: derived {derived[(robot, group)]} vs modality.json "
                f"{(entry['start'], entry['end'])}"
            )


@pytest.mark.parametrize(("robot", "width"), [(None, 86), ("robot_a", 43), ("robot_b", 43)])
def test_state_width(robot, width):
    assert mhbench_policy.state_dim(robot) == width
    assert sum(s.width for s in mhbench_policy.state_slices(robot)) == width


def test_state_slices_are_contiguous_per_robot(keys):
    """A robot's joints are one contiguous block -- the seven slices only spell
    out the groups so a reordering shows up as a changed slice."""
    for index, robot in enumerate(keys.ROBOTS):
        slices = mhbench_policy.state_slices(robot)
        assert slices[0].start == index * keys.JOINTS_PER_ROBOT
        assert slices[-1].end == (index + 1) * keys.JOINTS_PER_ROBOT
        for previous, current in zip(slices, slices[1:], strict=False):
            assert previous.end == current.start


# -- action -----------------------------------------------------------------

_COLUMN_FOR = {
    mhbench_policy.JOINT_COLUMN: "action",
    mhbench_policy.BASE_HEIGHT_COLUMN: "teleop.base_height_command",
    mhbench_policy.NAVIGATE_COLUMN: "teleop.navigate_command",
}


@pytest.mark.parametrize("robot", [None, "robot_a", "robot_b"])
def test_action_slices_match_modality_json(modality, keys, robot):
    """The 35/70 numbers, in ``mhbench_keys.ACTION_KEYS`` order, come from the
    columns and offsets ``meta/modality.json`` names."""
    expected = _spans_from_modality(modality, "action", keys.action_keys(robot), "action")
    actual = [
        (_COLUMN_FOR[s.column], s.start, s.end) for s in mhbench_policy.action_slices(robot)
    ]
    assert actual == expected


@pytest.mark.parametrize(("robot", "width"), [(None, 70), ("robot_a", 35), ("robot_b", 35)])
def test_action_width(robot, width):
    assert mhbench_policy.action_dim(robot) == width
    assert sum(s.width for s in mhbench_policy.action_slices(robot)) == width


def test_action_layout_matches_the_environments_unpacking(keys):
    """``MHBenchTaskEnv.take_action`` reads a robot's 35 back as
    ``joint_targets = flat[0:31]``, ``height = flat[31:32]``,
    ``base_vel = flat[32:35]`` -- the same split ACT and DP pack. Assert our
    ordering puts those three where it looks for them."""
    slices = mhbench_policy.action_slices("robot_a")
    joint = [s for s in slices if s.column == mhbench_policy.JOINT_COLUMN]
    assert sum(s.width for s in joint) == 31
    assert slices[len(joint)].column == mhbench_policy.BASE_HEIGHT_COLUMN
    assert slices[len(joint)].width == 1
    assert slices[len(joint) + 1].column == mhbench_policy.NAVIGATE_COLUMN
    assert slices[len(joint) + 1].width == 3
    assert [s.column for s in slices[: len(joint)]] == [mhbench_policy.JOINT_COLUMN] * len(joint)


# -- the transform itself ---------------------------------------------------


def _fake_row(horizon: int = 4) -> dict:
    rng = np.random.default_rng(0)
    return {
        "observation/state": rng.normal(size=86).astype(np.float32),
        "observation/ego_a": rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8),
        "observation/ego_b": rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8),
        "action/joints": rng.normal(size=(horizon, 86)).astype(np.float32),
        "action/navigate": rng.normal(size=(horizon, 6)).astype(np.float32),
        "action/base_height": rng.normal(size=(horizon, 2)).astype(np.float32),
        "prompt": "carry it together",
    }


@pytest.mark.parametrize(
    ("robot", "state_width", "action_width", "live_cameras"),
    [(None, 86, 70, 2), ("robot_a", 43, 35, 1), ("robot_b", 43, 35, 1)],
)
def test_inputs_shapes_and_masks(robot, state_width, action_width, live_cameras):
    out = mhbench_policy.MHBenchInputs(model_type=_model.ModelType.PI05, robot=robot)(_fake_row())

    assert out["state"].shape == (state_width,)
    assert out["actions"].shape == (4, action_width)
    # Only the slots this target carries -- see MHBenchInputs on why dropping
    # beats masking. base_0_rgb is always first, so a one-camera target uses the
    # slot pi0.5 does not augment as a wrist view.
    assert list(out["image"]) == list(mhbench_policy.PI0_IMAGE_SLOTS[:live_cameras])
    assert all(bool(m) for m in out["image_mask"].values())
    for image in out["image"].values():
        assert image.shape == (240, 320, 3) and image.dtype == np.uint8


def test_decentralized_reads_its_own_ego_view():
    row = _fake_row()
    for robot in ("robot_a", "robot_b"):
        out = mhbench_policy.MHBenchInputs(model_type=_model.ModelType.PI05, robot=robot)(row)
        expected = row[f"observation/{mhbench_policy.EGO_VIEW[robot]}"]
        np.testing.assert_array_equal(out["image"]["base_0_rgb"], expected)


def test_centralized_is_the_two_halves_concatenated():
    """One policy's 70 is robot_a's 35 followed by robot_b's 35 -- so a
    decentralized pair and the centralized policy command the same numbers."""
    row = _fake_row()
    make = lambda robot: mhbench_policy.MHBenchInputs(  # noqa: E731
        model_type=_model.ModelType.PI05, robot=robot
    )(row)
    duo, a, b = make(None), make("robot_a"), make("robot_b")
    np.testing.assert_array_equal(duo["actions"], np.concatenate([a["actions"], b["actions"]], -1))
    np.testing.assert_array_equal(duo["state"], np.concatenate([a["state"], b["state"]], -1))


def test_inference_row_needs_no_action_columns():
    row = {k: v for k, v in _fake_row().items() if not k.startswith("action/")}
    out = mhbench_policy.MHBenchInputs(model_type=_model.ModelType.PI05, robot=None)(row)
    assert "actions" not in out
    assert out["state"].shape == (86,)


@pytest.mark.parametrize(("robot", "padded", "real"), [(None, 86, 70), ("robot_a", 43, 35)])
def test_outputs_trim_the_padding(robot, padded, real):
    actions = np.arange(4 * padded, dtype=np.float32).reshape(4, padded)
    out = mhbench_policy.MHBenchOutputs(robot=robot)({"actions": actions})
    assert out["actions"].shape == (4, real)
    np.testing.assert_array_equal(out["actions"], actions[:, :real])


def test_chw_float_images_are_accepted():
    """The env hands HWC uint8; LeRobot hands CHW float in [0, 1]. Both land as
    the same uint8 HWC picture."""
    hwc = np.random.default_rng(0).integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    chw_float = np.transpose(hwc, (2, 0, 1)).astype(np.float32) / 255.0
    np.testing.assert_allclose(mhbench_policy._parse_image(chw_float), hwc, atol=1)
    np.testing.assert_array_equal(mhbench_policy._parse_image(hwc), hwc)


def test_dropping_unused_image_slots_is_free():
    """The whole justification for emitting fewer than three images.

    `MHBenchInputs` hands pi0.5 only the cameras a target carries, rather than
    padding to three with `image_mask=False`. That is a pure saving -- a masked
    slot still costs a SigLIP forward and 256 prefix tokens -- but only if the
    two are arithmetically identical. Assert it end to end on a dummy-sized
    model rather than by reading `make_attn_mask`.
    """
    import jax
    import jax.numpy as jnp

    from openpi.models.pi0_config import Pi0Config

    config = Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=35,
        action_horizon=10,
        max_token_len=64,
    )
    model = config.create(jax.random.key(0))

    rng = np.random.default_rng(0)
    image = jnp.asarray(rng.uniform(-1, 1, (1, 224, 224, 3)), dtype=jnp.float32)
    shared = {
        "state": jnp.asarray(rng.normal(size=(1, 43)), dtype=jnp.float32),
        "tokenized_prompt": jnp.asarray(rng.integers(0, 1000, (1, 64)), dtype=jnp.int32),
        "tokenized_prompt_mask": jnp.ones((1, 64), dtype=bool),
    }
    actions = jnp.asarray(rng.normal(size=(1, 10, 35)), dtype=jnp.float32)

    one = _model.Observation.from_dict(
        {**shared, "image": {"base_0_rgb": image}, "image_mask": {"base_0_rgb": jnp.array([True])}}
    )
    three = _model.Observation.from_dict(
        {
            **shared,
            "image": {
                "base_0_rgb": image,
                "left_wrist_0_rgb": jnp.zeros_like(image),
                "right_wrist_0_rgb": jnp.zeros_like(image),
            },
            "image_mask": {
                "base_0_rgb": jnp.array([True]),
                "left_wrist_0_rgb": jnp.array([False]),
                "right_wrist_0_rgb": jnp.array([False]),
            },
        }
    )

    key = jax.random.key(42)
    np.testing.assert_array_equal(
        np.asarray(model.compute_loss(key, one, actions, train=False)),
        np.asarray(model.compute_loss(key, three, actions, train=False)),
    )
