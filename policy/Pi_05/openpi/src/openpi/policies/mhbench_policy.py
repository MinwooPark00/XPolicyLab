"""Policy transforms for MHBench's two Unitree G1 humanoids.

MHBench records a *pair* of robots. One episode therefore carries two of
everything, and a policy is trained against one of two targets:

    centralized     one policy drives both robots -- 86-dim state (both robots'
                    43 joint angles), 70-dim action (both robots' 35), and both
                    ego cameras.
    decentralized   one policy per agent -- 43-dim state, 35-dim action, and
                    that robot's own ego camera. The shipped form is one
                    *shared* checkpoint over every task and both roles, told
                    them apart by the instruction each agent is given
                    (:class:`MHBenchSharedInputs`, on the flattened all-task
                    dataset); naming a robot trains the single-task per-robot
                    variant instead.

Both read the *same* LeRobot export (``datasets/<task>/lerobot``); the slicing
happens here rather than in a repacked copy on disk, which is how GR00T reads
it too. The column layout is derived from ``configs/gr00t/mhbench_keys.py`` --
the module the exporter itself is built from -- so a key renamed there breaks
the import rather than silently shifting a slice.

Two things about pi0.5 that shape this file:

*   The action a policy commands is **not** a slice of one column. It is
    thirty-five numbers assembled from three: the joint targets live in
    ``action`` (86-dim, both robots, URDF order), the base height in
    ``teleop.base_height_command`` and the navigation velocity in
    ``teleop.navigate_command``. ``LeRobotMHBenchDataConfig`` asks the loader to
    chunk all three; :class:`MHBenchInputs` assembles them in
    ``mhbench_keys.ACTION_KEYS`` order, which is the order ACT and Diffusion
    Policy were trained on and the order ``MHBenchTaskEnv.take_action``
    unpacks.

*   The actions are **absolute** -- joint targets, an absolute height, a
    velocity command -- and the state and action columns are not index-aligned
    (state is 43 joints in URDF order; the action is 35 in GR00T's listing order
    plus two teleop columns). ``DeltaActions`` subtracts ``state[..., :n]`` and
    would therefore quietly train against a wrong target. It is not used, and
    there is no flag to turn it on.
"""

import dataclasses
import functools
import os
from pathlib import Path
import sys
from typing import Any

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _mhbench_keys():
    """``configs/gr00t/mhbench_keys.py``, the shared key authority.

    Found the same way ``GR00T_N17/configs/unitree_g1x2_*_config.py`` finds it:
    ``MHBENCH_CONFIG_DIR``, defaulting to the MHBench checkout's own
    ``configs/gr00t``. It is a pure-data module -- no third-party import -- so it
    loads unchanged inside openpi's Python 3.11 venv.
    """
    # .../policy/Pi_05/openpi/src/openpi/policies/mhbench_policy.py
    #  parents: policies openpi src openpi Pi_05 policy XPolicyLab baselines <repo>
    repo_root = Path(__file__).resolve().parents[8]
    config_dir = Path(os.environ.get("MHBENCH_CONFIG_DIR", repo_root / "configs" / "gr00t"))
    if str(config_dir) not in sys.path:
        sys.path.insert(0, str(config_dir))
    import mhbench_keys  # noqa: PLC0415

    return mhbench_keys


# -- column layout ----------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Slice:
    """One block of one source column, in the order the target vector wants it."""

    column: str
    start: int
    end: int

    @property
    def width(self) -> int:
        return self.end - self.start


JOINT_COLUMN = "joints"
BASE_HEIGHT_COLUMN = "base_height"
NAVIGATE_COLUMN = "navigate"


@functools.lru_cache(maxsize=None)
def _joint_spans() -> dict[tuple[str, str], tuple[int, int]]:
    """``(robot, joint group) -> [start, end)`` in the 86-wide joint vector.

    Storage order, which is the G1 URDF's own: both robots in turn, and within a
    robot the seven groups of ``mhbench_keys.JOINT_GROUPS``. Derived rather than
    copied from ``meta/modality.json`` so this file needs no dataset to import;
    ``test_mhbench_policy.py`` asserts the two agree.
    """
    keys = _mhbench_keys()
    spans: dict[tuple[str, str], tuple[int, int]] = {}
    cursor = 0
    for robot in keys.ROBOTS:
        for group in keys.JOINT_GROUPS:
            width = keys.JOINT_GROUP_WIDTHS[group]
            spans[(robot, group)] = (cursor, cursor + width)
            cursor += width
    assert cursor == keys.JOINTS_PER_ROBOT * len(keys.ROBOTS), cursor
    return spans


def state_dim(robot: str | None) -> int:
    keys = _mhbench_keys()
    return keys.JOINTS_PER_ROBOT * (1 if robot else len(keys.ROBOTS))


def action_dim(robot: str | None) -> int:
    keys = _mhbench_keys()
    return keys.ACTION_DIMS_PER_ROBOT * (1 if robot else len(keys.ROBOTS))


def state_slices(robot: str | None) -> tuple[Slice, ...]:
    """The joint angles this target observes. Contiguous, but spelled out per
    group so a layout change shows up as a changed slice rather than a changed
    number."""
    keys = _mhbench_keys()
    spans = _joint_spans()
    robots = keys.ROBOTS if robot is None else (robot,)
    return tuple(
        Slice(JOINT_COLUMN, *spans[(r, group)]) for r in robots for group in keys.JOINT_GROUPS
    )


def action_slices(robot: str | None) -> tuple[Slice, ...]:
    """The thirty-five (or seventy) numbers a policy commands, in
    ``mhbench_keys.ACTION_KEYS`` order.

    That order is ``(left_arm, right_arm, left_hand, right_hand, waist,
    base_height_command, navigate_command)`` -- note it is *not* the storage
    order, which puts each hand beside its arm. ``MHBenchTaskEnv`` reads the
    result back as ``joint_targets = flat[0:31]``, ``height = flat[31:32]``,
    ``base_vel = flat[32:35]``.
    """
    keys = _mhbench_keys()
    spans = _joint_spans()
    robots = keys.ROBOTS if robot is None else (robot,)

    out: list[Slice] = []
    for r in robots:
        index = keys.ROBOTS.index(r)
        out.extend(Slice(JOINT_COLUMN, *spans[(r, group)]) for group in keys.ACTION_JOINT_GROUPS)
        out.append(Slice(BASE_HEIGHT_COLUMN, index, index + 1))
        out.append(Slice(NAVIGATE_COLUMN, 3 * index, 3 * index + 3))
    return tuple(out)


# -- cameras ----------------------------------------------------------------

# pi0.5 has exactly three image slots. MHBench fills as many as the target
# actually carries and masks the rest: the pair sees both ego views, one robot
# sees only its own. The room camera ("scene") is deliberately left out -- a
# deployed pair does not carry it, and DP trains on two cameras / GR00T defaults
# to MHBENCH_SCENE_CAMERA=0, so leaving it out keeps the baselines comparable.
PI0_IMAGE_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

EGO_VIEW = {"robot_a": "ego_a", "robot_b": "ego_b"}


def camera_slots(robot: str | None) -> tuple[str, ...]:
    """Which MHBench camera goes into each pi0.5 slot, most-important first.

    A slot past the end of this tuple is fed zeros with ``image_mask=False``.
    """
    if robot is None:
        return ("ego_a", "ego_b")
    return (EGO_VIEW[robot],)


def _parse_image(image) -> np.ndarray:
    """Anything the loader or the env hands us -> uint8 (H, W, C)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


# -- transforms -------------------------------------------------------------


def _gather(data: dict[str, Any], slices: tuple[Slice, ...], columns: dict[str, str]) -> np.ndarray:
    blocks = []
    for span in slices:
        key = columns[span.column]
        if key not in data:
            raise KeyError(f"MHBench transform needs {key!r} (for {span.column}); got {sorted(data)}")
        blocks.append(np.asarray(data[key])[..., span.start : span.end])
    return np.concatenate(blocks, axis=-1).astype(np.float32)


STATE_COLUMNS = {JOINT_COLUMN: "observation/state"}
ACTION_COLUMNS = {
    JOINT_COLUMN: "action/joints",
    BASE_HEIGHT_COLUMN: "action/base_height",
    NAVIGATE_COLUMN: "action/navigate",
}


@dataclasses.dataclass(frozen=True)
class MHBenchInputs(transforms.DataTransformFn):
    """MHBench rows (post-repack) -> pi0.5 model inputs.

    Expects ``observation/state`` to be the **full** 86-dim joint vector in both
    modes, so training and the eval adapter hand over the same thing and this
    class is the only place that knows where a robot's columns start.
    """

    model_type: _model.ModelType
    # None for the pair, "robot_a"/"robot_b" for one robot.
    robot: str | None = None

    def __call__(self, data: dict) -> dict:
        state = _gather(data, state_slices(self.robot), STATE_COLUMNS)

        cameras = camera_slots(self.robot)
        # Only the slots this target actually carries. A masked-off slot would
        # still cost a full SigLIP forward and 256 prefix tokens, and dropping it
        # is arithmetically identical to masking it: `positions` is a cumsum over
        # `input_mask`, `make_attn_mask` excludes masked tokens as both query and
        # key, and an image block contributes only `False` to `ar_mask`. See the
        # `image_keys` argument threaded through `pi0.Pi0.compute_loss` /
        # `sample_actions`. For the decentralized target -- one camera -- this is
        # the difference between running one vision encoder and three.
        if len(cameras) > len(PI0_IMAGE_SLOTS):
            raise ValueError(f"pi0.5 has {len(PI0_IMAGE_SLOTS)} image slots; asked for {cameras}")
        images = {
            slot: _parse_image(data[f"observation/{camera}"])
            for slot, camera in zip(PI0_IMAGE_SLOTS, cameras)  # noqa: B905 - cameras is the shorter
        }
        masks = dict.fromkeys(images, np.True_)

        inputs: dict[str, Any] = {"state": state, "image": images, "image_mask": masks}

        if "action/joints" in data:
            inputs["actions"] = _gather(data, action_slices(self.robot), ACTION_COLUMNS)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class MHBenchSharedInputs(transforms.DataTransformFn):
    """One flattened row of the all-task dataset -> pi0.5 model inputs.

    The shared decentralized policy trains on
    ``baselines/data/multitask/lerobot``, where every row is already *one*
    robot: 43-dim state, 35-dim action in ``mhbench_keys.ACTION_KEYS`` order,
    one camera. So unlike :class:`MHBenchInputs` there is nothing to slice or
    assemble -- which is the point of building that dataset rather than
    teaching a mixture loader to do it per sample.

    What tells the four tasks and the two roles apart is the prompt, read per
    row from ``meta/tasks.jsonl`` by ``PromptFromLeRobotTask`` and, at serving
    time, sent over the wire by the eval client.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs: dict[str, Any] = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            # One camera, in the first slot. The other two are dropped rather
            # than masked, for the reason MHBenchInputs spells out.
            "image": {PI0_IMAGE_SLOTS[0]: _parse_image(data["observation/ego"])},
            "image_mask": {PI0_IMAGE_SLOTS[0]: np.True_},
        }
        if "action/all" in data:
            inputs["actions"] = np.asarray(data["action/all"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class MHBenchOutputs(transforms.DataTransformFn):
    """Trim the model's padded action back to what the environment accepts.

    ``model.action_dim`` is the state width (86 / 43) rather than the action
    width (70 / 35) so that the padded state matches ``fake_obs``; the extra
    columns are zeros the model learns to predict and are dropped here.
    """

    robot: str | None = None

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[..., : action_dim(self.robot)]}
