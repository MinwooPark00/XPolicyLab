"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import json
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.mhbench_policy as mhbench_policy
import openpi.policies.wuji_policy as wuji_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

# RoboDojo normalization assets bundled with this adapter (openpi/assets/RoboDojo_assets).
_ROBODOJO_ASSETS_DIR = pathlib.Path(__file__).resolve().parents[3] / "assets" / "RoboDojo_assets"
# Norm stats for Tianji Marvin + Wuji Hand (written by compute_norm_stats).
_WUJI_ASSETS_DIR = pathlib.Path(__file__).resolve().parents[3] / "assets" / "Wuji_assets"

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Video decoder backend for LeRobot datasets. Forced to pyav by default because
    # torchcodec is present in some environments but not fully functional at runtime.
    video_backend: Literal["pyav", "torchcodec", "video_reader"] = "pyav"
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False
    # Episodes to train on; None means the whole dataset. LeRobot declares its
    # own splits in `meta/info.json`, and the loader does not read them: without
    # this a dataset with a held-out validation split would be trained on in
    # full.
    episodes: Sequence[int] | None = None
    # Episodes to score but never train on. Set alongside `episodes` from the
    # same `meta/info.json`; `None` means the trainer has nothing to validate
    # against and skips the pass.
    val_episodes: Sequence[int] | None = None

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotWujiDataConfig(DataConfigFactory):
    """
    Config for Tianji Marvin + dual Wuji Hand Gen1 (adapted from wuji-openpi).

    Dataset features:
    - observation.state: 54 dims (7 left arm + 20 left hand + 7 right arm + 20 right hand)
    - action: 54 dims
    - observation.images.cam_left_wrist / cam_right_wrist
    - base camera: observation.images.stereo_right (default) or cam_high via base_image_key

    Requires model.action_dim=54. Load pi05_base with PartialCheckpointWeightLoader.
    """

    extra_delta_transform: bool = False
    # LeRobot key for the head / stereo base camera.
    base_image_key: str = "observation.images.stereo_right"
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": self.base_image_key,
                        "observation/left_wrist_image": "observation.images.cam_left_wrist",
                        "observation/right_wrist_image": "observation.images.cam_right_wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[wuji_policy.WujiInputs(model_type=model_config.model_type)],
            outputs=[wuji_policy.WujiOutputs(action_dim=model_config.action_dim)],
        )

        # Arms: delta; hands: absolute (same as wuji-openpi).
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(7, -20, 7, -20)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotMHBenchDataConfig(DataConfigFactory):
    """MHBench: two Unitree G1 humanoids, one LeRobot export, three targets.

    The same dataset trains a `centralized` policy (both robots: 86-dim state,
    70-dim action, both ego views) and two `decentralized` ones (one robot:
    43-dim state, 35-dim action, its own ego view). Which is which is `robot`;
    everything else about the columns is derived in
    `openpi.policies.mhbench_policy` from MHBench's own key authority.

    Two MHBench-specific points:

    * The dataset must be the **v3.0 view** written by
      `baselines/scripts/convert_mhbench_lerobot_v30.py`. MHBench exports v2.1
      for GR00T, and the pinned lerobot rejects that outright.
    * The instruction is a constant per (task, role) -- the pair is told to carry
      the basket, robot_a to side-step right, robot_b to side-step left. It is
      therefore injected as a default prompt read from `meta/tasks.parquet`
      rather than looked up per row, which also means training and serving use
      the same sentence by construction: `ModelTransformFactory` runs in both.
    """

    # None for the pair, "robot_a"/"robot_b" for a single robot.
    robot: str | None = None

    # Row of `meta/tasks.parquet` holding this target's instruction. MHBench
    # writes the pair's sentence at 0 and each robot's at 1 and 2 -- the order
    # of `mhbench_keys.LANGUAGE_KEYS`.
    prompt_task_index: int = 0

    # Splits of `meta/info.json`. MHBench holds out the last episodes; without
    # `train_split` the loader would take all of them, and `val_split` is what
    # the trainer scores so the curve is not entirely on data it is fitting.
    train_split: str = "train"
    val_split: str = "val"

    # The chunk is assembled from three columns: joint targets, the base height
    # command and the navigation command. All three need delta_timestamps.
    action_sequence_keys: Sequence[str] = (
        "action",
        "teleop.navigate_command",
        "teleop.base_height_command",
    )

    def _dataset_root(self) -> pathlib.Path | None:
        try:
            from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
        except ImportError:
            return None
        root = pathlib.Path(HF_LEROBOT_HOME) / self.repo_id
        return root if (root / "meta" / "info.json").exists() else None

    def _split_episodes(self, root: pathlib.Path | None, split: str, *, required: bool) -> Sequence[int] | None:
        # No dataset on this host means nothing is going to read it either --
        # `create_trained_policy` builds a DataConfig just to reach the
        # transforms. Only a dataset that is present but does not declare the
        # split is an error worth raising, and only for the training split: an
        # export with no validation episodes is a smaller run, not a wrong one.
        if root is None:
            return None
        splits = json.loads((root / "meta" / "info.json").read_text()).get("splits") or {}
        if split not in splits:
            if not required:
                return None
            raise ValueError(
                f"{root} declares no {split!r} split (has {sorted(splits) or 'none'}); "
                "re-export it or training will silently include the held-out episodes."
            )
        start, end = (int(bound) for bound in splits[split].split(":"))
        return tuple(range(start, end))

    def _prompt(self, root: pathlib.Path | None) -> str:
        if root is None:
            raise ValueError(
                f"MHBench dataset {self.repo_id!r} not found under HF_LEROBOT_HOME. It is needed even "
                "for serving, because the instruction the policy was trained with is read from it. "
                "Run baselines/scripts/convert_mhbench_lerobot_v30.py."
            )
        import pandas as pd

        tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
        # v3.0 indexes by the sentence and stores the index as the column.
        matches = tasks.index[tasks["task_index"] == self.prompt_task_index]
        if len(matches) != 1:
            raise ValueError(
                f"{root}/meta/tasks.parquet has {len(matches)} rows with task_index="
                f"{self.prompt_task_index}; expected exactly one."
            )
        return str(matches[0])

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        root = self._dataset_root()

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/state": "observation.state",
                        "observation/ego_a": "observation.images.ego_a",
                        "observation/ego_b": "observation.images.ego_b",
                        "action/joints": "action",
                        "action/navigate": "teleop.navigate_command",
                        "action/base_height": "teleop.base_height_command",
                    }
                )
            ]
        )

        # No DeltaActions. MHBench commands absolute joint targets, an absolute
        # height and a velocity, and `state` (43 joints, URDF order, legs
        # included) is not index-aligned with `actions` (35, GR00T group order,
        # legs excluded) at any offset -- subtracting one from the other would
        # train against a wrong target without failing.
        data_transforms = _transforms.Group(
            inputs=[mhbench_policy.MHBenchInputs(model_type=model_config.model_type, robot=self.robot)],
            outputs=[mhbench_policy.MHBenchOutputs(robot=self.robot)],
        )

        model_transforms = ModelTransformFactory(default_prompt=self._prompt(root))(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            episodes=self._split_episodes(root, self.train_split, required=True),
            val_episodes=self._split_episodes(root, self.val_split, required=False),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotMHBenchSharedDataConfig(LeRobotMHBenchDataConfig):
    """MHBench: one policy over every task and both roles.

    The dataset is the flattened all-task export
    (`scripts/build_multitask_lerobot.py`, then the v3.0 view): every row is
    one robot already, so there is no slicing to do and `robot` is meaningless
    here. What differs from the per-task configs is only where the instruction
    comes from -- **per row**, out of `meta/tasks.jsonl`, rather than one
    constant baked in as a default prompt. That is what tells the four tasks
    and the two roles apart, and it is the whole mechanism: same weights, eight
    jobs, eight sentences.
    """

    # Read per sample by PromptFromLeRobotTask, which the loader inserts when
    # `prompt_from_task` is set -- before the repack, so `prompt` is already in
    # the row by the time MHBenchSharedInputs looks for it.
    prompt_task_index: int = 0

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        root = self._dataset_root()

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/state": "observation.state",
                        "observation/ego": "observation.images.ego",
                        "action/all": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[mhbench_policy.MHBenchSharedInputs(model_type=model_config.model_type)],
            outputs=[mhbench_policy.MHBenchOutputs(robot="robot_a")],
        )
        # No default_prompt: every row carries its own.
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            # One column now -- the merged export writes the 35 numbers already
            # assembled, so there is nothing to chunk alongside it.
            action_sequence_keys=("action",),
            prompt_from_task=True,
            episodes=self._split_episodes(root, self.train_split, required=True),
            val_episodes=self._split_episodes(root, self.val_split, required=False),
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"
    # Optional exact checkpoint directory. XPolicyLab train.sh uses this to
    # keep policy checkpoints under policy/<name>/checkpoints/<6-tuple>.
    checkpoint_dir_override: str | None = None

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 8
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to score the held-out split. `None` -- the default,
    # and upstream's only behaviour -- never scores it, so a run's loss curve is
    # entirely on data it is fitting. Needs `DataConfig.val_episodes`.
    val_interval: int | None = None
    # Batches per validation pass. The loader is built unshuffled and finite, so
    # every pass sees the same samples, and the noise the flow-matching loss
    # draws is fixed too: without both, consecutive validations differ by more
    # than the model moved between them.
    val_batches: int = 16
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        if self.checkpoint_dir_override:
            return pathlib.Path(self.checkpoint_dir_override).resolve()
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    TrainConfig(
        name="pi05_base_aloha_full_sim_arx-x5_seed_0",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="RoboDojo_sim_arx-x5_v30",
            assets=AssetsConfig(
                assets_dir=str(_ROBODOJO_ASSETS_DIR),
                asset_id="arx_x5_sim",
            ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        seed=0,
        batch_size=256,
        fsdp_devices=2,
        num_train_steps=60000,
    ),
    TrainConfig(
        name="pi05_base_aloha_full_sim_arx-x5_seed_1",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="RoboDojo_sim_arx-x5_v30",
            assets=AssetsConfig(
                assets_dir=str(_ROBODOJO_ASSETS_DIR),
                asset_id="arx_x5_sim",
            ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        seed=1,
        batch_size=256,
        fsdp_devices=2,
        num_train_steps=60000,
    ),
    TrainConfig(
        name="pi05_base_aloha_full_sim_arx-x5_seed_2",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="RoboDojo_sim_arx-x5_v30",
            assets=AssetsConfig(
                assets_dir=str(_ROBODOJO_ASSETS_DIR),
                asset_id="arx_x5_sim",
            ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        seed=2,
        batch_size=256,
        fsdp_devices=2,
        num_train_steps=60000,
    ),
    # Tianji Marvin + dual Wuji Hand Gen1 (54D). Set data.repo_id to your LeRobot dataset
    # before compute_norm_stats / train. Skip RoboDojo process_data.py; use wuji mcap→LeRobot.
    TrainConfig(
        name="pi05_wuji_marvin_54d",
        model=pi0_config.Pi0Config(pi05=True, action_dim=54, action_horizon=50, max_token_len=256),
        data=LeRobotWujiDataConfig(
            # Replace with your LeRobot repo id or local dataset id under HF_LEROBOT_HOME.
            repo_id="tianji_marvin_wuji",
            assets=AssetsConfig(
                assets_dir=str(_WUJI_ASSETS_DIR),
                asset_id="tianji_marvin_wuji",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
            # If your dataset uses cam_high instead of stereo_right, set:
            # base_image_key="observation.images.cam_high",
        ),
        weight_loader=weight_loaders.PartialCheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=30_000,
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-6,
        ),
    ),
]

# -- MHBench ----------------------------------------------------------------
# Four tasks x three targets. The pair is one policy driving both robots; the
# other two are the decentralized halves, trained independently and served side
# by side. Generated rather than written out fifteen times, because they differ
# only in the task, the robot and the instruction row. The shared multitask
# config below is the shipped decentralized policy; these are the single-task
# per-robot variants and the centralized ones.
#
# LoRA, matching the GR00T_N17 baseline these are compared against -- same
# batch, same step count. `action_dim` is the real action width -- pi0.5 has no
# `state_proj`, so the state reaches the model as prompt text and its width is
# independent of this number.
MHBENCH_TASKS = ("cocarry", "handover", "framehang", "doorpassage")

# (suffix, robot, tasks.parquet row). Row order is mhbench_keys.LANGUAGE_KEYS:
# the pair's shared instruction, then robot_a's, then robot_b's.
MHBENCH_TARGETS = (
    ("centralized", None, 0),
    ("robot_a", "robot_a", 1),
    ("robot_b", "robot_b", 2),
)

# Measured, not guessed: pi0.5 spells the state out as digits in the prompt, and
# PaligemmaTokenizer truncates past this with only a logging.warning. Worst case
# over all four tasks with every value three digits wide is 384 tokens for the
# 86-dim pair and 217 for a single robot's 43. The pi0.5 default of 200 would
# cut both. `mhbench_policy_test.py` re-measures and asserts the headroom.
MHBENCH_MAX_TOKEN_LEN = {None: 400, "robot_a": 256, "robot_b": 256}


def _mhbench_lora_model(robot: str | None) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=mhbench_policy.action_dim(robot),
        # 50 steps at MHBench's 50 Hz is 1.0 s, and pi05_base is natively a
        # 50-step model.
        action_horizon=50,
        max_token_len=MHBENCH_MAX_TOKEN_LEN[robot],
        # Explicit, though it is also the pi05 default: with pi05=True there is
        # no state_proj at all (see pi0.Pi0.__init__), so turning this off would
        # drop the state from the model entirely while training happily.
        discrete_state_input=True,
    )


_CONFIGS.extend(
    TrainConfig(
        name=f"pi05_mhbench_{task}_{suffix}",
        project_name="MHBench-Pi05",
        model=_mhbench_lora_model(robot),
        data=LeRobotMHBenchDataConfig(
            repo_id=f"mhbench-{task}",
            robot=robot,
            prompt_task_index=prompt_index,
        ),
        # pi05_base is a 32-dim-action model; the action projections cannot be
        # reused at 35 or 70 and are left randomly initialised. Everything else
        # -- SigLIP, both Gemmas, the pi05 time MLPs -- loads.
        weight_loader=weight_loaders.PartialCheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=_mhbench_lora_model(robot).get_freeze_filter(),
        # LoRA: an EMA copy would shadow the frozen base weights too, for nothing.
        ema_decay=None,
        # The benchmark's shared budget, matched to the GR00T_N17 baseline
        # these are compared against (baselines/scripts/train/GR00T_N17.sh:
        # GLOBAL_BATCH_SIZE=32, MAX_STEPS=40000, SAVE_STEPS=2000), so the two
        # differ in method and not in compute. 40k since 2026-08-29; the
        # checkpoints trained before that stopped at 20k.
        batch_size=32,
        num_train_steps=40_000,
        # The schedule has to span the run. openpi's default decay_steps is
        # 30 000, which matches its default num_train_steps -- raising only the
        # step count to 40 000 (2026-08-29) left the cosine bottoming out at 30k
        # and the last quarter of training crawling at the 2.5e-6 floor. optax's
        # decay_steps is the *total* length including warmup, so 40 000 is the
        # run. peak_lr and decay_lr are openpi's own, untouched.
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=40_000),
        save_interval=2000,
        # 16 CPUs per GPU on the partitions these run on (DefCpuPerGPU=16);
        # video decode is the loader's cost and 8 leaves half of them idle.
        num_workers=12,
        # GR00T_N17 scores its held-out split every 1000 steps over at most 512
        # samples (train_groot_*.sbatch: EVAL_STEPS, EVAL_MAX_SAMPLES); 16
        # batches of 32 is the same 512.
        val_interval=1000,
        val_batches=16,
        # Keep the latest checkpoint (max_to_keep=1 in checkpoints.py) plus
        # every 10 000th step, which is exactly the set the benchmark evaluates:
        # 10k/20k/30k/40k, four points per run. keep_period=None kept only the
        # latest, so a sweep launched after training finished would find one
        # checkpoint and three deleted directories -- the run having thrown away
        # three quarters of its own results. pi0.5 params are ~12 GB each, so
        # this is ~48 GB per run rather than the ~12 GB of keeping only the
        # last; the openpi default (5000) would be twice that again, for points
        # nothing measures.
        keep_period=10_000,
    )
    for task in MHBENCH_TASKS
    for suffix, robot, prompt_index in MHBENCH_TARGETS
)

# The shared decentralized policy: one set of weights over all four tasks and
# both roles, the same budget as one of the per-task runs above so a comparison
# is about the method rather than the compute. `robot_a`'s model shape -- 43-dim
# state, 35-dim action, one camera -- because every row of the flattened dataset
# is one robot.
_CONFIGS.append(
    TrainConfig(
        name="pi05_mhbench_multitask_decentralized",
        project_name="MHBench-Pi05",
        model=_mhbench_lora_model("robot_a"),
        data=LeRobotMHBenchSharedDataConfig(repo_id="mhbench-multitask"),
        weight_loader=weight_loaders.PartialCheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=_mhbench_lora_model("robot_a").get_freeze_filter(),
        ema_decay=None,
        batch_size=32,
        num_train_steps=40_000,
        # The schedule has to span the run. openpi's default decay_steps is
        # 30 000, which matches its default num_train_steps -- raising only the
        # step count to 40 000 (2026-08-29) left the cosine bottoming out at 30k
        # and the last quarter of training crawling at the 2.5e-6 floor. optax's
        # decay_steps is the *total* length including warmup, so 40 000 is the
        # run. peak_lr and decay_lr are openpi's own, untouched.
        lr_schedule=_optimizer.CosineDecaySchedule(decay_steps=40_000),
        save_interval=2000,
        num_workers=12,
        val_interval=1000,
        val_batches=16,
        keep_period=10_000,   # the benchmark's four evaluation points, as above
    )
)

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
