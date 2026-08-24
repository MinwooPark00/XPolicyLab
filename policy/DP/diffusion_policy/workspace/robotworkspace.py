if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import wandb
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy

import tqdm, random
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class RobotWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    # Anything that changes a saved tensor's shape. Resuming across a change to
    # one of these fails inside load_state_dict with a shape error that names a
    # layer, not the setting that moved -- so say it here instead.
    resume_critical_keys = ("horizon", "n_obs_steps", "n_action_steps")

    def run_ckpt_dir(self) -> pathlib.Path:
        """Where this run's checkpoints live -- stable across jobs.

        `output_dir` is hydra's timestamped directory, which is new on every
        submission, so a `latest.ckpt` written under it can never be found
        again. That is why `training.resume` had nothing to resume from and a
        preempted run started over from epoch 0.
        """
        save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
        # `checkpoint.run_tag` suffixes the run directory, the same way
        # eval_policy.sbatch's CKPT_TAG suffixes what it looks for. Without it a
        # second run of the same task and seed silently overwrites the first
        # one's checkpoints, which is how a baseline gets lost to its own A/B.
        run_tag = self.cfg.checkpoint.get("run_tag", None)
        run_dir = f"{save_name}-{self.cfg.training.seed}" + (f"-{run_tag}" if run_tag else "")
        return pathlib.Path(__file__).resolve().parents[2] / "checkpoints" / run_dir

    def get_checkpoint_path(self, tag="latest"):
        return self.run_ckpt_dir() / f"{tag}.ckpt"

    def save_latest_checkpoint(self):
        """Write `latest.ckpt` so that dying mid-write cannot cost the last one.

        The copy lands on `latest.ckpt.tmp` and only a rename -- atomic within a
        directory -- publishes it, so a job killed while saving leaves the
        previous `latest.ckpt` intact and resumable. The write is awaited rather
        than threaded for the same reason: the rename must not run ahead of the
        bytes it publishes.
        """
        if self._saving_thread is not None and self._saving_thread.is_alive():
            self._saving_thread.join()
        path = self.get_checkpoint_path()
        tmp = path.parent / (path.name + ".tmp")
        self.save_checkpoint(path=tmp, use_thread=True)
        self._saving_thread.join()
        self._saving_thread = None
        os.replace(tmp, path)
        return str(path)

    def init_wandb(self, cfg):
        """wandb.init, where a bookkeeping failure must not cost the run.

        The launcher sets cfg.logging.id to the run name so an evaluation can
        resume the run that trained the policy. wandb refuses an id that was
        created and deleted before -- "was previously created and deleted; try
        a new run id" -- and that refusal killed a job that had already loaded
        its checkpoint and spent twenty minutes reading its dataset, having
        trained nothing. Take a generated id instead and keep the name: the name
        is what a reader compares runs by, and dp_eval_report resolves a run by
        name when its id is not the name.
        """
        kwargs = dict(dir=str(self.output_dir),
                      config=OmegaConf.to_container(cfg, resolve=True),
                      **cfg.logging)
        try:
            return wandb.init(**kwargs)
        except Exception as error:  # noqa: BLE001 - any init failure, not just a taken id
            if not kwargs.get("id"):
                raise
            print(f"wandb.init(id={kwargs['id']}) failed ({error}); "
                  f"retrying with a wandb-generated id, keeping the name")
            kwargs["id"] = None
            return wandb.init(**kwargs)

    def check_resume_compatible(self, saved_cfg):
        moved = {
            key: (saved_cfg.get(key, None), self.cfg.get(key, None))
            for key in self.resume_critical_keys
            if saved_cfg.get(key, None) != self.cfg.get(key, None)
        }
        if moved:
            raise ValueError(
                "latest.ckpt was written with different settings: " +
                ", ".join(f"{k} {was} -> {now}" for k, (was, now) in moved.items()) +
                ". Resuming would load weights of the wrong shape. Give this run its own "
                "checkpoint.run_tag (DP_RUN_TAG), or delete "
                f"{self.get_checkpoint_path()} to start it over.")

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                payload = self.load_checkpoint(path=lastest_ckpt_path)
                self.check_resume_compatible(payload["cfg"])
                print(f"Resumed at epoch {self.epoch} / global_step {self.global_step}")

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = create_dataloader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = create_dataloader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) //
            cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
            # EMAModel keeps its warmup counter outside the workspace, so it is
            # not in the checkpoint. A resumed job would restart it at 0, where
            # get_decay returns 0.0 and the very first step overwrites the
            # averaged weights with the live ones -- discarding the average that
            # is the thing actually served. One ema.step() per optimizer step is
            # what global_step counts.
            ema.optimization_step = self.global_step

        # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)
        env_runner = None

        # configure logging
        wandb_run = self.init_wandb(cfg)
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, "checkpoints"),
                                             **cfg.checkpoint.topk)

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")

        with JsonLogger(log_path) as json_logger:
            # From self.epoch, not 0: a resumed run finishes num_epochs in
            # total rather than running num_epochs again.
            for local_epoch_idx in range(self.epoch, cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dataset.postprocess(batch, device)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        # Proprioceptive noise, in the raw units the state is
                        # recorded in (radians / metres), applied before the
                        # policy's own normalizer sees it.
                        #
                        # The rollout is what asks for this. A policy trained
                        # without it is only ever shown joint configurations
                        # that lie exactly on a demonstration: measured on
                        # framehang's centralized checkpoint, 0.02 rad of noise
                        # on agent_pos -- one degree, less than the drift the
                        # rollout accumulates in its first 30 steps -- moves the
                        # predicted action's error by 12x and pushes 15% of the
                        # commanded dimensions onto clip_sample's box. The
                        # conditioning is valid on the demo manifold and
                        # nowhere else, so the first step off it is the last
                        # useful one. 0.0 (the default) is the behaviour that
                        # produced that measurement.
                        obs_noise = float(getattr(cfg.training, "obs_noise", 0.0) or 0.0)
                        if obs_noise > 0:
                            pos = batch["obs"]["agent_pos"]
                            batch["obs"] = {
                                **batch["obs"],
                                "agent_pos": pos + torch.randn_like(pos) * obs_noise,
                            }
                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if (self.global_step % cfg.training.gradient_accumulate_every == 0):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps
                                is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                                val_dataloader,
                                desc=f"Validation epoch {self.epoch}",
                                leave=False,
                                mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dataset.postprocess(batch, device)
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps
                                        is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log["val_loss"] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]

                        result = policy.predict_action(obs_dict)
                        pred_action = result["action_pred"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint
                # run_ckpt_dir() is anchored to the policy directory so eval can
                # resolve checkpoints/<bench>-<ckpt>-<env_cfg>-<action>-<seed>/
                # regardless of cwd.
                if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:
                    self.save_checkpoint(str(self.run_ckpt_dir() / f"{self.epoch + 1}.ckpt"))

                # ========= eval end for this epoch ==========
                policy.train()

                # What the batch size actually cost, so the next run's is a
                # measurement rather than a guess.
                if torch.cuda.is_available():
                    step_log["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

                # `latest.ckpt` is what training.resume picks up, so this
                # interval is how much work a preemption, a node failure or a
                # TIMEOUT costs. It is deliberately shorter than
                # checkpoint_every: the numbered checkpoints exist to be
                # evaluated, this one exists to be continued from.
                #
                # Written after the counters advance, so self.epoch means
                # "epochs finished" rather than "epoch in progress". Saving it
                # one line earlier makes the resumed run redo the epoch it
                # already has, and leaves a finished run repeating its last
                # epoch on every relaunch instead of exiting.
                resume_every = int(getattr(cfg.training, "resume_every", 0) or 0)
                if (cfg.checkpoint.get("save_last_ckpt", True) and resume_every > 0
                        and (self.epoch % resume_every) == 0):
                    self.save_latest_checkpoint()

        wandb_run.finish()


class BatchSampler:

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
    ):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[:-self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch


def create_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int = 0,
):
    batch_sampler = BatchSampler(len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True)

    def collate(x):
        assert len(x) == 1
        return x[0]

    dataloader = DataLoader(
        dataset,
        collate_fn=collate,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
    )
    return dataloader


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = RobotWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()