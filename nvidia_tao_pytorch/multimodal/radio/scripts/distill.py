# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-teacher distillation for RADIO."""

import os

from omegaconf import OmegaConf
from pytorch_lightning import LightningModule, Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.tlt_logging import logging, obfuscate_logs
from nvidia_tao_pytorch.core.utilities import get_latest_checkpoint
from nvidia_tao_pytorch.multimodal.radio.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.multimodal.radio.dataloader.radio_data_module import RadioDataModule
from nvidia_tao_pytorch.multimodal.radio.distillation.distiller import (
    MultiTeacherDistiller,
    _resolve_distillation_runtime_modes,
)


_RESUME_CONFIG_FILENAME = "resume_experiment.yaml"


def _resume_config_path(results_dir):
    """Return the frozen resolved config path for interruption resumes."""
    return os.path.join(results_dir, _RESUME_CONFIG_FILENAME)


def _resolve_resume_config(experiment_config):
    """Use the original resolved config for implicit checkpoint resumes.

    SLURM auto-requeue restarts the Python process and Hydra re-reads the YAML
    from disk. Keep implicit resumes tied to the config that produced the
    checkpoint so edits to the spec file do not change loss/data behavior
    mid-run. Explicit resume paths still use the caller-provided config.
    """
    results_dir = experiment_config["results_dir"]
    explicit_resume = experiment_config["train"]["resume_training_checkpoint_path"]
    auto_resume_ckpt = None if explicit_resume else get_latest_checkpoint(results_dir)
    frozen_config_path = _resume_config_path(results_dir)

    if auto_resume_ckpt and os.path.exists(frozen_config_path):
        logging.info("Loading frozen RADIO resume config from %s", frozen_config_path)
        return OmegaConf.load(frozen_config_path)

    os.makedirs(results_dir, exist_ok=True)
    if auto_resume_ckpt:
        logging.warning(
            "Implicit RADIO resume from %s has no frozen config at %s; "
            "continuing with the current config.",
            auto_resume_ckpt,
            frozen_config_path,
        )

    OmegaConf.save(config=experiment_config, f=frozen_config_path)
    logging.info("Saved frozen RADIO resume config to %s", frozen_config_path)
    return experiment_config


def run_experiment(experiment_config, key):
    """Start the distillation training."""
    experiment_config = _resolve_resume_config(experiment_config)
    resume_ckpt, trainer_kwargs = initialize_train_experiment(experiment_config, key)

    num_nodes = experiment_config.train.num_nodes
    clip_grad_norm = experiment_config.train.clip_grad_norm
    sync_batchnorm, broadcast_buffers, _ = (
        _resolve_distillation_runtime_modes(experiment_config.distill)
    )

    if experiment_config.train.precision.lower() == 'fp16':
        precision = '16-mixed'
    elif experiment_config.train.precision.lower() == 'bf16':
        precision = 'bf16-mixed'
    elif experiment_config.train.precision.lower() == 'fp32':
        precision = '32-true'
    else:
        raise NotImplementedError(
            f"{experiment_config.train.precision} is not supported. Only bf16, fp16, and fp32 are supported")

    dm = RadioDataModule(experiment_config.dataset, experiment_config=experiment_config)
    # Lightning calls ``setup("fit")`` after DDP initializes the process group.
    # Rank-aware partitioning must not build its loader before then.

    # Resuming without needing to save teacher weights
    LightningModule.strict_loading = False
    model = MultiTeacherDistiller(experiment_config)

    strategy = 'auto'
    if len(trainer_kwargs['devices']) > 1 or num_nodes > 1:
        # Use an explicit DDPStrategy with a long (2h) NCCL timeout instead of the
        # bare 'ddp_find_unused_parameters_true' string, which builds DDPStrategy
        # with the default ~10-min process-group timeout. At 256 GPUs the post-save
        # NCCL barrier blocks all ranks while rank0 serially writes the large
        # (~38 GB) checkpoint to Lustre every epoch; the heavy teacher config
        # (DINOv3-7B + SAM3) crossed the default timeout and hung at the epoch
        # boundary. Mirrors multimodal/clip/scripts/train.py; semantics-preserving.
        from datetime import timedelta
        from pytorch_lightning.strategies import DDPStrategy
        # Disable rank-0 buffer broadcast for partitioned per-resolution BatchNorm.
        strategy = DDPStrategy(
            timeout=timedelta(hours=2),
            find_unused_parameters=True,
            broadcast_buffers=broadcast_buffers,
        )

    dump_batches = os.environ.get("RADIO_DUMP_BATCHES")
    if dump_batches:
        trainer_kwargs["limit_train_batches"] = int(dump_batches)
        trainer_kwargs["limit_val_batches"] = 0
        trainer_kwargs["max_epochs"] = 1

    # Skip startup sanity-check validation. initialize_train_experiment may
    # already provide this key, so keep it in trainer_kwargs to avoid passing a
    # duplicate Trainer argument.
    trainer_kwargs["num_sanity_val_steps"] = 0

    trainer = Trainer(
        **trainer_kwargs,
        gradient_clip_val=clip_grad_norm,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        use_distributed_sampler=False,
        # Partitioned training always keeps BN rank-local during training.
        sync_batchnorm=sync_batchnorm,
    )

    trainer.fit(model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="distill",
    schema=ExperimentConfig,
)
@monitor_status(name="Class_pt", mode="distill")
def main(cfg: ExperimentConfig) -> None:
    """Run the distillation process."""
    obfuscate_logs(cfg)
    run_experiment(experiment_config=cfg, key=cfg.encryption_key)


if __name__ == "__main__":
    main()
