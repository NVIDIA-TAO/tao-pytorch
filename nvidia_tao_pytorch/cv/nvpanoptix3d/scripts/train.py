# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVPanoptix3D training script."""

import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.config.nvpanoptix3d.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.pl_model_2d import Mask2formerPlModule
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.pl_model_3d import NVPanoptix3DPlModule
from nvidia_tao_pytorch.cv.nvpanoptix3d.dataloader.pl_data_module import NVPanoptix3DDataModule


def run_experiment(experiment_config):
    """Start the training."""
    enable_3d = experiment_config.dataset.enable_3d
    model_path, trainer_kwargs = initialize_train_experiment(experiment_config)

    pl_data = NVPanoptix3DDataModule(experiment_config)
    if not experiment_config.train.iters_per_epoch:
        experiment_config.train.iters_per_epoch = len(pl_data.train_dataloader())

    # Select model based on enable_3d flag
    if enable_3d:
        pl_model = NVPanoptix3DPlModule(experiment_config)
        if model_path is None and experiment_config.train.checkpoint_2d:
            pl_model.load_2d_checkpoint(experiment_config.train.checkpoint_2d)
            logging.info("Loaded 2D checkpoint successfully.")
    else:
        pl_model = Mask2formerPlModule(experiment_config)

    if experiment_config.train.precision.lower() == "fp32":
        precision = "32-true"
    else:
        raise ValueError("Only fp32 precision is supported.")

    strategy = "auto"
    sync_batchnorm = False
    if len(trainer_kwargs["devices"]) > 1:
        distributed_strategy = experiment_config.train.distributed_strategy.lower()
        activation_checkpoint = experiment_config.train.activation_checkpoint
        if distributed_strategy == "ddp":
            if activation_checkpoint:
                strategy = "ddp"
            else:
                strategy = "ddp_find_unused_parameters_true"
        if enable_3d and "fan" in experiment_config.model.backbone:
            logging.info("Setting sync batch norm")
            sync_batchnorm = True

    trainer_kwargs["use_distributed_sampler"] = False
    trainer = Trainer(
        **trainer_kwargs,
        num_nodes=experiment_config.train.num_nodes,
        strategy=strategy, precision=precision,
        gradient_clip_val=experiment_config.train.clip_grad_norm,
        sync_batchnorm=sync_batchnorm,
        fast_dev_run=experiment_config.train.is_dry_run,
        max_steps=experiment_config.train.optim.max_steps,
        num_sanity_val_steps=0,
    )
    trainer.callbacks.append(LearningRateMonitor(logging_interval="step"))
    trainer.fit(pl_model, pl_data, ckpt_path=model_path)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="spec_front3d_3d", schema=ExperimentConfig,
)
@monitor_status(name="NVPanoptix3D", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
