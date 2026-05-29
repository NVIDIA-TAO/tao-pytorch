# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train Sparse4D model."""

import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs, logging
from nvidia_tao_pytorch.config.sparse4d.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.sparse4d.dataloader.pl_sparse4d_data_module import Sparse4DDataModule
from nvidia_tao_pytorch.cv.sparse4d.dataloader.callbacks import PklResampleCallback
from nvidia_tao_pytorch.cv.sparse4d.model.sparse4d_pl_model import Sparse4DPlModel
from nvidia_tao_pytorch.cv.sparse4d.utils.misc import load_pretrained_weights


def run_experiment(experiment_config, key):
    """Start the training."""
    # results_dir, resume_ckpt, gpus, ptl_loggers = initialize_train_experiment(experiment_config, key)
    resume_ckpt, trainer_kwargs = initialize_train_experiment(experiment_config, key)

    num_nodes = experiment_config.train.num_nodes

    # Load pretrained model as starting point if pretrained path is provided
    pretrained_path = experiment_config.train.pretrained_model_path

    precision = experiment_config.train.precision
    if precision.lower() == 'fp16':
        precision = '16-mixed'
    elif precision.lower() == 'bf16':
        precision = 'bf16-mixed'
    elif precision.lower() == 'fp32':
        precision = '32-true'
    else:
        raise NotImplementedError(f"{precision} is not supported. \
                                  Only bf16, fp16, and fp32 are supported")

    sync_batchnorm = True

    dm = Sparse4DDataModule(experiment_config)
    batch_size = experiment_config.dataset.batch_size
    num_frames = experiment_config.dataset.num_frames
    num_gpus = experiment_config.train.num_gpus
    num_bev_groups = experiment_config.dataset.num_bev_groups
    num_epochs = experiment_config.train.num_epochs
    num_iters_per_epoch = int(num_frames * num_bev_groups // (num_nodes * num_gpus * batch_size))
    grad_clip = experiment_config.train.optim.grad_clip.max_norm

    # Instantiate the model
    model = Sparse4DPlModel(experiment_config)
    logging.info(model)

    if pretrained_path:
        logging.info(f"Loading checkpoint from: {pretrained_path}")
        new_state_dict = load_pretrained_weights(pretrained_path)
        model.load_state_dict(new_state_dict, strict=False)
        logging.info(f"Successfully loaded weights into Sparse4DPlModel from {pretrained_path}")

    strategy = 'auto'
    if len(trainer_kwargs['devices']) > 1:
        strategy = 'ddp_find_unused_parameters_true'

    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [lr_monitor]
    pkl_sample_size = experiment_config.dataset.get("pkl_sample_size", 0)
    if pkl_sample_size > 0:
        callbacks.append(PklResampleCallback(num_iters_per_epoch=num_iters_per_epoch))
        logging.info(f"PklResampleCallback enabled (interval={num_iters_per_epoch} steps)")

    trainer = Trainer(
        **trainer_kwargs,
        num_nodes=num_nodes,
        max_steps=num_iters_per_epoch * num_epochs,
        limit_train_batches=num_iters_per_epoch,
        reload_dataloaders_every_n_epochs=0,
        log_every_n_steps=50,
        num_sanity_val_steps=0,
        strategy=strategy,
        precision=precision,
        use_distributed_sampler=False,
        sync_batchnorm=sync_batchnorm,
        callbacks=callbacks,
        gradient_clip_val=grad_clip,
    )

    trainer.fit(model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="Sparse4D", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)
    run_experiment(
        experiment_config=cfg,
        key=cfg.encryption_key
    )


if __name__ == "__main__":
    main()
