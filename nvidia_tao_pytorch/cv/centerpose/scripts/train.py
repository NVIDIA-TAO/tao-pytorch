# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train CenterPose model."""

import os

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.centerpose.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.cv.centerpose.model.pl_centerpose_model import CenterPosePlModel
from nvidia_tao_pytorch.cv.centerpose.dataloader.pl_cp_data_module import CPDataModule


def run_experiment(experiment_config, key):
    """Start the training."""
    resume_ckpt, trainer_kwargs = initialize_train_experiment(experiment_config, key)

    dm = CPDataModule(experiment_config.dataset)

    # Load pretrained model as starting point if pretrained path is provided,
    pretrained_path = experiment_config.train.pretrained_model_path
    if pretrained_path not in (None, ""):
        pt_model = CenterPosePlModel.load_from_checkpoint(pretrained_path,
                                                          map_location="cpu",
                                                          experiment_spec=experiment_config)
    else:
        pt_model = CenterPosePlModel(experiment_config)

    clip_grad_val = experiment_config.train.clip_grad_val
    is_dry_run = experiment_config.train.is_dry_run

    if experiment_config.train.precision.lower() == 'fp16':
        precision = '16-mixed'
    elif experiment_config.train.precision.lower() == 'fp32':
        precision = '32-true'
    else:
        raise NotImplementedError(f"{experiment_config.train.precision} is not supported. Only fp32 and fp16 are supported")

    trainer = Trainer(**trainer_kwargs,
                      strategy='auto',
                      precision=precision,
                      gradient_clip_val=clip_grad_val,
                      use_distributed_sampler=False,
                      sync_batchnorm=False,
                      fast_dev_run=is_dry_run)

    trainer.fit(pt_model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="train", schema=ExperimentConfig
)
@monitor_status(name='Centerpose', mode='train')
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
