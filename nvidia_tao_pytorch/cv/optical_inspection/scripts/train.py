# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train Optical Inspection Siamese Network model."""
import os
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.config.optical_inspection.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.pl_oi_data_module import OIDataModule
from nvidia_tao_pytorch.cv.optical_inspection.model.pl_oi_model import OpticalInspectionModel


def run_experiment(experiment_config, key):
    """Start the training."""
    resume_ckpt, trainer_kwargs = initialize_train_experiment(experiment_config, key)

    dm = OIDataModule(experiment_config)

    # Load pretrained model as starting point if pretrained path is provided,
    pretrained_path = experiment_config.train.pretrained_model_path
    if pretrained_path:
        oi_model = OpticalInspectionModel.load_from_checkpoint(pretrained_path,
                                                               map_location="cpu",
                                                               experiment_spec=experiment_config,
                                                               dm=dm)
    else:
        oi_model = OpticalInspectionModel(experiment_config, dm)

    clip_grad = experiment_config['train']['clip_grad_norm']
    enable_tensorboard = experiment_config.train.tensorboard.enabled

    if enable_tensorboard:
        infrequent_logging_frequency = experiment_config.train.tensorboard.infrequent_logging_frequency
        assert max(0, infrequent_logging_frequency) <= trainer_kwargs['max_epochs'], (
            f"infrequent_logging_frequency {infrequent_logging_frequency} must be < num_epochs {trainer_kwargs['max_epochs']}"
        )
        logging.info("Tensorboard logging enabled.")
    else:
        logging.info("Tensorboard logging disabled.")
    acc_flag = 'auto'
    trainer = Trainer(
        **trainer_kwargs,
        strategy=acc_flag,
        gradient_clip_val=clip_grad,
    )

    trainer.fit(oi_model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name="Optical Inspection", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
