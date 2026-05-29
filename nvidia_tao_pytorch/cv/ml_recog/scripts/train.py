# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train metric-learning recognition model."""

import math
import os

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.ml_recog.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.cv.ml_recog.dataloader.pl_ml_data_module import MLDataModule
from nvidia_tao_pytorch.cv.ml_recog.model.pl_ml_recog_model import MLRecogModel


def run_experiment(experiment_config):
    """Starts the training.

    Args:
        experiment_config (DictConfig): Configuration dictionary

    """
    resume_ckpt, trainer_kwargs = initialize_train_experiment(experiment_config)

    # update experiment_config in Trainer
    experiment_config['train']['resume_training_checkpoint_path'] = resume_ckpt

    dm = MLDataModule(experiment_config)
    dm.setup('fit')

    ml_recog = MLRecogModel(
        experiment_config,
        dm,
        subtask="train")

    clip_grad = experiment_config['train']['clip_grad_norm']

    # See REID for why we do this
    num_batches = len(dm.train_dataloader())
    val_check_interval = math.floor(((num_batches - 1) / num_batches) * 100) / 100

    trainer = Trainer(**trainer_kwargs,
                      val_check_interval=val_check_interval,
                      num_sanity_val_steps=0,
                      strategy='auto',
                      gradient_clip_val=clip_grad)

    trainer.fit(ml_recog, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="train", schema=ExperimentConfig
)
@monitor_status(name="Metric Learning Recognition", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process.

    Args:
        cfg (DictConfig): Hydra config object.
    """
    obfuscate_logs(cfg)

    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
