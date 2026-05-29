# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a trained re-identification model."""
import logging
import os
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment
from nvidia_tao_pytorch.config.re_identification.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.re_identification.dataloader.pl_reid_data_module import REIDDataModule
from nvidia_tao_pytorch.cv.re_identification.model.pl_reid_model import ReIdentificationModel

logger = logging.getLogger(__name__)


def run_experiment(experiment_config, key):
    """
    Run the evaluation process.

    This function initializes the necessary components for evaluation, including the model, data loader,
    and inferencer. It performs evaluation on the test dataset and computes evaluation metrics.

    Args:
        experiment_config (dict): The experiment configuration containing the model and evaluation parameters.
        key (str): The encryption key for intermediate checkpoints.

    Raises:
        Exception: If any error occurs during the evaluation process.
    """
    model_path, trainer_kwargs = initialize_evaluation_experiment(experiment_config, key)
    if len(trainer_kwargs['devices']) > 1:
        trainer_kwargs['devices'] = [trainer_kwargs['devices'][0]]
        logger.info(f"Re-Identification does not support multi-GPU evaluation at this time. Using only GPU {trainer_kwargs['devices']}")

    dm = REIDDataModule(experiment_config)
    model = ReIdentificationModel.load_from_checkpoint(model_path,
                                                       map_location="cpu",
                                                       experiment_spec=experiment_config,
                                                       prepare_for_training=False)

    if "swin" in experiment_config.model.backbone:
        model.model.load_param(model_path)

    trainer = Trainer(**trainer_kwargs)

    trainer.test(model, datamodule=dm)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name="ReIdentification", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """
    Run the evaluation process.

    This function serves as the entry point for the evaluation script.
    It loads the experiment specification, updates the results directory, and calls the 'run_experiment' function.

    Args:
        cfg (ExperimentConfig): The experiment configuration retrieved from the Hydra configuration files.
    """
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
