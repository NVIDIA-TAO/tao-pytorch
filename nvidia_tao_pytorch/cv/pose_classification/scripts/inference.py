# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference on single patch."""
import logging
import os
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.config.pose_classification.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.pose_classification.dataloader.pl_pc_data_module import PCDataModule
from nvidia_tao_pytorch.cv.pose_classification.model.pl_pc_model import PoseClassificationModel

logger = logging.getLogger(__name__)


def run_experiment(experiment_config, key):
    """
    Start the inference process.

    This function initializes the necessary components for inference, including the model, data loader,
    and inferencer. It performs inference on the provided data and saves the results in the specified output file.

    Args:
        experiment_config (dict): The experiment configuration containing the model and inference parameters.
        key (str): The encryption key for intermediate checkpoints.

    Raises:
        Exception: If any error occurs during the inference process.
    """
    model_path, trainer_kwargs = initialize_inference_experiment(experiment_config, key)
    if len(trainer_kwargs['devices']) > 1:
        trainer_kwargs['devices'] = [trainer_kwargs['devices'][0]]
        logger.info(f"Pose Classification does not support multi-GPU inference at this time. Using only GPU {trainer_kwargs['devices']}")

    dm = PCDataModule(experiment_config)
    model = PoseClassificationModel.load_from_checkpoint(model_path,
                                                         map_location="cpu",
                                                         experiment_spec=experiment_config)

    trainer = Trainer(**trainer_kwargs)

    trainer.predict(model, datamodule=dm)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name="Pose Classification", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """
    Run the inference process.

    This function serves as the entry point for the inference script.
    It loads the experiment specification, obfuscates logs, updates the results directory, and calls the 'run_experiment' function.

    Args:
        cfg (ExperimentConfig): The experiment configuration retrieved from the Hydra configuration files.
    """
    # Obfuscate logs.
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
