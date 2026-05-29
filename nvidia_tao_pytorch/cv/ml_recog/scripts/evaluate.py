# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a trained Metric Learning Recognition model."""

import os

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.ml_recog.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment
from nvidia_tao_pytorch.cv.ml_recog.dataloader.pl_ml_data_module import MLDataModule
from nvidia_tao_pytorch.cv.ml_recog.model.pl_ml_recog_model import MLRecogModel
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs


def run_experiment(experiment_config):
    """Starts the evaluate.

    Args:
        experiment_config (DictConfig): Configuration dictionary

    """
    model_path, trainer_kwargs = initialize_evaluation_experiment(experiment_config)

    dm = MLDataModule(experiment_config)
    # Trigger setup() here so that get_query_accuracy() below will have access to necessary variables
    dm.setup(stage='test')

    ml_recog = MLRecogModel.load_from_checkpoint(
        model_path,
        map_location="cpu",
        experiment_spec=experiment_config,
        dm=dm,
        subtask="evaluate")

    trainer = Trainer(**trainer_kwargs)

    trainer.test(ml_recog, datamodule=dm)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="evaluate", schema=ExperimentConfig
)
@monitor_status(name="Metric Learning Recognition", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process.

    Args:
        cfg (DictConfig): Hydra config object.

    """
    obfuscate_logs(cfg)
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
