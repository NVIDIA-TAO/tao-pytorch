# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MAE evaluation script."""
import logging
import os
import warnings

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment

from nvidia_tao_pytorch.config.mae.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.mae.dataloader.pl_data_module import MAEDataModule
from nvidia_tao_pytorch.ssl.mae.model.pl_model import MAEPlModule

warnings.filterwarnings("ignore")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level='INFO')
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="eval", schema=ExperimentConfig
)
@monitor_status(name="MAE", mode="evaluate")
def run_experiment(cfg: ExperimentConfig) -> None:
    """Run training experiment."""
    model_path, trainer_kwargs = initialize_evaluation_experiment(cfg)

    logger.info("Setting up dataloader...")
    data_module = MAEDataModule(cfg=cfg)

    logger.info("Building MAE models...")

    pl_model = MAEPlModule.load_from_checkpoint(
        model_path,
        map_location="cpu",
        cfg=cfg)

    trainer = Trainer(**trainer_kwargs)

    trainer.test(pl_model, data_module)


if __name__ == '__main__':
    run_experiment()
