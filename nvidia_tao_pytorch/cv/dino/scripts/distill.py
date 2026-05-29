# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train DINO model."""

import os
from pytorch_lightning import LightningModule

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner

from nvidia_tao_pytorch.config.dino.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.dino.distillation.distiller import DINODistiller as DINOPlModel

from nvidia_tao_pytorch.cv.dino.scripts.train import run_experiment

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="distill", schema=ExperimentConfig
)
@monitor_status(name="DINO", mode="distill")
def main(cfg: ExperimentConfig) -> None:
    """Run the distillation process."""
    # Override specific to distillation to match student and teacher features
    if cfg.dataset.augmentation.random_resize_max_size != 1344 or cfg.dataset.augmentation.fixed_padding is not True:
        cfg.dataset.augmentation.random_resize_max_size = 1344
        cfg.dataset.augmentation.fixed_padding = True
        status_logging.get_status_logger().write(
            verbosity_level=status_logging.Verbosity.INFO,
            message="Overriding random resize max size to recommended 1344 and enabling fixed padding for distillation"
        )

    # This is for resuming to work without needing to save the teacher weights
    LightningModule.strict_loading = False

    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key,
                   lightning_module=DINOPlModel)


if __name__ == "__main__":
    main()
