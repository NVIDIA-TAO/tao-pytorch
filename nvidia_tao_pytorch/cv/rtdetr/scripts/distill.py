# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distill RT-DETR model."""

import os
from pytorch_lightning import LightningModule

from nvidia_tao_pytorch.config.rtdetr.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.rtdetr.distillation.distiller import RtdetrDistiller
from nvidia_tao_pytorch.cv.rtdetr.scripts.train import run_experiment

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="distill", schema=ExperimentConfig
)
@monitor_status(name="RTDETR", mode="distill")
def main(cfg: ExperimentConfig) -> None:
    """Run the distillation process."""
    # This is for resuming to work without needing to save the teacher weights
    LightningModule.strict_loading = False

    run_experiment(experiment_config=cfg,
                   lightning_module=RtdetrDistiller)


if __name__ == "__main__":
    main()
