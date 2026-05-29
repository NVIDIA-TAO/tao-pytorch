# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVPanoptix3D evaluation script."""

import os
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.config.nvpanoptix3d.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.nvpanoptix3d.dataloader.pl_data_module import NVPanoptix3DDataModule
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.pl_model_2d import Mask2formerPlModule
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.pl_model_3d import NVPanoptix3DPlModule


def _run_evaluate_2d(experiment_config):
    """Evaluate 2D model."""
    model_path, trainer_kwargs = initialize_evaluation_experiment(experiment_config)
    pl_data = NVPanoptix3DDataModule(experiment_config)
    pl_model = Mask2formerPlModule.load_from_checkpoint(
        model_path, map_location="cpu", cfg=experiment_config
    )
    trainer_kwargs["use_distributed_sampler"] = False
    trainer_kwargs["enable_checkpointing"] = False
    trainer = Trainer(**trainer_kwargs)
    trainer.test(pl_model, datamodule=pl_data)


def _run_evaluate_3d(experiment_config):
    """Evaluate 3D model."""
    model_path, trainer_kwargs = initialize_evaluation_experiment(experiment_config)
    pl_data = NVPanoptix3DDataModule(experiment_config)
    pl_model = NVPanoptix3DPlModule.load_from_checkpoint(
        model_path, map_location="cpu", cfg=experiment_config
    )
    trainer_kwargs["use_distributed_sampler"] = False
    trainer_kwargs["enable_checkpointing"] = False
    trainer = Trainer(**trainer_kwargs)
    trainer.test(pl_model, datamodule=pl_data)


def run_experiment(experiment_config):
    """Dispatch based on enable_3d."""
    if experiment_config.dataset.enable_3d:
        _run_evaluate_3d(experiment_config)
    else:
        _run_evaluate_2d(experiment_config)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="spec_front3d_3d", schema=ExperimentConfig,
)
@monitor_status(name="NVPanoptix3D", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """Run the evaluation process."""
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
