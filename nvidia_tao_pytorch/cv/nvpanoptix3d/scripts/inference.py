# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVPanoptix3D inference script."""

import os
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.config.nvpanoptix3d.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.nvpanoptix3d.dataloader.pl_data_module import NVPanoptix3DDataModule
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.pl_model_2d import Mask2formerPlModule
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.pl_model_3d import NVPanoptix3DPlModule


def _run_inference_2d(experiment_config):
    """Run 2D inference"""
    model_path, trainer_kwargs = initialize_inference_experiment(experiment_config)
    trainer_kwargs["use_distributed_sampler"] = False
    trainer_kwargs["enable_checkpointing"] = False
    pl_data = NVPanoptix3DDataModule(experiment_config)
    pl_model = Mask2formerPlModule.load_from_checkpoint(
        model_path, map_location="cpu", cfg=experiment_config, strict=False,
    )
    trainer = Trainer(**trainer_kwargs)
    trainer.predict(pl_model, pl_data, return_predictions=False)


def _run_inference_3d(experiment_config):
    """Run 3D inference."""
    model_path, trainer_kwargs = initialize_inference_experiment(experiment_config)
    trainer_kwargs["use_distributed_sampler"] = False
    trainer_kwargs["enable_checkpointing"] = False
    pl_data = NVPanoptix3DDataModule(experiment_config)
    pl_model = NVPanoptix3DPlModule.load_from_checkpoint(
        model_path, map_location="cpu", cfg=experiment_config
    )
    trainer = Trainer(**trainer_kwargs)
    trainer.predict(pl_model, pl_data, return_predictions=False)


def run_experiment(experiment_config):
    """Dispatch based on enable_3d."""
    if experiment_config.dataset.enable_3d:
        _run_inference_3d(experiment_config)
    else:
        _run_inference_2d(experiment_config)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="spec_front3d_3d", schema=ExperimentConfig,
)
@monitor_status(name="NVPanoptix3D", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the inference process."""
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
