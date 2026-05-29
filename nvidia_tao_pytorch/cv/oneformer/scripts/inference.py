# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" OneFormer inference script."""
import os

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner

from nvidia_tao_pytorch.config.oneformer.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.oneformer.dataloader.pl_data_module import SemSegmDataModule
from nvidia_tao_pytorch.cv.oneformer.model.pl_oneformer import OneformerPlModule


def run_experiment(experiment_config):
    """Start the inference."""
    model_path, trainer_kwargs = initialize_inference_experiment(experiment_config)
    pl_data = SemSegmDataModule(experiment_config)

    pl_model = OneformerPlModule.load_from_checkpoint(
        model_path,
        map_location="cpu",
        cfg=experiment_config)

    trainer = Trainer(**trainer_kwargs)

    trainer.predict(pl_model, pl_data, return_predictions=False)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="spec_coco", schema=ExperimentConfig
)
@monitor_status(name="OneFormer", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the inference process."""
    run_experiment(cfg)


if __name__ == "__main__":
    main()
