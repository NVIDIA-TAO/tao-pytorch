# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mask2former inference script."""
import os

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner

from nvidia_tao_pytorch.config.mask2former.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.mask2former.dataloader.pl_data_module import SemSegmDataModule
from nvidia_tao_pytorch.cv.mask2former.model.pl_model import Mask2formerPlModule


def run_experiment(experiment_config):
    """Start the inference."""
    model_path, trainer_kwargs = initialize_inference_experiment(experiment_config)
    pl_data = SemSegmDataModule(experiment_config.dataset)

    # Run inference using TAO model
    # pl_model = Mask2formerPlModule(experiment_config)
    # pl_model.load_pretrained_weights(model_path)
    pl_model = Mask2formerPlModule.load_from_checkpoint(
        model_path,
        map_location="cpu",
        cfg=experiment_config)

    trainer = Trainer(**trainer_kwargs)

    trainer.predict(pl_model, pl_data, return_predictions=False)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="spec", schema=ExperimentConfig
)
@monitor_status(name="Mask2Former", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the inference process."""
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
