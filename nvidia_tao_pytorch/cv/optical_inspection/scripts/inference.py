# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Inference on inspection images
"""
import os
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.config.optical_inspection.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.pl_oi_data_module import OIDataModule
from nvidia_tao_pytorch.cv.optical_inspection.model.pl_oi_model import OpticalInspectionModel


def run_experiment(experiment_config, key):
    """Start the inference."""
    model_path, trainer_kwargs = initialize_inference_experiment(experiment_config, key)

    dm = OIDataModule(experiment_config)

    model = OpticalInspectionModel.load_from_checkpoint(
        model_path,
        map_location="cpu",
        experiment_spec=experiment_config,
        dm=dm
    )

    trainer = Trainer(**trainer_kwargs)

    trainer.predict(model, datamodule=dm)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name="Optical Inspection", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
