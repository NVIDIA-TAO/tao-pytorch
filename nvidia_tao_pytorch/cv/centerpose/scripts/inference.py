# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Inference on single patch. """

import os

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.centerpose.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.cv.centerpose.dataloader.pl_cp_data_module import CPDataModule
from nvidia_tao_pytorch.cv.centerpose.model.pl_centerpose_model import CenterPosePlModel


def run_experiment(experiment_config, key):
    """Start the inference."""
    model_path, trainer_kwargs = initialize_inference_experiment(experiment_config, key)

    if model_path.endswith('.tlt') or model_path.endswith('.pth'):

        # build data module
        dm = CPDataModule(experiment_config.dataset)
        dm.setup(stage="predict")

        # Run inference using tlt model
        model = CenterPosePlModel.load_from_checkpoint(model_path,
                                                       map_location="cpu",
                                                       experiment_spec=experiment_config)

        trainer = Trainer(**trainer_kwargs)

        trainer.predict(model, datamodule=dm)

    elif model_path.endswith('.engine'):
        raise NotImplementedError("TensorRT inference is supported through tao-deploy. "
                                  "Please use tao-deploy to generate TensorRT enigne and run inference.")
    else:
        raise NotImplementedError("Model path format is only supported for .tlt or .pth")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="infer", schema=ExperimentConfig
)
@monitor_status(name="Centerpose", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the inference process."""
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
