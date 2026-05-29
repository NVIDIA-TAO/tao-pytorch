# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference of Segformer model."""

import os
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.config.segformer.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.segformer.dataloader.pl_segformer_data_module import SFDataModule
from nvidia_tao_pytorch.cv.segformer.model.segformer_pl_model import SegFormerPlModel


def run_experiment(experiment_config, key):
    """Start the inferencing."""
    model_path, trainer_kwargs = initialize_inference_experiment(experiment_config, key)

    dm = SFDataModule(experiment_config.dataset.segment)
    dm.setup(stage="predict")

    model = SegFormerPlModel.load_from_checkpoint(
        model_path,
        map_location="cpu",
        experiment_spec=experiment_config,
        dm=dm
    )
    trainer = Trainer(
        **trainer_kwargs
    )

    trainer.predict(model, datamodule=dm)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="SegFormer", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)
    run_experiment(
        experiment_config=cfg,
        key=cfg.encryption_key
    )


if __name__ == "__main__":
    main()
