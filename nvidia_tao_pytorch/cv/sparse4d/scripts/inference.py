# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference of Sparse4D model."""

import os
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.config.sparse4d.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.sparse4d.dataloader.pl_sparse4d_data_module import Sparse4DDataModule
from nvidia_tao_pytorch.cv.sparse4d.model.sparse4d_pl_model import Sparse4DPlModel
from nvidia_tao_pytorch.cv.sparse4d.utils.misc import load_pretrained_weights


def run_experiment(experiment_config, key):
    """Start the inferencing."""
    _, trainer_kwargs = initialize_inference_experiment(experiment_config, key)

    # Instantiate the model
    model = Sparse4DPlModel(experiment_config)
    pretrained_path = experiment_config.inference.checkpoint

    if pretrained_path:
        logging.info(f"Loading checkpoint from: {pretrained_path}")
        new_state_dict = load_pretrained_weights(pretrained_path)
        model.load_state_dict(new_state_dict, strict=False)
        logging.info(f"Successfully loaded weights into Sparse4DPlModel from {pretrained_path}")

    dm = Sparse4DDataModule(experiment_config)

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
@monitor_status(name="Sparse4D", mode="inference")
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
