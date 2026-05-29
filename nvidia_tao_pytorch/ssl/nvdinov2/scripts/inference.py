# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Inference of NVDINOv2 SSL.
"""
import os

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.core.tlt_logging import logging, obfuscate_logs
from nvidia_tao_pytorch.config.nvdinov2.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.nvdinov2.dataloader.pl_dinov2_data_module import DinoV2DataModule
from nvidia_tao_pytorch.ssl.nvdinov2.model.pl_model import DinoV2PlModel


def run_experiment(experiment_config, key):
    """Start the inference."""
    model_path, trainer_kwargs = initialize_inference_experiment(experiment_config, key)

    precision = experiment_config.train.precision

    dm = DinoV2DataModule(experiment_config)

    model = DinoV2PlModel(experiment_config)

    if model_path is not None and (model_path.endswith('.tlt') or model_path.endswith('.pth')):
        model.pretrained_weights = model_path
        model.restore_pretrained_weights()
        logging.info("loading model from {model_path}".format(model_path=model_path))
    else:
        raise NotImplementedError("Model path format is only supported for .tlt or .pth")

    trainer = Trainer(**trainer_kwargs,
                      precision=precision
                      )

    trainer.predict(model, dm)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="NVDINOv2", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
