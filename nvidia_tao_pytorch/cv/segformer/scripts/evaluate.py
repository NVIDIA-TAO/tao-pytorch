# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluation of Segformer model.
"""
import os
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.config.segformer.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.segformer.dataloader.pl_segformer_data_module import SFDataModule
from nvidia_tao_pytorch.cv.segformer.model.segformer_pl_model import SegFormerPlModel


def run_experiment(experiment_config, key):
    """Run experiment."""
    model_path, trainer_kwargs = initialize_evaluation_experiment(experiment_config, key)

    if model_path.endswith('.tlt') or model_path.endswith('.pth'):
        # build dataloader
        dm = SFDataModule(experiment_config.dataset.segment)
        dm.setup(stage="test")

        # build model and load from the given checkpoint
        model = SegFormerPlModel.load_from_checkpoint(
            model_path,
            map_location="cpu",
            experiment_spec=experiment_config
        )

    elif model_path.endswith('.engine'):
        raise NotImplementedError("TensorRT evaluation is supported through tao-deploy. Please use tao-deploy to generate TensorRT engine and run evaluation.")

    else:
        raise NotImplementedError("Model path format is only supported for .tlt or .pth")

    trainer = Trainer(**trainer_kwargs)

    trainer.test(model, datamodule=dm)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="Segformer", mode="evaluate")
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
