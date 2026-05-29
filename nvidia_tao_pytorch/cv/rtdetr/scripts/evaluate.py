# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a trained rt-detr model."""
import os
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment

from nvidia_tao_pytorch.config.rtdetr.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.rtdetr.dataloader.pl_od_data_module import ODDataModule
from nvidia_tao_pytorch.core.quantization.utils import create_quantized_model_from_config
from nvidia_tao_pytorch.cv.rtdetr.model.pl_rtdetr_model import RTDETRPlModel


def run_experiment(experiment_config):
    """Run experiment."""
    model_path, trainer_kwargs = initialize_evaluation_experiment(experiment_config)

    if model_path.endswith('.pth'):
        # build dataloader
        dm = ODDataModule(experiment_config.dataset, subtask_config=experiment_config.evaluate)
        dm.setup(stage="test")

        # build model and load from the given checkpoint (supports quantized models)
        if experiment_config.evaluate.is_quantized:
            model = create_quantized_model_from_config(model_path, RTDETRPlModel, experiment_config=experiment_config)
        else:
            model = RTDETRPlModel.load_from_checkpoint(
                model_path,
                map_location="cpu",
                experiment_spec=experiment_config,
                export=False,
            )

        trainer = Trainer(**trainer_kwargs)

        trainer.test(model, datamodule=dm)

    elif model_path.endswith('.engine'):
        raise NotImplementedError("TensorRT evaluation is supported through tao-deploy. "
                                  "Please use tao-deploy to generate TensorRT enigne and run evaluation.")
    else:
        raise NotImplementedError("Model path format is only supported for .tlt or .pth")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="evaluate", schema=ExperimentConfig
)
@monitor_status(name="RT-DETR", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """Run the evaluate process."""
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
