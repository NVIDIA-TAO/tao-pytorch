# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Inference on single patch. """
import os

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner

from nvidia_tao_pytorch.config.mask_grounding_dino.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.mask_grounding_dino.dataloader.od_data_module import ODVGDataModule
from nvidia_tao_pytorch.cv.mask_grounding_dino.model.pl_gdino_model import MaskGDINOPlModel


def run_experiment(experiment_config):
    """Start the inference."""
    model_path, trainer_kwargs = initialize_inference_experiment(experiment_config)
    # build dataset and model for mask branch
    experiment_config.dataset.has_mask = True
    experiment_config.model.has_mask = True
    if not model_path:
        raise FileNotFoundError("inference.checkpoint is not set!")

    # tlt inference
    if model_path.endswith('.pth'):

        # build data module
        dm = ODVGDataModule(experiment_config.dataset, subtask_config=experiment_config.inference)
        dm.setup(stage="predict")
        cap_lists = dm.pred_dataset.cap_lists

        # Run inference using tlt model
        model = MaskGDINOPlModel.load_from_checkpoint(
            model_path,
            map_location="cpu",
            experiment_spec=experiment_config,
            cap_lists=cap_lists)

        trainer = Trainer(**trainer_kwargs)

        trainer.predict(model, datamodule=dm, return_predictions=False)
    elif model_path.endswith('.engine'):
        raise NotImplementedError("TensorRT inference is supported through tao-deploy. "
                                  "Please use tao-deploy to generate TensorRT enigne and run inference.")
    else:
        raise NotImplementedError("Model path format is only supported for .pth")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="infer", schema=ExperimentConfig
)
@monitor_status(name="Mask Grounding DINO", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the inference process."""
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
