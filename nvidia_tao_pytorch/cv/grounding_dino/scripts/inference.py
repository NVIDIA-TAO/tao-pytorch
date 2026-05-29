# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Inference on single patch. """
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import tempfile

from nvidia_tao_pytorch.config.grounding_dino.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment

from nvidia_tao_pytorch.cv.grounding_dino.dataloader.pl_odvg_data_module import ODVGDataModule
from nvidia_tao_pytorch.cv.grounding_dino.model.pl_gdino_model import GDINOPlModel
from nvidia_tao_pytorch.cv.grounding_dino.model.utils import grounding_dino_parser


def run_experiment(experiment_config):
    """Start the inference."""
    model_path, trainer_kwargs = initialize_inference_experiment(experiment_config)

    if model_path.endswith('.pth'):

        # build data module
        dm = ODVGDataModule(experiment_config.dataset, subtask_config=experiment_config.inference)
        dm.setup(stage="predict")
        cap_lists = dm.pred_dataset.cap_lists

        # Check if the checkpoint is coming from TAO PTL trained or not
        original = torch.load(model_path, map_location="cpu")
        if "pytorch-lightning_version" not in original:
            # parse public checkpoint
            final = grounding_dino_parser(original["model"])
            tmp = tempfile.NamedTemporaryFile()
            model_path = tmp.name
            torch.save({"state_dict": final, 'pytorch-lightning_version': pl.__version__}, model_path)

        model = GDINOPlModel.load_from_checkpoint(model_path,
                                                  map_location="cpu",
                                                  experiment_spec=experiment_config,
                                                  cap_lists=cap_lists)

        trainer = Trainer(**trainer_kwargs)

        trainer.predict(model, datamodule=dm)

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
@monitor_status(name="Grounding DINO", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the inference process."""
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
