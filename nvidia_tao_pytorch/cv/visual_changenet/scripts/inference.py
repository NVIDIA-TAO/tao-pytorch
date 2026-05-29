# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluation of Visual ChangeNet model.
"""
import os

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.pl_oi_data_module import OIDataModule
from nvidia_tao_pytorch.config.visual_changenet.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.dataloader.pl_changenet_data_module import CNDataModule
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models.cn_pl_model import ChangeNetPlModel as ChangeNetPlSegment
from nvidia_tao_pytorch.cv.visual_changenet.classification.models.cn_pl_model import ChangeNetPlModel as ChangeNetPlClassifier

from pytorch_lightning import Trainer


def run_experiment(experiment_config, key):
    """Start the inference."""
    model_path, trainer_kwargs = initialize_inference_experiment(experiment_config, key)

    task = experiment_config.task

    assert task in ['segment', 'classify'], "Visual ChangeNet only supports 'segment' and 'classify' tasks."
    if task == 'segment':
        if model_path.endswith('.tlt') or model_path.endswith('.pth'):
            # build dataloader
            dm = CNDataModule(experiment_config.dataset.segment)
            dm.setup(stage="predict")

            # build model and load from the given checkpoint
            model = ChangeNetPlSegment.load_from_checkpoint(model_path,
                                                            map_location="cpu",
                                                            experiment_spec=experiment_config)

        elif model_path.endswith('.engine'):
            raise NotImplementedError("TensorRT evaluation is supported through tao-deploy. "
                                      "Please use tao-deploy to generate TensorRT engine and run evaluation.")
        else:
            raise NotImplementedError("Model path format is only supported for .tlt or .pth")

    elif task == 'classify':
        dm = OIDataModule(experiment_config, changenet=True)
        model = ChangeNetPlClassifier.load_from_checkpoint(model_path,
                                                           map_location="cpu",
                                                           experiment_spec=experiment_config,
                                                           dm=dm
                                                           )
    else:
        raise NotImplementedError('Only tasks supported by Visual ChangeNet are: "segment" and "classify"')

    trainer = Trainer(**trainer_kwargs)

    trainer.predict(model, datamodule=dm)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="Visual ChangeNet", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
