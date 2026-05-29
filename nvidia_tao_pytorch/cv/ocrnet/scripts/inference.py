# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Inference OCRNet script.
"""
import logging
import os

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.config.ocrnet.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.ocrnet.dataloader.pl_ocr_data_module import OCRDataModule
from nvidia_tao_pytorch.cv.ocrnet.model.pl_ocrnet import OCRNetModel
from nvidia_tao_pytorch.cv.ocrnet.model.model import Model
from nvidia_tao_pytorch.cv.ocrnet.utils.utils import load_checkpoint

logger = logging.getLogger(__name__)


def run_experiment(experiment_spec, key):
    """run experiment."""
    if "train_gt_file" in experiment_spec["dataset"]:
        if experiment_spec["dataset"]["train_gt_file"] == "":
            experiment_spec["dataset"]["train_gt_file"] = None
    if "val_gt_file" in experiment_spec["dataset"]:
        if experiment_spec["dataset"]["val_gt_file"] == "":
            experiment_spec["dataset"]["val_gt_file"] = None

    model_path, trainer_kwargs = initialize_inference_experiment(experiment_spec, key)
    if len(trainer_kwargs['devices']) > 1:
        trainer_kwargs['devices'] = [trainer_kwargs['devices'][0]]
        logger.info(f"OCRNet does not support multi-GPU inference at this time. Using only GPU {trainer_kwargs['devices']}")

    dm = OCRDataModule(experiment_spec)
    dm.setup(stage='predict')

    # If pruned, will load the pruned model graph during construction
    model = OCRNetModel(experiment_spec, dm)
    # load model
    ckpt = load_checkpoint(model_path,
                           key=key,
                           to_cpu=True)

    if not isinstance(ckpt, Model):
        if "modelopt_state" in ckpt.keys():
            # Evaluate the quantized model
            logger.info(f"loading pretrained quantized model from {model_path}")
            import modelopt.torch.opt as mto
            from modelopt.torch.quantization import QuantModuleRegistry
            import torch.nn as nn
            QuantModuleRegistry.unregister(nn.LSTM)
            qat_model = mto.restore(model.model, model_path)
            model.model = qat_model
        else:
            # For loading public pretrained weights
            model.model.load_state_dict(ckpt.state_dict(), strict=True)
    else:
        logger.info('loading pretrained model from %s' % model_path)
        model.model.load_state_dict(ckpt.state_dict(), strict=True)

    trainer = Trainer(**trainer_kwargs)

    trainer.predict(model, datamodule=dm)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name="OCRNet", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_spec=cfg,
                   key=cfg.encryption_key)


if __name__ == '__main__':
    main()
