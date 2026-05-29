#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a trained ocdnet model."""

import os
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.ocdnet.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment
from nvidia_tao_pytorch.cv.ocdnet.data_loader.pl_ocd_data_module import OCDDataModule
from nvidia_tao_pytorch.cv.ocdnet.utils.util import load_checkpoint
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs

from nvidia_tao_pytorch.cv.ocdnet.model.pl_ocd_model import OCDnetModel

import pycuda
import pycuda.autoinit
pyc_dev = pycuda.autoinit.device
pyc_ctx = pyc_dev.retain_primary_context()


def run_experiment(experiment_config):
    """Run experiment."""
    experiment_config = OmegaConf.to_container(experiment_config)
    model_path, trainer_kwargs = initialize_evaluation_experiment(experiment_config)

    experiment_config['model']['pretrained'] = False
    experiment_config["dataset"]["train_dataset"] = experiment_config["dataset"]["validate_dataset"]
    if model_path.split(".")[-1] in ["trt", "engine"]:
        raise Exception("Please use tao_deploy to run evaluation against tensorrt engine.")
    else:
        raw_checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        dm = OCDDataModule(experiment_config)
        dm.setup(stage='test')

        if not isinstance(raw_checkpoint, dict):
            model = raw_checkpoint
        else:
            checkpoint = load_checkpoint(model_path, to_cpu=True)
            model = OCDnetModel(experiment_config, dm, 'test')
            layers = checkpoint.keys()
            ckpt = dict()
            # Support loading official pretrained weights for eval
            for layer in layers:
                new_layer = layer
                if new_layer.startswith("model.module."):
                    new_layer = new_layer[13:]
                if new_layer == "decoder.in5.weight":
                    new_layer = "neck.in5.weight"
                elif new_layer == "decoder.in4.weight":
                    new_layer = "neck.in4.weight"
                elif new_layer == "decoder.in3.weight":
                    new_layer = "neck.in3.weight"
                elif new_layer == "decoder.in2.weight":
                    new_layer = "neck.in2.weight"
                elif new_layer == "decoder.out5.0.weight":
                    new_layer = "neck.out5.0.weight"
                elif new_layer == "decoder.out4.0.weight":
                    new_layer = "neck.out4.0.weight"
                elif new_layer == "decoder.out3.0.weight":
                    new_layer = "neck.out3.0.weight"
                elif new_layer == "decoder.out2.weight":
                    new_layer = "neck.out2.weight"
                elif new_layer == "decoder.binarize.0.weight":
                    new_layer = "head.binarize.0.weight"
                elif new_layer == "decoder.binarize.1.weight":
                    new_layer = "head.binarize.1.weight"
                elif new_layer == "decoder.binarize.1.bias":
                    new_layer = "head.binarize.1.bias"
                elif new_layer == "decoder.binarize.1.running_mean":
                    new_layer = "head.binarize.1.running_mean"
                elif new_layer == "decoder.binarize.1.running_var":
                    new_layer = "head.binarize.1.running_var"
                elif new_layer == "decoder.binarize.3.weight":
                    new_layer = "head.binarize.3.weight"
                elif new_layer == "decoder.binarize.3.bias":
                    new_layer = "head.binarize.3.bias"
                elif new_layer == "decoder.binarize.4.weight":
                    new_layer = "head.binarize.4.weight"
                elif new_layer == "decoder.binarize.4.bias":
                    new_layer = "head.binarize.4.bias"
                elif new_layer == "decoder.binarize.4.running_mean":
                    new_layer = "head.binarize.4.running_mean"
                elif new_layer == "decoder.binarize.4.running_var":
                    new_layer = "head.binarize.4.running_var"
                elif new_layer == "decoder.binarize.6.weight":
                    new_layer = "head.binarize.6.weight"
                elif new_layer == "decoder.binarize.6.bias":
                    new_layer = "head.binarize.6.bias"
                elif new_layer == "decoder.thresh.0.weight":
                    new_layer = "head.thresh.0.weight"
                elif new_layer == "decoder.thresh.1.weight":
                    new_layer = "head.thresh.1.weight"
                elif new_layer == "decoder.thresh.1.bias":
                    new_layer = "head.thresh.1.bias"
                elif new_layer == "decoder.thresh.1.running_mean":
                    new_layer = "head.thresh.1.running_mean"
                elif new_layer == "decoder.thresh.1.running_var":
                    new_layer = "head.thresh.1.running_var"
                elif new_layer == "decoder.thresh.3.weight":
                    new_layer = "head.thresh.3.weight"
                elif new_layer == "decoder.thresh.3.bias":
                    new_layer = "head.thresh.3.bias"
                elif new_layer == "decoder.thresh.4.weight":
                    new_layer = "head.thresh.4.weight"
                elif new_layer == "decoder.thresh.4.bias":
                    new_layer = "head.thresh.4.bias"
                elif new_layer == "decoder.thresh.4.running_mean":
                    new_layer = "head.thresh.4.running_mean"
                elif new_layer == "decoder.thresh.4.running_var":
                    new_layer = "head.thresh.4.running_var"
                elif new_layer == "decoder.thresh.6.weight":
                    new_layer = "head.thresh.6.weight"
                elif new_layer == "decoder.thresh.6.bias":
                    new_layer = "head.thresh.6.bias"
                elif "num_batches_tracked" in new_layer:
                    continue
                elif "backbone.fc" in new_layer:
                    continue
                elif "backbone.smooth" in new_layer:
                    continue
                ckpt[new_layer] = checkpoint[layer]
            model.model.load_state_dict(ckpt)

    trainer = Trainer(**trainer_kwargs)

    trainer.test(model, datamodule=dm)

    pyc_ctx.pop()


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="evaluate", schema=ExperimentConfig
)
@monitor_status(name="OCDNet", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """Run the evaluation process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)
    pyc_ctx.push()

    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
