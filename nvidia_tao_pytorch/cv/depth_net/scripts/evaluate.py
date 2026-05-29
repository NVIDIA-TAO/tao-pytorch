# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a trained depthnet model."""

import os
import torch
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.depth_net.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment
from nvidia_tao_pytorch.cv.depth_net.dataloader import build_pl_data_module
from nvidia_tao_pytorch.cv.depth_net.utils.misc import parse_mono_depth_checkpoint
from nvidia_tao_pytorch.cv.depth_net.model.build_pl_model import build_pl_model, get_pl_module


def run_experiment(experiment_config, key):
    """Run experiment."""
    model_path, trainer_kwargs = initialize_evaluation_experiment(experiment_config, key)

    if model_path.endswith('.tlt') or model_path.endswith('.pth'):
        # build data module
        dm = build_pl_data_module(experiment_config.dataset)
        dm.setup(stage="test")

        # FFS commercial ckpt is a research-pickled nn.Module (not a PL ckpt
        # nor a plain state_dict). Route it through load_ffs_pretrained
        # which handles the pickle stub + prefix/substring remap and
        # reports missing/unexpected explicitly. Mirrors scripts/inference.py.
        if experiment_config.model.model_type == 'FastFoundationStereo':
            from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.fast_foundation_stereo.ckpt_utils import (
                load_ffs_pretrained,
            )
            model = build_pl_model(experiment_config)
            result = load_ffs_pretrained(model.model, model_path)
            assert not result['missing'], (
                f"FFS ckpt missing keys: {result['missing']}")
            assert not result['unexpected'], (
                f"FFS ckpt unexpected keys: {result['unexpected']}")
        else:
            model_dict = torch.load(model_path, map_location="cpu")
            if "pytorch-lightning_version" not in model_dict:
                # parse public checkpoint
                if experiment_config.model.model_type in ['MetricDepthAnything', 'RelativeDepthAnything']:
                    model_dict = parse_mono_depth_checkpoint(model_dict, experiment_config.model.model_type)
                model = build_pl_model(experiment_config)
                model.load_state_dict(model_dict, strict=True)
            else:
                model = get_pl_module(experiment_config).load_from_checkpoint(
                    model_path,
                    map_location="cpu",
                    experiment_spec=experiment_config
                )
        trainer = Trainer(**trainer_kwargs)
        trainer.test(model, datamodule=dm)

    elif model_path.endswith('.engine'):
        raise NotImplementedError("TensorRT evaluation is supported through tao-deploy. "
                                  "Please use tao-deploy to generate TensorRT engine and run evaluation.")
    else:
        raise NotImplementedError("Model path format is only supported for .tlt or .pth")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="evaluate", schema=ExperimentConfig
)
@monitor_status(name="Depth Net", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """Run the evaluate process."""
    run_experiment(experiment_config=cfg,
                   key=cfg.encryption_key)


if __name__ == "__main__":
    main()
