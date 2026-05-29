# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Inference of BEVFusion model. """

import os
from mmengine.config import Config

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging

# Triggers build of custom modules
from nvidia_tao_pytorch.config.bevfusion.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.bevfusion.utils.config import BEVFusionConfig
from nvidia_tao_pytorch.cv.bevfusion.inferencer import TAOMultiModalDet3DInferencer, prepare_inferencer_args
from nvidia_tao_pytorch.cv.bevfusion.visualization import TAO3DLocalVisualizer # noqa pylint: disable=W0611
from nvidia_tao_pytorch.cv.bevfusion.model import * # noqa pylint: disable=W0401, W0614
from nvidia_tao_pytorch.cv.bevfusion.datasets import * # noqa pylint: disable=W0401, W0614


def run_experiment(experiment_config):
    """Start the inference.
    Args:
        experiment_config (Dict): Config dictionary containing epxeriment parameters
        results_dir (str): Results dir to save the inference images

    """
    results_dir = experiment_config.results_dir
    status_logger = status_logging.get_status_logger()
    # log to file
    status_logger.write(message="**********************Start logging for Inference**********************.")

    bev_config = BEVFusionConfig(experiment_config, phase="inference")
    infer_cfg = bev_config.updated_config
    infer_cfg["visualizer"]["save_dir"] = results_dir
    infer_cfg = Config(infer_cfg)

    checkpoint = experiment_config.inference.checkpoint

    init_args, call_args = prepare_inferencer_args(infer_cfg, checkpoint, results_dir)

    inferencer = TAOMultiModalDet3DInferencer(**init_args)
    inferencer(**call_args)

    status_logger.write(message="Completed Inference.", status_level=status_logging.Status.RUNNING)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="test_isbi", schema=ExperimentConfig
)
@monitor_status(name="Bevfusion", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the Inference process."""
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
