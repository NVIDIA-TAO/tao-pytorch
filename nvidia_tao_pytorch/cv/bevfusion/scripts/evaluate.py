# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Evaluation of BEVFusion model. """
import os
import logging

from mmengine.config import Config
from mmengine.runner import Runner

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner

# Triggers build of custom modules
from nvidia_tao_pytorch.config.bevfusion.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.bevfusion.utils.config import BEVFusionConfig
from nvidia_tao_pytorch.cv.bevfusion.visualization import TAO3DLocalVisualizer  # noqa pylint: disable=W0611, W0401, F401
from nvidia_tao_pytorch.cv.bevfusion.evaluation import TAO3DMetric  # noqa pylint: disable=W0611,W0401, F401
from nvidia_tao_pytorch.cv.bevfusion.model import *  # noqa pylint: disable=W0401, W0614, W0611
from nvidia_tao_pytorch.cv.bevfusion.datasets import *  # noqa pylint: disable=W0401, W0614, W0611


def run_experiment(experiment_config):
    """Start the evaluate.
    Args:
        experiment_config (Dict): Config dictionary containing epxeriment parameters
        results_dir (str): Results dir to save the evaluation result

    """
    results_dir = experiment_config.results_dir
    # numba logging supression
    nb_logger = logging.getLogger('numba')
    nb_logger.setLevel(logging.ERROR)  # only show error

    status_logger = status_logging.get_status_logger()
    status_logger.write(status_level=status_logging.Status.STARTED,
                        message="********************** Start BEVFusion Evaluation **********************.")

    bev_config = BEVFusionConfig(experiment_config, phase="evaluate")
    eval_cfg = bev_config.updated_config
    eval_cfg["work_dir"] = results_dir
    # This is provided in notebook
    eval_cfg["load_from"] = experiment_config.evaluate.checkpoint

    eval_cfg = Config(eval_cfg)
    runner = Runner.from_cfg(eval_cfg)

    # start testing
    runner.test()
    status_logger.write(status_level=status_logging.Status.RUNNING,
                        message="********************** Completed BEVFusion Evaluation **********************.")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="export", schema=ExperimentConfig
)
@monitor_status(name="Bevfusion", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
