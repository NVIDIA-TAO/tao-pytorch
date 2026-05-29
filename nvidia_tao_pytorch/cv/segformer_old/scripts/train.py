# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train Segformer model."""

import os

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.config.segformer.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.segformer_old.utils.config import MMSegmentationConfig

# Triggers build of custom modules
from nvidia_tao_pytorch.cv.segformer_old.model import * # noqa pylint: disable=W0401, W0614
from nvidia_tao_pytorch.cv.segformer_old.dataloader import * # noqa pylint: disable=W0401, W0614

from mmengine.runner import Runner
from mmengine.config import Config


def get_latest_pth_model(results_dir):
    """Utility function to return the latest pth model in a dir.
    Args:
        results_dir (str): Results dir to save the checkpoints.

    """
    trainable_ckpts = [int(item.split('.')[0].split('_')[1]) for item in os.listdir(results_dir)
                       if item.endswith(".pth")]
    num_ckpts = len(trainable_ckpts)
    if num_ckpts == 0:
        return None
    latest_step = sorted(trainable_ckpts, reverse=True)[0]
    latest_checkpoint = os.path.join(results_dir, f"iter_{latest_step}.pth")
    if not os.path.isfile(latest_checkpoint):
        raise FileNotFoundError("Checkpoint file not found at {}")
    return latest_checkpoint


def run_experiment(experiment_config):
    """Start the training.
    Args:
        experiment_config (Dict): Config dictionary containing epxeriment parameters
        results_dir (str): Results dir to save the trained checkpoints.

    """
    if experiment_config.get("train", {}).get("resume_training_checkpoint_path", None) == "":
        experiment_config["train"]["resume_training_checkpoint_path"] = None
    results_dir = experiment_config.results_dir
    status_logger = status_logging.get_status_logger()
    status_logger.write(status_level=status_logging.Status.STARTED,
                        message="********************** Start Segformer Training **********************.")

    mmseg_config = MMSegmentationConfig(experiment_config, phase="train")
    train_cfg = mmseg_config.updated_config

    train_cfg["work_dir"] = results_dir
    resume_checkpoint = get_latest_pth_model(results_dir) if not train_cfg["load_from"] else train_cfg["load_from"]
    if resume_checkpoint:
        train_cfg["load_from"] = resume_checkpoint
        train_cfg["resume"] = True
        train_cfg["model"]["backbone"]["init_cfg"] = None  # Disable pretrained weights if there are any

    # Converts dict to cfg
    # (This is necessary due to a bug in mmseg for model.test_cfg which errors if it's a dict)
    train_cfg = Config(train_cfg)

    runner = Runner.from_cfg(train_cfg)
    runner.train()
    status_logger.write(status_level=status_logging.Status.RUNNING,
                        message="********************** Completed Segformer Training **********************.")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="train_isbi", schema=ExperimentConfig
)
@monitor_status(name="Segformer", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
