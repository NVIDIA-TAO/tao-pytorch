# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset convert script for PointPillars."""
import os

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.config.pointpillars.default_config import ExperimentConfig
from nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.datasets.general.pc_dataset import create_pc_infos

from pathlib import Path


spec_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "cfgs")


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=spec_root, config_name="pointpillar_general", schema=ExperimentConfig
)
def convert_dataset(cfg: ExperimentConfig) -> None:
    """Convert dataset."""
    names = cfg.dataset.class_names
    data_path = cfg.dataset.data_path
    results_dir = cfg.results_dir
    save_dir = os.path.join(results_dir, "data_info")
    cfg.dataset.data_info_path = save_dir
    os.makedirs(save_dir, exist_ok=True)
    # Set status logging
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file, append=True))
    status_logging.get_status_logger().write(status_level=status_logging.Status.STARTED, message="Starting PointPillars dataset convert")
    try:
        create_pc_infos(
            dataset_cfg=cfg.dataset,
            class_names=names,
            data_path=Path(data_path),
            save_path=Path(save_dir),
            status_logging=status_logging
        )
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.RUNNING,
            message="Dataset convert finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Dataset convert was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e


if __name__ == "__main__":
    convert_dataset()
