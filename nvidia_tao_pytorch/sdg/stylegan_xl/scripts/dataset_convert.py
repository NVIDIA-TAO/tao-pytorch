# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert StyleGAN-XL Dataset to required input format."""

import os

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
# Triggers build of custom modules
from nvidia_tao_pytorch.config.stylegan_xl.dataset import DataConvertExpConfig
from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.sdg.stylegan_xl.utils.dataset_tool import convert_dataset


def run_experiment(experiment_config,
                   results_dir):
    """Start the Data Converter."""
    check_and_create(results_dir)

    # Set status logging
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            append=True
        )
    )
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.STARTED,
        message="Starting StyleGAN-XL dataset convert"
    )
    source = experiment_config['source']
    dest = os.path.join(results_dir, experiment_config['dest_file_name'])
    resolution = (experiment_config['resolution'][0], experiment_config['resolution'][1])
    transform = experiment_config['transform']

    convert_dataset(
        source=source,
        dest=dest,
        max_images=None,
        transform=transform,
        resolution=resolution
    )


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="convert", schema=DataConvertExpConfig
)
def main(cfg: DataConvertExpConfig) -> None:
    """Run the convert dataset process."""
    try:
        run_experiment(experiment_config=cfg,
                       results_dir=cfg.results_dir)
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.RUNNING,
            message="Dataset convert finished successfully"
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
    main()
