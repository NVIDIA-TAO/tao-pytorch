# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert BEVFusion Dataset to required input format."""

import os

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner

# Triggers build of custom modules
from nvidia_tao_pytorch.config.bevfusion.default_config import BEVFusionDataConvertExpConfig
from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.cv.bevfusion.datasets import kitti_data_prep, tao3d_data_prep
from nvidia_tao_pytorch.cv.bevfusion.utils import sanity_check


def run_experiment(experiment_config,
                   results_dir):
    """Start the Data Converter."""
    check_and_create(results_dir)
    sanity_check(experiment_config)

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
        message="Starting BEVFusion dataset convert"
    )
    dataset = experiment_config['dataset']
    output_prefix = experiment_config['output_prefix']
    if dataset.lower() == 'kitti':
        if output_prefix is None:
            output_prefix = 'kitti_person'
        kitti_data_prep(
            root_path=experiment_config['root_dir'],
            mode=experiment_config['mode'],
            info_prefix=output_prefix,
            out_dir=results_dir,
            with_plane=experiment_config['with_plane'])
    elif dataset.lower() == 'tao3d':
        if experiment_config['sequence_list'] is None or experiment_config['per_sequence'] is False:
            raise ValueError('you must specify both per_sequence and sequence_list in the config file. Currently per_sequence is {} and sequence_list is {}'.format(experiment_config['per_sequence'], experiment_config['sequence_list']))

        if output_prefix is None:
            output_prefix = 'tao3d'

        tao3d_data_prep(root_path=experiment_config['root_dir'],
                        mode=experiment_config['mode'],
                        seq_list=experiment_config['sequence_list'],
                        is_synthetic=experiment_config['is_synthetic'],
                        dimension_order=experiment_config['dimension_order'],
                        info_prefix=output_prefix,
                        out_dir=results_dir,
                        merge_only=experiment_config['merge_only'])
    else:
        raise NotImplementedError(f'Don\'t support {dataset} dataset.')


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="convert", schema=BEVFusionDataConvertExpConfig
)
def main(cfg: BEVFusionDataConvertExpConfig) -> None:
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
