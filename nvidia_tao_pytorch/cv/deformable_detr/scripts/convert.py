# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert ODDataset to sharded json format."""
import os

from nvidia_tao_pytorch.config.deformable_detr.dataset import DDDatasetConvertConfig
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.cv.deformable_detr.utils.converter import KITTIConverter


def build_converter(experiment_config, input_source):
    """Build a DatasetConverter object.

    Build and return an object of desired subclass of DatasetConverter based on
    given dataset convert configuration.

    Args:
        experiment_config (DDDatasetConvertConfig): Dataset convert configuration object
        input_source (string).
    Return:
        converter (DatasetConverter): An object of desired subclass of DatasetConverter.
    """
    constructor_kwargs = {'data_root': experiment_config["data_root"],
                          'input_source': input_source,
                          'partition_mode': experiment_config["partition_mode"],
                          'num_partitions': experiment_config["num_partitions"],
                          'num_shards': experiment_config["num_shards"],
                          'output_dir': experiment_config["results_dir"],
                          'mapping_path': experiment_config["mapping_path"]}

    constructor_kwargs['image_dir_name'] = experiment_config["image_dir_name"]
    constructor_kwargs['label_dir_name'] = experiment_config["label_dir_name"]

    # Those two directories are by default empty string in proto
    # Here we do some check to make them default to None(in constructor)
    # Otherwise it will raise error if we pass the empty strings
    # directly to constructors.

    constructor_kwargs['extension'] = experiment_config["image_extension"] or '.png'
    constructor_kwargs['val_split'] = experiment_config["val_split"]

    converter = KITTIConverter(**constructor_kwargs)

    return converter


def run_experiment(experiment_config,
                   results_dir):
    """Start the Data Converter."""
    input_sources = experiment_config["input_source"]
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
        message="Starting DDETR/DINO dataset convert"
    )

    with open(input_sources, 'r') as f:
        seq_txt = f.readlines()
        for input_source in seq_txt:
            input_source = input_source.rstrip('\n')
            converter = build_converter(experiment_config, input_source)
            converter.convert()


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="convert", schema=DDDatasetConvertConfig
)
def main(cfg: DDDatasetConvertConfig) -> None:
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
