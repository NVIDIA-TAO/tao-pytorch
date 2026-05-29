# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export metric-learning recognition model to ONNX."""

import os
import tempfile
import torch
from nvidia_tao_pytorch.cv.ml_recog.dataloader.pl_ml_data_module import MLDataModule
from onnxsim import simplify
import onnx

from nvidia_tao_pytorch.config.ml_recog.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.ml_recog.model.pl_ml_recog_model import MLRecogModel
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_export(args):
    """Wrapper to run export of .pth checkpoints.

    Args:
        args (DictConfig): Configuration dictionary
    """
    experiment_config = args
    results_dir = experiment_config.results_dir

    if experiment_config['export']["on_cpu"]:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
        gpu_id = experiment_config.export.gpu_id
        torch.cuda.set_device(gpu_id)
    else:
        error_msg = "No GPU available for export."
        raise ValueError(error_msg)

    checkpoint = experiment_config["export"]["checkpoint"]
    dm = MLDataModule(experiment_config)
    if checkpoint is not None:
        status_logging.get_status_logger().write(
            message=f"Loading checkpoint: {experiment_config['export']['checkpoint']}",
            status_level=status_logging.Status.STARTED)
        pl_model = MLRecogModel.load_from_checkpoint(experiment_config["export"]["checkpoint"],
                                                     map_location="cpu",
                                                     experiment_spec=experiment_config,
                                                     dm=dm,
                                                     subtask="export")
        # Set default output filename if the filename
        # isn't provided over the command line.
        if experiment_config['export']['onnx_file'] is None:
            split_name = os.path.splitext(os.path.basename(checkpoint))[0]
            output_file = os.path.join(results_dir, f"{split_name}.onnx")
        else:
            output_file = experiment_config['export']['onnx_file']

    else:
        pl_model = MLRecogModel(experiment_config,
                                dm,
                                subtask="export")
        if experiment_config['export']['onnx_file'] is None:
            output_file = os.path.join(results_dir, "ml_recog.onnx")
        else:
            output_file = experiment_config['export']['onnx_file']

    assert not os.path.exists(output_file), f"output file {output_file} "\
        "already exists."

    # Make an output directory if necessary.
    output_root = os.path.dirname(os.path.realpath(output_file))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    model = pl_model.model
    model.eval()
    model.to(device)

    input_names = ["input"]
    output_names = ["fc_pred"]

    # create dummy input
    input_channels = experiment_config["model"]["input_channels"]
    input_width = experiment_config["model"]["input_width"]
    input_height = experiment_config["model"]["input_height"]
    batch_size = experiment_config["export"]["batch_size"]
    if batch_size == -1:
        dynamic_axes = {"input": {0: "batch"}, "fc_pred": {0: "batch"}}
        dummy_input = torch.randn(
            1,  input_channels, input_width, input_height).to(device)
    elif batch_size >= 1:
        dynamic_axes = None
        dummy_input = torch.randn(
            batch_size,  input_channels, input_width, input_height).to(device)
    else:
        raise ValueError("`export.batch_size` must be greater than 0 or -1.")

    # export
    status_logging.get_status_logger().write(
        message="Exporting model to ONNX",
        status_level=status_logging.Status.STARTED)
    os_handle, tmp_onnx_file = tempfile.mkstemp(suffix=".onnx")
    os.close(os_handle)
    torch.onnx.export(model,
                      dummy_input,
                      tmp_onnx_file,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=experiment_config["export"]["opset_version"],
                      dynamic_axes=dynamic_axes,
                      verbose=experiment_config["export"]["verbose"],
                      dynamo=False)

    # add simplification
    status_logging.get_status_logger().write(
        message="Simplifying ONNX model",
        status_level=status_logging.Status.STARTED)
    simplified_model, _ = simplify(
        tmp_onnx_file,
        test_input_shapes={'input': (1, input_channels, input_width, input_height)},
        check_n=3)
    onnx.save(simplified_model, output_file)
    status_logging.get_status_logger().write(
        message=f"ONNX model saved at {output_file}",
        status_level=status_logging.Status.RUNNING)


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="export", schema=ExperimentConfig
)
@monitor_status(name="Metric Learning Recognition", mode="export")
def main(cfg: ExperimentConfig) -> None:
    """CLI wrapper to run export.

    This function parses the command line interface for tlt-export, instantiates the respective
    exporter and serializes the trained model to an onnx file. The tools also runs optimization
    to the int8 backend.

    Args:
        cl_args(list): Arguments to parse.

    Returns:
        No explicit returns.
    """
    obfuscate_logs(cfg)

    run_export(cfg)


if __name__ == "__main__":
    main()
