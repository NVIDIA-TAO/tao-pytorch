# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export Sparse4D model to ONNX."""

import os
import torch

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.config.sparse4d.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.sparse4d.model.sparse4d_pl_model import Sparse4DPlModel
from nvidia_tao_pytorch.cv.sparse4d.utils.onnx_export import Sparse4DExporter
from nvidia_tao_pytorch.cv.sparse4d.utils.misc import load_pretrained_weights

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="Sparse4D", mode="export")
def main(cfg: ExperimentConfig) -> None:
    """CLI wrapper to run export.
    This function parses the command line interface for tlt-export, instantiates the respective
    exporter and serializes the trained model to an etlt file. The tools also runs optimization
    to the int8 backend.

    Args:
        cl_args(list): Arguments to parse.

    Returns:
        No explicit returns.
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run_export(cfg)


def run_export(experiment_config):
    """Wrapper to run export of tlt models.

    Args:
        args (dict): Dictionary of parsed arguments to run export.

    Returns:
        No explicit returns.
    """
    gpu_id = experiment_config.export.gpu_id
    torch.cuda.set_device(gpu_id)

    # Parsing command line arguments.
    model_path = experiment_config.export.checkpoint
    key = experiment_config.encryption_key

    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    output_file = experiment_config.export.onnx_file
    on_cpu = experiment_config.export.on_cpu

    # Set default output filename if the filename
    # isn't provided over the command line.
    if output_file is None:
        split_name = os.path.splitext(model_path)[0]
        output_file = "{}.onnx".format(split_name)

    # Warn the user if an exported file already exists.
    assert not os.path.exists(output_file), "Default onnx file {} already "\
        "exists".format(output_file)

    # Make an output directory if necessary.
    output_root = os.path.dirname(os.path.realpath(output_file))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Instantiate the model first
    model = Sparse4DPlModel(experiment_config)

    pretrained_path = experiment_config.export.checkpoint

    if pretrained_path:
        logging.info(f"Loading checkpoint from: {pretrained_path}")
        new_state_dict = load_pretrained_weights(pretrained_path)
        model.load_state_dict(new_state_dict, strict=False)
        logging.info(f"Successfully loaded weights into Sparse4DPlModel from {pretrained_path}")

    model.eval()
    if not on_cpu:
        model.cuda()

    sparse4d_exporter = Sparse4DExporter(model)
    sparse4d_exporter.export_model(
        experiment_config,
        model,
        output_file
    )

    sparse4d_exporter.check_onnx(output_file)

    logging.info(f"ONNX file stored at {output_file}")


if __name__ == "__main__":
    main()
