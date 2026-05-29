# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export NVPanoptix3D model to ONNX."""

import os
import torch
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.config.nvpanoptix3d.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.nvpanoptix3d.export.onnx_exporter import export_2d_model


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="spec_front3d_2d", schema=ExperimentConfig
)
@monitor_status(name="NVPanoptix3D", mode="export")
def main(cfg: ExperimentConfig) -> None:
    """CLI wrapper to run export.

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
    output_file_2d = experiment_config.export.onnx_file_2d
    input_width = experiment_config.export.input_width
    input_height = experiment_config.export.input_height
    opset_version = experiment_config.export.opset_version
    batch_size = experiment_config.export.batch_size
    on_cpu = experiment_config.export.on_cpu
    if batch_size is None or batch_size == -1:
        input_batch_size = 1
    else:
        input_batch_size = batch_size

    # Export 2D model:
    export_2d_model(
        cfg=experiment_config,
        output_path=output_file_2d,
        batch_size=input_batch_size,
        input_height=input_height,
        input_width=input_width,
        device="cpu" if on_cpu else "cuda",
        opset_version=opset_version,
        verbose=experiment_config.export.verbose,
        checkpoint_path=model_path,
    )


if __name__ == "__main__":
    main()
