# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export Mask2former model to ONNX."""

import os
import torch

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner

from nvidia_tao_pytorch.config.mask2former.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.mask2former.model.pl_model import Mask2formerPlModule
from nvidia_tao_pytorch.cv.mask2former.export.onnx_exporter import ONNXExporter


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="spec", schema=ExperimentConfig
)
@monitor_status(name="Mask2Former", mode="export")
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
    experiment_config.model.export = True
    model_path = experiment_config.export.checkpoint
    output_file = experiment_config.export.onnx_file
    input_channel = experiment_config.export.input_channel
    input_width = experiment_config.export.input_width
    input_height = experiment_config.export.input_height
    opset_version = experiment_config.export.opset_version
    batch_size = experiment_config.export.batch_size
    on_cpu = experiment_config.export.on_cpu
    mode = experiment_config.model.mode
    if batch_size is None or batch_size == -1:
        input_batch_size = 1
    else:
        input_batch_size = batch_size

    # Set default output filename if the filename
    # isn't provided over the command line.
    if not output_file:
        split_name = os.path.splitext(model_path)[0]
        output_file = f"{split_name}.onnx"

    # Warn the user if an exported file already exists.
    assert not os.path.exists(output_file), f"Default onnx file {output_file} already exists"

    # Make an output directory if necessary.
    output_root = os.path.dirname(os.path.realpath(output_file))
    os.makedirs(output_root, exist_ok=True)

    # load model
    pl_model = Mask2formerPlModule.load_from_checkpoint(
        model_path,
        map_location='cpu' if on_cpu else 'cuda',
        cfg=experiment_config,
    )

    model = pl_model.model
    model.eval()
    if not on_cpu:
        model.cuda()

    input_names = ['inputs']
    if mode == 'instance':
        output_names = ["pred_masks", "pred_scores", "pred_classes"]
    elif mode == 'panoptic':
        output_names = ["prob_masks", "pred_masks", "pred_scores", "pred_classes"]
    elif mode == 'semantic':
        output_names = ["pred_masks"]
    else:
        raise ValueError("Only instance, panoptic and semantic modes are supported.")
    # create dummy input
    if on_cpu:
        dummy_input = torch.ones(input_batch_size, input_channel, input_height, input_width, device='cpu')
    else:
        dummy_input = torch.ones(input_batch_size, input_channel, input_height, input_width, device='cuda')

    exporter = ONNXExporter()
    exporter.export_model(
        model, batch_size,
        output_file,
        dummy_input,
        input_names=input_names,
        opset_version=opset_version,
        output_names=output_names,
        do_constant_folding=False,
        verbose=experiment_config.export.verbose)
    exporter.check_onnx(output_file)
    exporter.onnx_change(output_file)

    print(f"ONNX file stored at {output_file}")


if __name__ == "__main__":
    main()
