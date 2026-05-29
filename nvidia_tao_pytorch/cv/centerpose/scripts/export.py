# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export CenterPose model to ONNX."""

import os
import torch

from nvidia_tao_pytorch.config.centerpose.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.utilities import encrypt_onnx
from nvidia_tao_pytorch.cv.centerpose.utils.onnx_export import ONNXExporter
from nvidia_tao_pytorch.cv.centerpose.model.pl_centerpose_model import CenterPosePlModel
from nvidia_tao_pytorch.cv.centerpose.model.post_processing import HeatmapDecoder
from nvidia_tao_pytorch.cv.centerpose.model.centerpose import CenterPoseWrapped

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="export", schema=ExperimentConfig
)
@monitor_status(name="Centerpose", mode="export")
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
    input_channel = experiment_config.export.input_channel
    input_width = experiment_config.export.input_width
    input_height = experiment_config.export.input_height
    opset_version = experiment_config.export.opset_version
    batch_size = experiment_config.export.batch_size
    num_select = experiment_config.export.num_select
    on_cpu = experiment_config.export.on_cpu
    do_constant_folding = experiment_config.export.do_constant_folding
    if batch_size is None or batch_size == -1:
        input_batch_size = 1
    else:
        input_batch_size = batch_size

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

    # Load model
    pl_model = CenterPosePlModel.load_from_checkpoint(model_path,
                                                      map_location='cpu' if on_cpu else 'cuda',
                                                      experiment_spec=experiment_config)
    model = pl_model.model
    model.eval()
    if not on_cpu:
        model.cuda()

    # Wrapped the heatmap decoder into the ONNX model to speed up the inference.
    hm_decoder = HeatmapDecoder(num_select)
    wrapped_model = CenterPoseWrapped(model, hm_decoder)
    wrapped_model.eval()
    if not on_cpu:
        wrapped_model.cuda()

    input_names = ['input']
    output_names = ['bboxes', 'scores', 'kps', 'clses', 'obj_scale', 'kps_displacement_mean', 'kps_heatmap_mean']

    # create dummy input
    if on_cpu:
        dummy_input = torch.ones(input_batch_size, input_channel, input_height, input_width, device='cpu')
    else:
        dummy_input = torch.ones(input_batch_size, input_channel, input_height, input_width, device='cuda')

    if output_file.endswith('.etlt'):
        tmp_onnx_file = output_file.replace('.etlt', '.onnx')
    else:
        tmp_onnx_file = output_file

    onnx_export = ONNXExporter()
    onnx_export.export_model(wrapped_model, batch_size,
                             tmp_onnx_file,
                             dummy_input,
                             input_names=input_names,
                             opset_version=opset_version,
                             output_names=output_names,
                             do_constant_folding=do_constant_folding,
                             verbose=experiment_config.export.verbose)
    onnx_export.check_onnx(tmp_onnx_file)

    if output_file.endswith('.etlt') and key:
        # encrypt the onnx if and only if key is provided and output file name ends with .etlt
        encrypt_onnx(tmp_file_name=tmp_onnx_file,
                     output_file_name=output_file,
                     key=key)

        os.remove(tmp_onnx_file)
    print(f"ONNX file stored at {output_file}")


if __name__ == "__main__":
    main()
