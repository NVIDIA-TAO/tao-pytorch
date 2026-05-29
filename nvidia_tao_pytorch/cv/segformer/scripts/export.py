# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export Segformer model to ONNX."""

import os
import torch

from nvidia_tao_pytorch.config.segformer.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.utilities import encrypt_onnx, get_nvdsinfer_yaml, write_classes_file
from nvidia_tao_pytorch.cv.segformer.dataloader.utils import build_target_class_list
from nvidia_tao_pytorch.cv.segformer.model.segformer_pl_model import SegFormerPlModel
from nvidia_tao_pytorch.cv.segformer.types.segformer_nvdsinfer import SFNvDSInferConfig
from nvidia_tao_pytorch.cv.segformer.utils.onnx_export import ONNXExporter

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_class_labels(experiment_config, output_root):
    """Get the labels file from the experiment spec.

    Args:
        experiment_config (ExperimentConfig): Config element for the experiment.
        output_root (str): Path to the results directory.

    Returns:
        tuple: (str, int) Path to labels file and number of classes.
    """
    target_classes = [
        target_class.name for target_class
        in build_target_class_list(dataset_config=experiment_config.dataset.segment)
    ]
    ds_labels_file = os.path.join(output_root, "labels.txt")
    num_classes = write_classes_file(ds_labels_file, target_classes)
    return ds_labels_file, num_classes


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="Segformer", mode="export")
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
        experiment_config (ExperimentConfig): Configuration for the experiment.

    Returns:
        None
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
    input_shape = [input_channel, input_height, input_width]
    serialize_nvdsinfer = experiment_config.export.serialize_nvdsinfer
    opset_version = experiment_config.export.opset_version
    batch_size = experiment_config.export.batch_size
    on_cpu = experiment_config.export.on_cpu
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

    # Setting input/output tensor names.
    input_names = ['input']
    output_names = ['output']

    if serialize_nvdsinfer:
        ds_labels_file, num_classes = get_class_labels(
            experiment_config=experiment_config,
            output_root=output_root
        )
        offsets = [
            value * 255.0 for value in experiment_config.dataset.segment.augmentation.mean
        ]
        config_str = get_nvdsinfer_yaml(
            SFNvDSInferConfig,
            ds_labels_file,
            num_classes=num_classes,
            output_file=output_file,
            input_shape=input_shape,
            output_names=output_names,
            offsets=offsets
        )
        nvdsinfer_yaml_file = os.path.join(
            output_root, "nvdsinfer_config.yaml"
        )
        logging.info("Serializing the deepstream config to %s", nvdsinfer_yaml_file)
        with open(nvdsinfer_yaml_file, "w", encoding="utf-8") as nvds_file:
            nvds_file.write(str(config_str))

    # load model
    sf_model = SegFormerPlModel.load_from_checkpoint(
        model_path,
        map_location="cpu" if on_cpu else 'cuda',
        experiment_spec=experiment_config,
        export=True
    )

    model = sf_model.model
    model.eval()
    if not on_cpu:
        model.cuda()

    # create dummy input
    if on_cpu:
        dummy_input = torch.ones(input_batch_size, *input_shape, device='cpu')
    else:
        dummy_input = torch.ones(input_batch_size, *input_shape, device='cuda')

    if output_file.endswith('.etlt'):
        tmp_onnx_file = output_file.replace('.etlt', '.onnx')
    else:
        tmp_onnx_file = output_file

    try:
        onnx_export = ONNXExporter()
        onnx_export.export_model(
            model, batch_size,
            tmp_onnx_file,
            dummy_input,
            input_names=input_names,
            opset_version=opset_version,
            output_names=output_names,
            do_constant_folding=True,
            verbose=experiment_config.export.verbose,
        )

        onnx_export.check_onnx(output_file)

    except ValueError as e:
        raise ValueError(
            f"Onnx export export and check failed due to {str(e)}") from e

    if output_file.endswith('.etlt') and key:
        # encrypt the onnx if and only if key is provided and output file name ends with .etlt
        encrypt_onnx(
            tmp_file_name=tmp_onnx_file,
            output_file_name=output_file,
            key=key
        )

        os.remove(tmp_onnx_file)
    logging.info("ONNX file stored at %s", output_file)


if __name__ == "__main__":
    main()
