# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export Visual ChangeNet model to ONNX."""

import os
import torch

from nvidia_tao_pytorch.config.visual_changenet.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.utilities import encrypt_onnx, write_classes_file, get_nvdsinfer_yaml
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.pl_oi_data_module import OIDataModule
from nvidia_tao_pytorch.cv.visual_changenet.types.visual_changenet_nvdsinfer import VisualChangeNetNvDSInferConfig
from nvidia_tao_pytorch.cv.visual_changenet.types.visual_changenet_preprocess import VisualChangeNetPreprocessConfig
from nvidia_tao_pytorch.cv.visual_changenet.utils.onnx_export import ONNXExporter
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models.cn_pl_model import ChangeNetPlModel as ChangeNetPlSegment
from nvidia_tao_pytorch.cv.visual_changenet.classification.models.cn_pl_model import ChangeNetPlModel as ChangeNetPlClassifier

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="Visual ChangeNet", mode="export")
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


def get_model_classes(experiment_config):
    """Get the number of classes and class names from the dataset configuration.

    Args:
        experiment_config: Experiment configuration containing dataset information

    Returns:
        tuple: (num_classes, class_names)
            - num_classes (int): Number of classes the model was trained on
            - class_names (list): List of class names in order of their IDs
    """
    if experiment_config.task == 'segment':
        num_classes = experiment_config.dataset.segment.num_classes
    elif experiment_config.task == 'classify':
        num_classes = experiment_config.dataset.classify.num_classes
    # TODO: add class names with class names if possible.
    class_names = [str(i) for i in range(num_classes)]
    return num_classes, class_names


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

    task = experiment_config.task
    output_file = experiment_config.export.onnx_file
    input_channel = experiment_config.export.input_channel
    input_width = experiment_config.export.input_width
    input_height = experiment_config.export.input_height
    input_shape = [input_channel, input_height, input_width]
    opset_version = experiment_config.export.opset_version
    batch_size = experiment_config.export.batch_size
    serialize_nvdsinfer = experiment_config.export.serialize_nvdsinfer
    on_cpu = experiment_config.export.on_cpu
    num_golden = experiment_config.dataset.classify.num_golden
    assert num_golden >= 1, "Number of golden samples must be greater than or equal to 1"

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

    assert task in ['segment', 'classify'], "Visual ChangeNet only supports 'segment' and 'classify' tasks."
    if task == 'classify':
        dm = OIDataModule(experiment_config)
        # load model
        cf_model = ChangeNetPlClassifier.load_from_checkpoint(model_path,
                                                              map_location="cpu" if on_cpu else 'cuda',
                                                              experiment_spec=experiment_config,
                                                              dm=dm,
                                                              export=True)

        output_names = ["output"]
        input_names = ["input_1", "input_2"]

    elif task == 'segment':
        # load model
        cf_model = ChangeNetPlSegment.load_from_checkpoint(model_path,
                                                           map_location="cpu" if on_cpu else 'cuda',
                                                           experiment_spec=experiment_config,
                                                           export=True)

        input_names = ['input0', 'input1']
        output_names = ["output0", "output1", 'output2', 'output3', 'output_final']
    else:
        raise NotImplementedError('Only tasks supported by Visual ChangeNet are: "segment" and "classify"')

    if serialize_nvdsinfer:
        assert task == 'classify', "Only classification task is supported for Inference Microservices."
        assert num_golden == 1, "Only one golden sample is supported for Inference Microservices."
        num_classes, class_names = get_model_classes(experiment_config)

        logging.info(f"Model was trained on {num_classes} classes: {class_names}")

        # Write class names to labels file
        labels_file = os.path.join(output_root, "labels.txt")
        write_classes_file(labels_file, class_names)

        nvdsinfer_yaml_file = os.path.join(
            output_root, "nvdsinfer_config.yaml"
        )
        logging.info("Serializing the deepstream config to {}".format(
            nvdsinfer_yaml_file
        ))
        # Write deepstream config to file
        nvdsinfer_config = get_nvdsinfer_yaml(
            VisualChangeNetNvDSInferConfig,
            labels_file,
            num_classes,
            output_file,
            input_shape,
            output_names
        )
        with open(nvdsinfer_yaml_file, "w") as nvds_file:
            nvds_file.write(nvdsinfer_config)

        input_tensor_id = 0
        for input_tensor_name in input_names:
            preprocess_config = VisualChangeNetPreprocessConfig()
            preprocess_config.property.tensor_name = input_tensor_name
            preprocess_config.property.processing_width = input_width
            preprocess_config.property.processing_height = input_height
            preprocess_config.property.network_input_shape = [2, *input_shape]
            preprocess_config.group.src_ids = [item + input_tensor_id for item in preprocess_config.group.src_ids]
            config_string = str(preprocess_config)
            print(f"Writing config string for input tensor {input_tensor_name} to nvdspreprocess_config_{input_tensor_id}.yaml")
            with open(os.path.join(output_root, f"nvdspreprocess_config_{input_tensor_id}.yaml"), "w") as f:
                f.write(config_string)
            input_tensor_id += 1

    model = cf_model.model
    model.eval()
    if not on_cpu:
        model.cuda()

    # create dummy input
    device = 'cpu' if on_cpu else 'cuda'
    dummy_input0 = torch.ones(input_batch_size, *input_shape, device=device)

    if num_golden == 1 or task == 'segment':
        dummy_input1 = torch.ones(input_batch_size, *input_shape, device=device)
    else:
        dummy_input1 = torch.ones(input_batch_size, num_golden, *input_shape, device=device)
    dummy_input = (dummy_input0, dummy_input1)

    if output_file.endswith('.etlt'):
        tmp_onnx_file = output_file.replace('.etlt', '.onnx')
    else:
        tmp_onnx_file = output_file

    onnx_export = ONNXExporter()
    onnx_export.export_model(model, batch_size,
                             tmp_onnx_file,
                             dummy_input,
                             input_names=input_names,
                             opset_version=opset_version,
                             output_names=output_names,
                             do_constant_folding=True,
                             verbose=experiment_config.export.verbose,
                             task=task)

    onnx_export.check_onnx(output_file)

    if output_file.endswith('.etlt') and key:
        # encrypt the onnx if and only if key is provided and output file name ends with .etlt
        encrypt_onnx(tmp_file_name=tmp_onnx_file,
                     output_file_name=output_file,
                     key=key)

        os.remove(tmp_onnx_file)
    print(f"ONNX file stored at {output_file}")


if __name__ == "__main__":
    main()
