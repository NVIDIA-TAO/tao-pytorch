# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export RT-DETR model to ONNX."""

import os
import torch

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.deformable_detr.utils.onnx_export import ONNXExporter
from nvidia_tao_pytorch.config.rtdetr.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.quantization.utils import create_quantized_model_from_config
from nvidia_tao_pytorch.cv.rtdetr.model.pl_rtdetr_model import RTDETRPlModel
from nvidia_tao_pytorch.cv.rtdetr.types.rtdetr_nvdsinfer import RTDETRNvDSInferConfig
from nvidia_tao_pytorch.core.utilities import write_classes_file, get_nvdsinfer_yaml
from nvidia_tao_pytorch.cv.rtdetr.dataloader.pl_od_data_module import ODDataModule
from nvidia_tao_pytorch.cv.rtdetr.dataloader.od_dataset import mscoco_category2name, mscoco_label2category


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="export", schema=ExperimentConfig
)
@monitor_status(name="RT-DETR", mode="export")
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


def get_class_info(experiment_config):
    """Get class information from the validation dataset.

    Args:
        experiment_config: Experiment configuration containing model info

    Returns:
        tuple: (num_classes, class_names)
    """
    num_classes = experiment_config.dataset.num_classes

    # Initialize the datamodule and setup validation dataset
    data_module = ODDataModule(experiment_config.dataset)
    data_module.setup('fit')  # 'fit' setup initializes both train and validation datasets

    if experiment_config.dataset.remap_mscoco_category:
        # When using MSCOCO remapping, class indices are remapped from original MSCOCO indices
        # to contiguous indices [0, num_classes-1]
        class_names = []
        for label in range(num_classes):
            # Convert label back to original MSCOCO category ID and get name
            category_id = mscoco_label2category[label]
            class_names.append(mscoco_category2name[category_id])
    else:
        # For custom datasets, get class names from the validation dataset's label map
        label_map = data_module.val_dataset.label_map

        # Extract class names in order from label map
        class_names = [category["name"] for category in sorted(label_map, key=lambda x: x["id"])]

    # Verify num_classes matches configuration
    if len(class_names) != num_classes:
        assert num_classes == len(class_names) + 1, "Number of classes in validation dataset label map ({len(class_names)}) "
        class_names.insert(0, "background")

    return len(class_names), class_names


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

    output_file = experiment_config.export.onnx_file
    input_channel = experiment_config.export.input_channel
    input_width = experiment_config.export.input_width
    input_height = experiment_config.export.input_height
    input_shape = [input_channel, input_height, input_width]
    opset_version = experiment_config.export.opset_version
    serialize_nvdsinfer = experiment_config.export.serialize_nvdsinfer
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
    input_names = ['inputs']
    output_names = ["pred_logits", "pred_boxes"]

    if serialize_nvdsinfer:
        # Get class information from validation dataset
        num_classes, class_names = get_class_info(experiment_config)

        # Write classes file
        classes_file = os.path.join(output_root, "labels.txt")
        write_classes_file(classes_file, class_names, delimiter='\n')

        # Generate nvdsinfer yaml
        nvdsinfer_yaml_file = os.path.join(output_root, "nvdsinfer_config.yaml")

        logging.info("Serializing the deepstream config to {}".format(nvdsinfer_yaml_file))
        nvds_config_str = get_nvdsinfer_yaml(
            RTDETRNvDSInferConfig,
            classes_file,
            num_classes,
            output_file,
            input_shape,
            output_names
        )

        with open(nvdsinfer_yaml_file, "w") as nvds_file:
            nvds_file.write(nvds_config_str)

    # load model (supports quantized models)
    if experiment_config.export.is_quantized:
        pl_model = create_quantized_model_from_config(model_path, RTDETRPlModel, experiment_config=experiment_config, export=True)
    else:
        pl_model = RTDETRPlModel.load_from_checkpoint(
            model_path,
            map_location="cpu",
            experiment_spec=experiment_config,
            export=True,
        )
    model = pl_model.model
    model.eval()
    if not on_cpu:
        model.cuda()

    # create dummy input
    if on_cpu:
        dummy_input = torch.ones(input_batch_size, *input_shape, device='cpu')
    else:
        dummy_input = torch.ones(input_batch_size, *input_shape, device='cuda')

    onnx_export = ONNXExporter()
    onnx_export.export_model(model, batch_size,
                             output_file,
                             dummy_input,
                             input_names=input_names,
                             opset_version=opset_version,
                             output_names=output_names,
                             do_constant_folding=False,
                             verbose=experiment_config.export.verbose)
    onnx_export.check_onnx(output_file)
    onnx_export.onnx_change(output_file)

    logging.info(f"ONNX file stored at {output_file}")


if __name__ == "__main__":
    main()
