# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export DINO model to ONNX."""

import os
import torch

from nvidia_tao_pytorch.config.dino.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.utilities import encrypt_onnx, write_classes_file, get_nvdsinfer_yaml
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.dino.model.pl_dino_model import DINOPlModel
from nvidia_tao_pytorch.cv.dino.types.dino_nvdsinfer import DINONvDSInferConfig
from nvidia_tao_pytorch.cv.dino.utils.onnx_export import ONNXExporter


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="export", schema=ExperimentConfig
)
@monitor_status(name="DINO", mode="export")
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
    from nvidia_tao_pytorch.cv.deformable_detr.dataloader.pl_od_data_module import ODDataModule

    dm = ODDataModule(experiment_config.dataset)
    dm.setup(stage="fit")

    categories = dm.val_dataset.label_map
    categories = sorted(categories, key=lambda x: x['id'])
    class_ids = [cat['id'] for cat in categories]
    class_names = ["unknown"] * (max(class_ids) + 1)
    for cat in categories:
        class_names[cat['id']] = cat['name']

    if min(class_ids) > 0:
        class_names[0] = "background"

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
    key = experiment_config.encryption_key

    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    output_file = experiment_config.export.onnx_file
    input_channel = experiment_config.export.input_channel
    input_width = experiment_config.export.input_width
    input_height = experiment_config.export.input_height
    input_shape = [input_channel, input_height, input_width]
    opset_version = experiment_config.export.opset_version
    batch_size = experiment_config.export.batch_size
    serialize_nvdsinfer = experiment_config.export.serialize_nvdsinfer
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

    # Setting up input/output tensor names.
    input_names = ['inputs']
    output_names = ["pred_logits", "pred_boxes"]

    if serialize_nvdsinfer:
        # Get class information
        num_classes, class_names = get_model_classes(experiment_config)
        logging.info(f"Model was trained on {num_classes} classes: {class_names}")

        # Write class names to labels file
        labels_file = os.path.join(output_root, "labels.txt")
        write_classes_file(labels_file, class_names, delimiter='\n')

        nvdsinfer_yaml_file = os.path.join(
            output_root, "nvdsinfer_config.yaml"
        )
        logging.info("Serializing the deepstream config to {}".format(
            nvdsinfer_yaml_file
        ))
        nvdsinfer_config = get_nvdsinfer_yaml(
            DINONvDSInferConfig,
            labels_file,
            num_classes,
            output_file,
            input_shape,
            output_names
        )

        with open(nvdsinfer_yaml_file, "w") as nvds_file:
            nvds_file.write(nvdsinfer_config)

    # load model
    pl_model = DINOPlModel.load_from_checkpoint(model_path,
                                                map_location='cpu' if on_cpu else 'cuda',
                                                experiment_spec=experiment_config,
                                                export=True)
    model = pl_model.model
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

    onnx_export = ONNXExporter()
    onnx_export.export_model(model, batch_size,
                             tmp_onnx_file,
                             dummy_input,
                             input_names=input_names,
                             opset_version=opset_version,
                             output_names=output_names,
                             do_constant_folding=False,
                             verbose=experiment_config.export.verbose)
    onnx_export.check_onnx(tmp_onnx_file)
    onnx_export.onnx_change(tmp_onnx_file)

    if output_file.endswith('.etlt') and key:
        # encrypt the onnx if and only if key is provided and output file name ends with .etlt
        encrypt_onnx(tmp_file_name=tmp_onnx_file,
                     output_file_name=output_file,
                     key=key)

        os.remove(tmp_onnx_file)
    logging.info(f"ONNX file stored at {output_file}")


if __name__ == "__main__":
    main()
