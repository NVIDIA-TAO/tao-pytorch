#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Script to export a MAE model to ONNX format."""

import os
import torch
import torch.nn as nn
import onnx
import onnxruntime

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.ssl.mae.model.pl_model import MAEPlModule
from nvidia_tao_pytorch.config.mae.default_config import ExperimentConfig

VALIDATE_ONNX = False

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_onnx_model(
    model: nn.Module,
    input_shape: list[int],
    input_batch_size: int,
    output_path: str,
    input_names: list[str],
    output_names: list[str],
    opset_version: int = 17,
    on_cpu: bool = False,
    dynamic_axis: bool = False
) -> None:
    """Create and export a PyTorch model to ONNX format.

    This function exports a PyTorch model to ONNX format with dynamic batch size support.
    It performs the following steps:
    1. Creates a dummy input tensor with the specified shape
    2. Sets up dynamic axes for batch size flexibility
    3. Exports the model to ONNX format
    4. Verifies the exported ONNX model
    5. Tests the model with ONNX Runtime

    Args:
        model (nn.Module): The PyTorch model to export. The model should be in eval mode
            and on the correct device (CPU/GPU) before calling this function.
        input_shape (list[int]): The shape of the input tensor excluding batch size,
            in the format [channels, height, width].
        input_batch_size (int): The batch size of the input tensor.
        output_path (str): Path where the ONNX model will be saved.
        input_names (list[str]): Names for the input tensors in the ONNX model.
            Typically ['input'] for single input models.
        output_names (list[str]): Names for the output tensors in the ONNX model.
            Typically ['output'] for single output models.
        opset_version (int, optional): ONNX opset version to use for export. Defaults to 17.
        on_cpu (bool, optional): Whether to export the model on CPU. Defaults to False.
        dynamic_axis (bool, optional): Whether to use dynamic axes for the ONNX model. Defaults to False.

    Raises:
        AssertionError: If the ONNX model verification fails.
        RuntimeError: If ONNX Runtime inference fails or if there are issues during export.

    Note:
        The model is loaded in eval mode and moved to the appropriate device (CPU/GPU)
        before export.
        The exported model will support dynamic batch sizes through the dynamic_axes
        configuration.

    Example:
        >>> create_onnx_model(
        ...     model=model,
        ...     input_shape=[3, 224, 224],
        ...     output_path="model.onnx",
        ...     input_names=["input"],
        ...     output_names=["output"]
        ... )
    """
    # Set model to eval mode and move to correct device
    model.eval()
    if not on_cpu:
        model.cuda()
    model.float()

    if input_shape[0] not in [1, 3]:
        raise ValueError(
            f"Invalid input channel: {input_shape[0]}. Only 1 or 3 are supported."
        )

    # Warn the user if an exported file already exists.
    if os.path.exists(output_path):
        raise ValueError(
            f"Default onnx file {output_path} already exists"
        )

    # Create dummy input
    if on_cpu:
        dummy_input = torch.ones(input_batch_size, *input_shape, device='cpu').float()
    else:
        dummy_input = torch.ones(input_batch_size, *input_shape, device='cuda').float()

    # Create dynamic axes
    if dynamic_axis:
        print("Using dynamic axes")
        dynamic_axes = {}
        for input_name in input_names:
            dynamic_axes[input_name] = {0: 'batch_size'}
        for output_name in output_names:
            dynamic_axes[output_name] = {0: 'batch_size'}
    else:
        dynamic_axes = None

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    # Verify ONNX model
    logging.info("Verifying ONNX model")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Test ONNX model with ONNX Runtime
    if VALIDATE_ONNX:
        logging.info("Validating ONNX model with ONNX Runtime")
        ort_session = onnxruntime.InferenceSession(output_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        outputs = ort_session.run(None, ort_inputs)
        logging.info("ONNX model outputs: {num_tensors}".format(num_tensors=len(outputs)))
        logging.info("ONNX model output shapes: {shapes}".format(shapes=[output.shape for output in outputs]))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="MAE", mode="export")
def main(cfg: ExperimentConfig) -> None:
    """Entry point for MAE model export process.

    This function serves as the main entry point for exporting a trained MAE (Masked Autoencoder)
    model. It configures the PyTorch backend settings and delegates the actual export process
    to the run_export function.

    Args:
        cfg (ExperimentConfig): Hydra configuration object containing all export parameters.
            This object is automatically populated by Hydra based on the experiment spec
            and command line arguments.

    Note:
        This function is decorated with @hydra_runner and @monitor_status to handle
        configuration loading and workflow monitoring respectively. The actual export
        logic is implemented in the run_export function.

    Example:
        >>> # The function is typically called via command line:
        >>> # python export.py export.onnx_file=model.onnx export.input_channel=3
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run_export(cfg)


def run_export(experiment_config: ExperimentConfig) -> None:
    """Execute the MAE model export process to ONNX format.

    This function handles the core export process for a trained MAE (Masked Autoencoder) model.
    It processes the experiment configuration, sets up the model with proper encryption,
    and exports it to ONNX format with the specified parameters.

    Args:
        experiment_config (ExperimentConfig): Configuration object containing export parameters:
            - export.gpu_id: GPU device ID to use for export
            - export.checkpoint: Path to the model checkpoint file
            - encryption_key: Key for model encryption/decryption
            - export.onnx_file: Output path for the ONNX model
            - export.input_channel: Number of input channels (e.g., 3 for RGB)
            - export.input_width: Input image width
            - export.input_height: Input image height
            - export.opset_version: ONNX opset version for export
            - export.batch_size: Batch size for export (defaults to 1 if None or -1)
            - export.on_cpu: Whether to perform export on CPU instead of GPU
            - train.pretrained_model_path: Path to the pretrained model weights

    Raises:
        AssertionError: If the output ONNX file already exists at the specified path.
        RuntimeError: If model loading fails or if there are issues during ONNX export.
        ValueError: If required configuration parameters are missing or invalid.

    Note:
        The function handles both CPU and GPU exports based on the configuration.
        For GPU exports, it sets the appropriate CUDA device before proceeding.

    Example:
        >>> config = ExperimentConfig()
        >>> config.export.onnx_file = "model.onnx"
        >>> config.export.input_channel = 3
        >>> config.export.input_width = 224
        >>> config.export.input_height = 224
        >>> run_export(config)
    """
    # Convert DictConfig to ExperimentConfig
    gpu_id = experiment_config.export.gpu_id
    torch.cuda.set_device(gpu_id)

    # Parsing command line arguments.
    model_path = experiment_config.export.checkpoint
    key = experiment_config.encryption_key

    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    output_file = experiment_config.export.onnx_file
    training_stage = experiment_config.train.stage
    input_channel = experiment_config.export.input_channel
    input_width = experiment_config.export.input_width
    input_height = experiment_config.export.input_height
    input_shape = [input_channel, input_height, input_width]
    opset_version = experiment_config.export.opset_version
    batch_size = experiment_config.export.batch_size
    on_cpu = experiment_config.export.on_cpu
    if batch_size is None or batch_size == -1:
        input_batch_size = 1
    else:
        input_batch_size = batch_size

    # Set default output filename if the filename.
    if output_file is None:
        split_name = os.path.splitext(model_path)[0]
        output_file = "{}.onnx".format(split_name)

    # Create output directory
    output_root = os.path.dirname(os.path.realpath(output_file))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Setting input/output tensor names.
    input_names = ['input']
    output_names = ['output']

    # Load model
    if training_stage == "pretrain":
        # WAR to construct the backbone from finetune stage
        # and load weights from the checkpoint file.
        experiment_config.train.pretrained_model_path = model_path
        mae_model = MAEPlModule(cfg=experiment_config, export=True)
    else:
        # During finetune stage, the model.forward() already accounts for the head, which is a single tensor
        # so we load the full model from the model checkpoint.
        mae_model = MAEPlModule.load_from_checkpoint(
            model_path,
            cfg=experiment_config,
            export=True,  # to remove all intermediate layers and get the backbone during pre-train stage.
            map_location='cpu'
        )

    model = mae_model.model
    try:
        dynamic_axis = False
        if batch_size == -1:
            dynamic_axis = True
        create_onnx_model(
            model,
            input_shape,
            input_batch_size,
            output_file,
            input_names,
            output_names,
            on_cpu=on_cpu,
            dynamic_axis=dynamic_axis,
            opset_version=opset_version,
        )
        logging.info("ONNX model saved to {output_file}".format(output_file=output_file))
    except Exception as e:
        logging.error(f"Error exporting model: {e}")
        raise e


if __name__ == "__main__":
    main()
