# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for export."""

from typing import Optional
from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    INT_FIELD,
    STR_FIELD
)


@dataclass
class OneFormerExportExpConfig:
    """Evaluation experiment config."""

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="Results directory",
        description="""
        Path to where all the assets generated from a task are stored.
        """
    )
    gpu_id: int = INT_FIELD(
        value=0,
        default_value=0,
        description="""The index of the GPU to build the TensorRT engine.""",
        display_name="GPU ID"
    )
    checkpoint: str = STR_FIELD(
        value="",
        default_value="",
        description="Path to the checkpoint file to run export.",
        display_name="checkpoint"
    )
    task: str = STR_FIELD(
        value="semantic",
        default_value="semantic",
        description="Segmentation task to export.",
        display_name="task",
        valid_options="semantic,instance,panoptic"
    )
    onnx_file: str = STR_FIELD(
        value="",
        default_value="",
        display_name="onnx file",
        description="Path to the onnx model file."
    )
    on_cpu: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="verbose",
        description="Flag to export CPU compatible model."
    )
    input_channel: int = INT_FIELD(
        value=3,
        default_value=3,
        description="Number of channels in the input Tensor.",
        display_name="input channel",
        valid_min=3,
    )
    input_width: int = INT_FIELD(
        value=640,
        default_value=640,
        description="Width of the input image tensor.",
        display_name="input width",
        valid_min=32,
    )
    input_height: int = INT_FIELD(
        value=640,
        default_value=640,
        description="Height of the input image tensor.",
        display_name="input height",
        valid_min=32,
    )
    opset_version: int = INT_FIELD(
        value=17,
        default_value=17,
        description="Operator set version of the ONNX model used to generate the TensorRT engine.",
        display_name="opset version",
        valid_min=1,
    )
    batch_size: int = INT_FIELD(
        value=-1,
        default_value=-1,
        valid_min=-1,
        description="The batch size of the input Tensor for the engine. A value of -1 implies dynamic tensor shapes.",
        display_name="batch size"
    )
    verbose: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="verbose",
        description="Flag to enable verbose TensorRT logging."
    )
