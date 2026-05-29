# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the deploy the model."""

from dataclasses import dataclass
from nvidia_tao_pytorch.config.utils.types import (
    DATACLASS_FIELD,
    STR_FIELD,
    INT_FIELD,
    BOOL_FIELD
)
from nvidia_tao_pytorch.config.common.common_config import (
    GenTrtEngineConfig,
    TrtConfig
)


@dataclass
class OneFormerTrtConfig(TrtConfig):
    """Trt config."""

    data_type: str = STR_FIELD(
        value="fp16",
        default_value="fp16",
        description="The precision to be set for building the TensorRT engine.",
        display_name="data type",
        valid_options=",".join(["fp16", "fp32"])
    )
    workspace_size: int = INT_FIELD(
        value=1024,
        default_value=1024,
        valid_min=1,
        description="The workspace size to be set for building the TensorRT engine.",
        display_name="workspace size",
    )


@dataclass
class OneFormerGenTrtEngineExpConfig(GenTrtEngineConfig):
    """Gen TRT Engine experiment config."""

    tensorrt: OneFormerTrtConfig = DATACLASS_FIELD(
        OneFormerTrtConfig(),
        description="Hyper parameters to configure the TensorRT Engine builder.",
        display_name="TensorRT hyper params."
    )
    trt_engine: str = STR_FIELD(
        value="",
        default_value="",
        description="Path to the TensorRT engine file.",
        display_name="TensorRT engine",
    )
    onnx_file: str = STR_FIELD(
        value="",
        default_value="",
        description="Path to the ONNX model file.",
        display_name="ONNX file",
    )
    batch_size: int = INT_FIELD(
        value=1,
        default_value=0,
        description="Batch size for the TensorRT engine.",
        display_name="Batch size",
    )
    verbose: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Verbose for the TensorRT engine.",
        display_name="Verbose",
    )
