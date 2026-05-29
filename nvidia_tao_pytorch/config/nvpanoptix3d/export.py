# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for export."""

from dataclasses import dataclass
from nvidia_tao_pytorch.config.common.common_config import ExportConfig

from nvidia_tao_pytorch.config.utils.types import (
    INT_FIELD,
    STR_FIELD
)


@dataclass
class NVPanoptix3DExportExpConfig(ExportConfig):
    """NVPanoptix3D export ONNX experiment config."""

    onnx_file_2d: str = STR_FIELD(
        value="",
        default_value="",
        display_name="onnx file 2d",
        description="Path to the onnx model 2d file."
    )
    onnx_file_3d: str = STR_FIELD(
        value="",
        default_value="",
        display_name="onnx file 3d",
        description="Path to the onnx model 3d file."
    )
    max_voxels: int = INT_FIELD(
        value=700000,
        default_value=700000,
        valid_min=1,
        description="The maximum number of voxels in the input Tensor for the engine.",
        display_name="max voxels"
    )
