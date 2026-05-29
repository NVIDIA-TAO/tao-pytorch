# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema to run inference on model."""

from dataclasses import dataclass
from omegaconf import MISSING

from nvidia_tao_pytorch.config.utils.types import (
    STR_FIELD,
    BOOL_FIELD,
    DATACLASS_FIELD
)
from nvidia_tao_pytorch.config.sparse4d.dataset import Sparse4DTrackingConfig
from nvidia_tao_pytorch.config.common.common_config import InferenceConfig


@dataclass
class Sparse4DInferenceConfig(InferenceConfig):
    """Inference configuration for Sparse4D."""

    checkpoint: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="Path to checkpoint file",
        display_name="Path to checkpoint file"
    )
    jsonfile_prefix: str = STR_FIELD(
        value="sparse4d_pred",
        default_value="sparse4d_pred",
        description="JSON file prefix",
        display_name="JSON file prefix"
    )
    output_nvschema: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Output NVSchema",
        display_name="Output NVSchema"
    )
    tracking: Sparse4DTrackingConfig = DATACLASS_FIELD(
        Sparse4DTrackingConfig(),
        description="Tracking config",
        display_name="Tracking config"
    )
