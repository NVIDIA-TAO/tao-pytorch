# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the evaluation."""

from typing import List
from dataclasses import dataclass
from nvidia_tao_pytorch.config.common.common_config import InferenceConfig

from nvidia_tao_pytorch.config.utils.types import (
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD
)


@dataclass
class NVPanoptix3DInferenceExpConfig(InferenceConfig):
    """NVPanoptix3D inference configuration."""

    num_gpus: int = INT_FIELD(
        value=1,
        valid_min=1,
        display_name="Number of GPUs",
        description="The number of GPUs to run the evaluation job.",
        popular="yes",
    )
    gpu_ids: List[int] = LIST_FIELD(
        arrList=[0],
        display_name="GPU IDs",
        description="List of GPU IDs to run the evaluation on. The length must equal evaluate.num_gpus.",
        popular="yes",
    )
    checkpoint: str = STR_FIELD(
        value="",
        description="Path to the checkpoint used for evaluation.",
        display_name="Checkpoint path",
    )
    mode: str = STR_FIELD(
        value="panoptic",
        description="Mode to run inference.",
        display_name="Mode",
        valid_options="semantic,instance,panoptic"
    )
    trt_engine: str = STR_FIELD(
        value="",
        description="Path to the TensorRT engine folder to be used for inference.",
        display_name="TensorRT Engine folder",
    )
    images_dir: str = STR_FIELD(
        value="",
        description="Path to the images directory.",
        display_name="Images directory",
    )
