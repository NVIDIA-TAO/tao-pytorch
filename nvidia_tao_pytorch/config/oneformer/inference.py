# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
class OneFormerInferenceConfig(InferenceConfig):
    """Evaluation configuration for OneFormer."""

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
    results_dir: str = STR_FIELD(
        value="",
        description="Path to the results directory.",
        display_name="Results directory",
    )
    mode: str = STR_FIELD(
        value="semantic",
        description="Mode to run inference.",
        display_name="Mode",
        valid_options="semantic,instance,panoptic"
    )
    image_size: List[int] = LIST_FIELD(
        arrList=[1024, 1024],
        description="Size of the image.",
        display_name="Image size",
    )
    trt_engine: str = STR_FIELD(
        value="",
        description="Path to the TensorRT engine to be used for inference.",
        display_name="TensorRT Engine",
    )
    images_dir: str = STR_FIELD(
        value="",
        description="Path to the images directory.",
        display_name="Images directory",
    )
