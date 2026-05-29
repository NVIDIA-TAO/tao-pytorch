# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the evaluation."""

from typing import List
from dataclasses import dataclass
from nvidia_tao_pytorch.config.common.common_config import EvaluateConfig

from nvidia_tao_pytorch.config.utils.types import (
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
    BOOL_FIELD
)


@dataclass
class OneFormerEvaluateConfig(EvaluateConfig):
    """Evaluation configuration for OneFormer."""

    num_gpus: int = INT_FIELD(
        value=1,
        valid_min=1,
        display_name="Number of GPUs",
        description="""The number of GPUs to run the evaluation job.""",
        popular="yes",
    )
    gpu_ids: List[int] = LIST_FIELD(
        arrList=[0],
        display_name="GPU IDs",
        description="""
        List of GPU IDs to run the evaluation on. The length of this list
        must be equal to the number of gpus in evaluate.num_gpus.""",
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
    trt_engine: str = STR_FIELD(
        value="",
        description="Path to the TensorRT engine to be used for evaluation.",
        display_name="TensorRT Engine",
    )
    iou_per_class: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Whether to log the IoU per class.",
        display_name="IoU per class",
    )
