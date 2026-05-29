# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the trainer."""

from typing import Optional, Dict, Any
from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    INT_FIELD,
    STR_FIELD,
    FLOAT_FIELD,
    DICT_FIELD,
    DATACLASS_FIELD
)
from nvidia_tao_pytorch.config.common.common_config import TrainConfig


@dataclass
class Sparse4DOptimizerConfig:
    """Optimizer config for Sparse4D."""

    type: str = STR_FIELD(
        value="adamw",
        default_value="adamw",
        description="Optimizer type",
        valid_options="adamw,adam,sgd",
        display_name="Optimizer type"
    )
    lr: float = FLOAT_FIELD(
        value=5e-5,
        default_value=5e-5,
        valid_min=0,
        valid_max="inf",
        automl_enabled="TRUE",
        description="Learning rate",
        display_name="Learning rate"
    )
    weight_decay: float = FLOAT_FIELD(
        value=0.001,
        default_value=0.001,
        description="Weight decay coefficient",
        display_name="Weight decay coefficient"
    )
    momentum: float = FLOAT_FIELD(
        value=0.9,
        default_value=0.9,
        description="Momentum for SGD",
        display_name="Momentum for SGD"
    )
    paramwise_cfg: Optional[Dict[str, Any]] = DICT_FIELD(
        hashMap={"custom_keys": {"img_backbone": {"lr_mult": 0.2}}},
        description="Parameters-wise configuration",
        default_value={"custom_keys": {"img_backbone": {"lr_mult": 0.2}}},
        display_name="Parameters-wise configuration"
    )
    grad_clip: Optional[Dict[str, Any]] = DICT_FIELD(
        hashMap={"max_norm": 25, "norm_type": "L2"},
        description="Gradient clipping configuration",
        default_value={"max_norm": 25, "norm_type": "L2"},
        display_name="Gradient clipping configuration"
    )
    lr_scheduler: Dict[str, Any] = DICT_FIELD(
        hashMap={
            "policy": "cosine",
            "warmup": "linear",
            "warmup_iters": 500,
            "warmup_ratio": 0.333333,
            "min_lr_ratio": 0.001
        },
        description="Learning rate scheduler configuration",
        default_value={
            "policy": "cosine",
            "warmup": "linear",
            "warmup_iters": 500,
            "warmup_ratio": 0.333333,
            "min_lr_ratio": 0.001
        },
        display_name="Learning rate scheduler configuration"
    )


@dataclass
class Sparse4DTrainConfig(TrainConfig):
    """Training configuration for Sparse4D."""

    validation_interval: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        description="Validation interval in epochs",
        display_name="Validation interval in epochs"
    )
    checkpoint_interval: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        description="Checkpoint interval in epochs",
        display_name="Checkpoint interval in epochs"
    )
    pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Path to pretrained model",
        display_name="Path to pretrained model"
    )
    optim: Sparse4DOptimizerConfig = DATACLASS_FIELD(
        Sparse4DOptimizerConfig(),
        description="Optimizer configuration",
        display_name="Optimizer configuration"
    )
    precision: str = STR_FIELD(
        value="bf16",
        default_value="bf16",
        description="Precision",
        display_name="Precision",
        valid_options="bf16,fp16,fp32",
    )
