# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the trainer."""

from dataclasses import dataclass
from typing import List, Optional

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
)
from nvidia_tao_pytorch.config.common.common_config import TrainConfig


@dataclass
class LRConfig:
    """Optimizer learning rate configuration for the LR scheduler."""

    bias_lr_factor: float = FLOAT_FIELD(
        value=1,
        valid_min=1.0,
        valid_max=10.0,
        math_cond=">= 1",
        display_name="bias lr factor",
        description="The bias learning rate factor for the WarmupMultiStepLR",
        automl_enabled="TRUE"
    )
    base_lr: float = FLOAT_FIELD(
        value=0.00035,
        valid_min=0.0,
        valid_max=1.0,
        math_cond="> 0.0",
        display_name="base lr",
        description="The initial learning rate for the training",
        automl_enabled="TRUE"
    )
    momentum: float = FLOAT_FIELD(
        value=0.9,
        valid_min=0.0,
        valid_max=1.0,
        math_cond="> 0.0",
        display_name="momentum",
        description="The momentum for the WarmupMultiStepLR optimizer",
        automl_enabled="TRUE"
    )
    weight_decay: float = FLOAT_FIELD(
        value=0.0005,
        valid_min=0.0,
        valid_max=1.0,
        math_cond="> 0.0",
        display_name="weight_decay",
        description="The weight decay coefficient for the optimizer",
        automl_enabled="TRUE"
    )
    weight_decay_bias: float = FLOAT_FIELD(
        value=0.0005,
        valid_min=0.0,
        valid_max=1.0,
        math_cond="> 0.0",
        display_name="weight_decay_bias",
        description="The weight decay bias for the optimizer",
        automl_enabled="TRUE"
    )


@dataclass
class OptimConfig:
    """Optimizer configuration for the LR scheduler."""

    name: str = STR_FIELD(
        value="Adam",
        default_value="Adam",
        display_name="name",
        description="The name of the optimizer. The Algorithms in torch.optim are supported.",
        valid_options="Adam, SGD, Adamax",
    )
    steps: List[int] = LIST_FIELD(
        arrList=[40, 70],
        description="The steps to decrease the learning rate for the MultiStep scheduler.",
        display_name="steps"
    )
    gamma: float = FLOAT_FIELD(
        value=0.1,
        valid_min=0.0,
        valid_max=1.0,
        math_cond="> 0.0",
        display_name="gamma",
        description="The decay rate for the WarmupMultiStepLR scheduler",
        automl_enabled="TRUE"
    )
    warmup_factor: float = FLOAT_FIELD(
        value=0.01,
        valid_min=0.0,
        valid_max=1.0,
        math_cond="> 0.0",
        display_name="warmup_factor",
        description="The warmup factor for the WarmupMultiStepLR scheduler",
        automl_enabled="TRUE"
    )
    warmup_iters: int = INT_FIELD(
        value=10,
        valid_min=1,
        valid_max=10000,
        math_cond="> 0",
        description="The number of warmup iterations for the WarmupMultiStepLR scheduler.",
        display_name="warmup_iters",
        automl_enabled="TRUE"
    )
    warmup_method: str = STR_FIELD(
        value="linear",
        default_value="linear",
        display_name="warmup_method",
        description="The warmup method for the optimizer",
        valid_options="linear, constant",
    )
    triplet_loss_margin: float = FLOAT_FIELD(
        value=0.3,
        valid_min=0.0,
        valid_max=2.0,
        math_cond="> 0.0",
        display_name="triplet_loss_margin",
        description="""The desired difference between the anchor-positive distance and the
                    anchor-negative distance""",
        automl_enabled="TRUE"
    )
    embedder: LRConfig = DATACLASS_FIELD(
        LRConfig(),
        description="The learning rate configuration for the embedder part of the model.",
        display_name="embedder",
    )
    trunk: LRConfig = DATACLASS_FIELD(
        LRConfig(),
        description="The learning rate configuration for the trunk part of the model.",
        display_name="trunk",
    )
    miner_function_margin: float = FLOAT_FIELD(
        value=0.1,
        valid_min=0.0,
        valid_max=2.0,
        math_cond="> 0.0",
        display_name="triplet_loss_margin",
        description="""Negative pairs are chosen if they have similarity greater than the hardest
                    positive pair, minus this margin; positive pairs are chosen if they have
                    similarity less than the hardest negative pair, plus the margin""",
        automl_enabled="TRUE"
    )


@dataclass
class MLTrainExpConfig(TrainConfig):
    """Train experiment configuration template."""

    optim: OptimConfig = DATACLASS_FIELD(
        OptimConfig(),
        description="Training optimization config.",
        display_name="Optimization config",
    )
    clip_grad_norm: float = FLOAT_FIELD(
        value=0.0,
        math_cond=">= 0.0",
        display_name="clip_grad_norm",
        description="The amount to clip the gradient by the L2 norm. A value of 0.0 specifies no clipping.",
    )
    report_accuracy_per_class: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to report accuracy per class at valiation or not.",
        display_name="report accuracy per class"
    )
    smooth_loss: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to smooth the triplet margin loss or not.",
        display_name="smooth_loss"
    )
    batch_size: int = INT_FIELD(
        value=4,
        default_value=4,
        description="The train batch size",
        display_name="batch_size",
    )
    val_batch_size: int = INT_FIELD(
        value=4,
        default_value=4,
        description="The validation batch size",
        display_name="val_batch_size",
    )
    train_trunk: Optional[bool] = BOOL_FIELD(
        value=True,
        default_value=True,
        description="[Optional] If False, the trunk part of the model would be frozen during training",
        display_name="train_trunk"
    )
    train_embedder: Optional[bool] = BOOL_FIELD(
        value=True,
        default_value=True,
        description="[Optional] If False, the embedder part of the model would be frozen during training",
        display_name="train_embedder"
    )
