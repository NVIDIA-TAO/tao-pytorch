# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the trainer."""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    BOOL_FIELD,
    LIST_FIELD,
    DICT_FIELD,
    DATACLASS_FIELD
)
from nvidia_tao_pytorch.config.common.common_config import TrainConfig


@dataclass
class BEVFusionOptimConfig:
    """Configuration parameters for Optimizer."""

    type: str = STR_FIELD(
        value="AdamW",
        display_name="Optimizer",
        description="Type of optimizer used to train the network."
    )
    lr: float = FLOAT_FIELD(
        value=2e-4,
        math_cond="> 0.0",
        display_name="learning rate",
        description="The initial learning rate for training the model."
    )
    weight_decay: float = FLOAT_FIELD(
        value=0.01,
        math_cond="> 0.0",
        display_name="weight decay",
        description="The weight decay coefficient."
    )
    betas: List[float] = LIST_FIELD(
        arrList=[0.9, 0.999],
        display_name="moving average beta",
        description="The moving average parameter for adaptive learning rate."
    )
    clip_grad: Dict[str, int] = DICT_FIELD(
        hashMap={"max_norm": 35, "norm_type": 2},
        display_name="clip gradient norm",
        description="Clip the gradient norm of an iterable of parameters."
    )
    wrapper_type: str = STR_FIELD(
        value="OptimWrapper",
        display_name="Optimizer wrapper",
        description="Opitmizer Wrapper in MMengine. AmpOptimWrapper to enables mixed precision training"
    )


@dataclass
class BEVFusionTrainExpConfig(TrainConfig):
    """Configuration parameters for Train Exp."""

    by_epoch: bool = BOOL_FIELD(
        value=True,
        display_name="by epoch",
        description="Whether EpochBasedRunner is used.",
    )
    logging_interval: int = INT_FIELD(
        value=1,
        display_name="logging interval",
        description="logging interval every k iterations.",
    )
    resume: bool = BOOL_FIELD(
        value=False,
        display_name="Is resume",
        description="Whether to resume the training or not.",
    )
    pretrained_checkpoint: Optional[str] = STR_FIELD(
        value=None,
        default_type=None,
        description="Path to a pre-trained BEVFusion model to initialize the current training from."
    )
    optimizer: BEVFusionOptimConfig = DATACLASS_FIELD(
        BEVFusionOptimConfig(),
        description="Hyper parameters to configure the optimizer",
        display_name="optimizer"
    )
    lr_scheduler: List[Dict[Any, Any]] = LIST_FIELD(
        arrList=[
            {'type': 'LinearLR', 'start_factor': 0.33333333, 'by_epoch': False, 'begin': 0, 'end': 500},
            {
                'type': 'CosineAnnealingLR',
                'T_max': 10,
                'eta_min_ratio': 1e-4,
                'begin': 0,
                'end': 10,
                'by_epoch': True
            },
            {'type': 'CosineAnnealingMomentum', 'eta_min': 0.8947, 'begin': 0, 'end': 2.4, 'by_epoch': True},
            {'type': 'CosineAnnealingMomentum',  'eta_min': 1, 'begin': 2.4, 'end': 10, 'by_epoch': True}
        ],
        description="Hyper parameters to configure the learning rate scheduler.",
        display_name="learning rate scheduler."
    )
