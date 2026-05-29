# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the trainer."""

from typing import Optional, List, Union
from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD
)
from nvidia_tao_pytorch.config.common.common_config import TrainConfig


@dataclass
class OptimConfig:
    """Optimizer config."""

    type: str = STR_FIELD(
        value="AdamW",
        default_value="AdamW",
        description="Type of optimizer used to train the network.",
        valid_options=",".join([
            "AdamW"
        ])
    )
    monitor_name: str = STR_FIELD(
        value="val_loss",
        default_value="val_loss",
        description="The metric value to be monitored for the :code:`AutoReduce` Scheduler.",
        display_name="monitor_name",
        valid_options=",".join(
            ["val_loss", "train_loss"]
        )
    )
    lr: float = FLOAT_FIELD(
        value=2e-4,
        default_value=2e-4,
        valid_min=1e-6,
        valid_max=1e-2,
        math_cond="> 0.0",
        display_name="learning rate",
        description="The initial learning rate for training the model.",
        automl_enabled="TRUE"
    )
    backbone_multiplier: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        valid_min=0.01,
        valid_max=1.0,
        math_cond="> 0.0",
        display_name="backbone learning rate multiplier",
        description="A multiplier for backbone learning rate.",
    )
    momentum: float = FLOAT_FIELD(
        value=0.9,
        default_value=0.9,
        valid_min=0.5,
        valid_max=0.999,
        math_cond="> 0.0",
        display_name="momentum - AdamW",
        description="The momentum for the AdamW optimizer.",
    )
    weight_decay: float = FLOAT_FIELD(
        value=0.05,
        default_value=0.05,
        valid_min=1e-4,
        valid_max=0.5,
        math_cond="> 0.0",
        display_name="weight decay",
        description="The weight decay coefficient.",
    )
    lr_scheduler: str = STR_FIELD(
        value="MultiStep",
        default_value="MultiStep",
        description="""The learning scheduler:
                    * MultiStep : Decrease the lr by lr_decay from lr_steps
                    * Warmuppoly : Poly learning rate schedule.""",
        display_name="learning rate scheduler",
        valid_options=",".join(
            ["MultiStep", "Warmuppoly"]
        )
    )
    milestones: List[int] = LIST_FIELD(
        arrList=[88, 96],
        default_value=[88, 96],
        description="""learning rate decay epochs.""",
        display_name="learning rate decay epochs."
    )
    gamma: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        math_cond="> 0.0",
        display_name="gamma",
        description="Multiplicative factor of learning rate decay.",
    )
    max_steps: int = INT_FIELD(
        value=160000,
        default_value=160000,
        math_cond="> 0",
        display_name="max steps",
        description="The maximum number of steps to train the model.",
    )
    warmup_factor: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        math_cond="> 0.0",
        display_name="warmup factor",
        description="The warmup factor for the learning rate scheduler.",
    )
    warmup_iters: int = INT_FIELD(
        value=0,
        default_value=0,
        math_cond="> 0",
        display_name="warmup iters",
        description="The number of warmup iterations.",
    )


@dataclass
class NVPanoptix3DTrainExpConfig(TrainConfig):
    """Train experiment config."""

    checkpoint_2d: str = STR_FIELD(
        value="",
        default_value="",
        description="Path to 2D stage checkpoint to initialize the 3D stage training.",
        display_name="2D stage checkpoint path"
    )
    checkpoint_3d: str = STR_FIELD(
        value="",
        default_value="",
        description="Path to 3D stage checkpoint to initialize the 3D stage training.",
        display_name="3D stage checkpoint path"
    )
    val_check_interval: int = INT_FIELD(
        value=5,
        default_value=5,
        math_cond="> 0",
        display_name="val check interval",
        description="The number of iterations between validation checks.",
    )
    freeze: Optional[List[str]] = LIST_FIELD(
        arrList=[],
        default_value=[],
        description="""
        List of layer names to freeze.
        Example: ["backbone", "transformer.encoder", "input_proj"].""",
        display_name="freeze"
    )
    clip_grad_norm: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        math_cond="> 0.0",
        display_name="clip gradient norm",
        description="Amount to clip the gradient by L2 Norm.",
    )
    clip_grad_norm_type: Union[float, str] = 2.0
    clip_grad_type: str = STR_FIELD(
        value='full',
        default_value='full',
        display_name='clip gradient type',
        description="Gradient clip type."
    )
    is_dry_run: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Is dry run",
        description="Whether to run the trainer in Dry Run mode.",
    )
    optim: OptimConfig = DATACLASS_FIELD(
        OptimConfig(),
        display_name="optimizer",
        description="Hyper parameters to configure the optimizer."
    )
    precision: str = STR_FIELD(
        value="fp32",
        default_value="fp32",
        description="Precision to run the training on.",
        display_name="precision",
        valid_options=",".join([
            "fp16", "fp32",
        ])
    )
    distributed_strategy: str = STR_FIELD(
        value="ddp",
        default_value="ddp",
        valid_options=",".join(
            ["ddp", "fsdp"]
        ),
        display_name="distributed_strategy",
        description="""
        The multi-GPU training strategy.
        DDP (Distributed Data Parallel) and Fully Sharded DDP are supported.""",
    )
    activation_checkpoint: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="enable activation checkpointing",
        description="""
        A True value instructs train to recompute in backward pass to save GPU memory,
        rather than storing activations.""",
    )
    verbose: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="enable verbose logs",
        description="Flag to enable printing of detailed learning rate scaling from the optimizer.",
    )
    iters_per_epoch: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        display_name="iteration per epoch",
        description="Number of iteration per epoch.",
    )
    results_dir: str = STR_FIELD(
        value="",
        default_value="",
        description="The folder to save the experiment.",
        display_name="results directory"
    )
