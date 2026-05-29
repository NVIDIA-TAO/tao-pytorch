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
class OptimConfig:
    """Optimizer config."""

    optimizer: str = STR_FIELD(
        value="AdamW",
        default_value="AdamW",
        description="Type of optimizer used to train the network.",
        valid_options=",".join([
            "AdamW", "SGD"
        ])
    )
    monitor_name: str = STR_FIELD(
        value="val_loss",
        description="The metric value to be monitored for the :code:`AutoReduce` Scheduler.",
        display_name="monitor_name",
        valid_options=",".join(
            ["val_loss", "train_loss"]
        )
    )
    lr: float = FLOAT_FIELD(
        value=2e-4,
        valid_min=0.0,
        valid_max=0.01,
        math_cond="> 0.0",
        display_name="learning rate",
        description="The initial learning rate for training the model, excluding the backbone.",
        automl_enabled="TRUE"
    )
    lr_backbone: float = FLOAT_FIELD(
        value=2e-5,
        valid_min=0.0,
        valid_max=0.001,
        math_cond="> 0.0",
        display_name="learning rate - backbone",
        description="The initial learning rate for training the backbone.",
        automl_enabled="TRUE"
    )
    lr_linear_proj_mult: float = FLOAT_FIELD(
        value=0.1,
        valid_min=0.0,
        valid_max=1.0,
        math_cond="> 0.0",
        display_name="learning rate - linear projection",
        description="The initial learning rate for training the linear projection layer.",
        automl_enabled="TRUE"
    )
    momentum: float = FLOAT_FIELD(
        value=0.9,
        valid_min=0.0,
        valid_max=1.0,
        math_cond="> 0.0",
        display_name="momentum - AdamW",
        description="The momentum for the AdamW optimizer.",
        automl_enabled="TRUE"
    )
    weight_decay: float = FLOAT_FIELD(
        value=1e-4,
        valid_min=0.0,
        valid_max=0.1,
        math_cond="> 0.0",
        display_name="weight decay",
        description="The weight decay coefficient.",
        automl_enabled="TRUE"
    )
    lr_scheduler: str = STR_FIELD(
        value="MultiStep",
        description=(
            "The learning scheduler: "
            "* MultiStep : Decrease the lr by lr_decay from lr_steps "
            "* StepLR : Decrease the lr by lr_decay at every lr_step_size."
        ),
        display_name="learning rate scheduler",
        valid_options=",".join(
            ["MultiStep", "StepLR"]
        )
    )
    lr_steps: List[int] = LIST_FIELD(
        arrList=[40],
        description=(
            "The steps at which the learning rate must be decreased. "
            "This is applicable only with the MultiStep LR."
        ),
        display_name="learning rate decay steps",
        value_type="list_2",
    )
    lr_step_size: int = INT_FIELD(
        value=40,
        valid_min=1,
        valid_max=10000,
        math_cond="> 0",
        display_name="learning rate step size",
        description="The number of steps to decrease the learning rate in the StepLR.",
        automl_enabled="TRUE"
    )
    lr_decay: float = FLOAT_FIELD(
        value=0.1,
        valid_min=0.0,
        valid_max=1.0,
        math_cond="> 0.0",
        display_name="learning rate decay",
        description="The decreasing factor for the learning rate scheduler.",
        automl_enabled="TRUE"
    )


@dataclass
class DDTrainExpConfig(TrainConfig):
    """Train experiment config."""

    freeze: Optional[List[str]] = LIST_FIELD(
        arrList=[],
        description="""
        List of layer names to freeze.
        Example: ["backbone", "transformer.encoder", "input_proj"].""",
        display_name="freeze"
    )
    pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_value='',
        description="Path to a pre-trained Deformable DETR model to initialize the current training from."
    )
    clip_grad_norm: float = FLOAT_FIELD(
        value=0.1,
        math_cond="> 0.0",
        display_name="clip gradient norm",
        description=(
            "Amount to clip the gradient by L2 Norm. "
            "A value of 0.0 specifies no clipping."
        ),
    )
    is_dry_run: bool = BOOL_FIELD(
        value=False,
        display_name="Is dry run",
        description=(
            "Whether to run the trainer in Dry Run mode. This serves "
            "as a good means to validate the spec file and run a sanity check on the trainer "
            "without actually initializing and running the trainer."
        ),
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
        valid_options=",".join(
            ["ddp", "fsdp"]
        ),
        display_name="distributed_strategy",
        description=(
            "The multi-GPU training strategy. "
            "DDP (Distributed Data Parallel) and Fully Sharded DDP are supported."
        ),
    )
    activation_checkpoint: bool = BOOL_FIELD(
        value=True,
        display_name="enable activation checkpointing",
        description=(
            "A True value instructs train to recompute in backward pass to save GPU memory, "
            "rather than storing activations."
        ),
    )
    verbose: bool = BOOL_FIELD(
        value=False,
        display_name="enable verbose logs",
        description=(
            "Flag to enable printing of detailed learning rate scaling from the optimizer."
        )
    )
    input_width: Optional[int] = INT_FIELD(
        value=None,
        description="Width of the input image tensor.",
        display_name="input width",
        valid_min=1,
    )
    input_height: Optional[int] = INT_FIELD(
        value=None,
        description="Height of the input image tensor.",
        display_name="input height",
        valid_min=1,
    )
