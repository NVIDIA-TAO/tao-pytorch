# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        value="train_loss",
        description="The metric value to be monitored for the :code:`AutoReduce` Scheduler.",
        display_name="monitor_name",
        valid_options=",".join(
            ["val_loss", "train_loss"]
        )
    )
    lr: float = FLOAT_FIELD(
        value=2e-4,
        valid_min=0.0,
        valid_max=1.0,
        math_cond="> 0.0",
        display_name="learning rate",
        description="The initial learning rate for training the model.",
        automl_enabled="TRUE"
    )
    backbone_multiplier: float = FLOAT_FIELD(
        value=0.1,
        valid_min=0.0,
        valid_max=1.0,
        math_cond="> 0.0",
        display_name="backbone learning rate multiplier",
        description="A multiplier for backbone learning rate.",
        automl_enabled="TRUE",
        popular="yes",
    )
    momentum: float = FLOAT_FIELD(
        value=0.9,
        valid_min=0.0,
        valid_max=1.0,
        math_cond="> 0.0",
        display_name="momentum - AdamW",
        description="The momentum for the AdamW optimizer.",
        automl_enabled="TRUE",
        popular="yes",
    )
    weight_decay: float = FLOAT_FIELD(
        value=0.05,
        valid_min=0.0,
        valid_max=1.0,
        math_cond="> 0.0",
        display_name="weight decay",
        description="The weight decay coefficient.",
        automl_enabled="TRUE",
        popular="yes",
    )
    lr_scheduler: str = STR_FIELD(
        value="MultiStep",
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
        description="""learning rate decay epochs.""",
        display_name="learning rate decay epochs."
    )
    gamma: float = FLOAT_FIELD(
        value=0.1,
        math_cond="> 0.0",
        display_name="gamma",
        description="Multiplicative factor of learning rate decay.",
    )


@dataclass
class Mask2FormerTrainExpConfig(TrainConfig):
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
        default_type=None,
        description="Path to a pre-trained Mask2Former model to initialize the current training from."
    )
    clip_grad_norm: float = FLOAT_FIELD(
        value=0.1,
        math_cond="> 0.0",
        display_name="clip gradient norm",
        description="""
        Amount to clip the gradient by L2 Norm.
        A value of 0.0 specifies no clipping.""",
    )
    clip_grad_norm_type: Union[float, str] = 2.0
    clip_grad_type: str = STR_FIELD(
        value='full',
        default_type='full',
        display_name='clip gradient type',
        description="Gradient clip type."
    )
    is_dry_run: bool = BOOL_FIELD(
        value=False,
        display_name="Is dry run",
        description="""
        Whether to run the trainer in Dry Run mode. This serves
        as a good means to validate the spec file and run a sanity check on the trainer
        without actually initializing and running the trainer.""",
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
        description="""
        The multi-GPU training strategy.
        DDP (Distributed Data Parallel) and Fully Sharded DDP are supported.""",
    )
    activation_checkpoint: bool = BOOL_FIELD(
        value=False,
        display_name="enable activation checkpointing",
        description="""
        A True value instructs train to recompute in backward pass to save GPU memory,
        rather than storing activations. Note: activation checkpointing is incompatible
        with find_unused_parameters in DDP and may cause errors if the model has unused
        parameters.""",
    )
    use_distributed_sampler: bool = BOOL_FIELD(
        value=False,
        display_name="use distributed sampler",
        description="Use distributed sampler for multi-GPU training.",
    )
    iters_per_epoch: Optional[int] = INT_FIELD(
        value=None,  # 20210, 118272
        display_name="iteration per epoch",
        description="Number of iteration per epoch.",
    )
