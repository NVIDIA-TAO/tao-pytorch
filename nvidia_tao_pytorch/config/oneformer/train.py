# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        value=0.00001,
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
        value="Warmuppoly",
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
        description="learning rate decay epochs.",
        display_name="learning rate decay epochs."
    )
    gamma: float = FLOAT_FIELD(
        value=0.1,
        math_cond="> 0.0",
        display_name="gamma",
        description="Multiplicative factor of learning rate decay.",
    )
    warmup_iters: int = INT_FIELD(
        value=1000,
        display_name="warmup iters",
        description="Number of iterations to warmup.",
    )
    warmup_factor: float = FLOAT_FIELD(
        value=0.001,
        display_name="warmup factor",
        description="Factor to warmup the learning rate.",
    )
    max_iter: int = INT_FIELD(
        value=368750,
        display_name="max iter",
        description="Number of iterations to train for.",
    )
    steps: List[int] = LIST_FIELD(
        arrList=[327778, 355092],
        description="learning rate decay epochs.",
        display_name="learning rate decay epochs."
    )


@dataclass
class ClipGradConfig:
    """Clip gradient config."""

    enabled: bool = BOOL_FIELD(
        value=True,
        display_name="enable clip gradient",
        description="Enable gradient clipping.",
    )
    clip_type: str = STR_FIELD(
        value="full_model",
        display_name="clip gradient type",
        description="Gradient clip type.",
    )
    clip_value: float = FLOAT_FIELD(
        value=1.0,
        display_name="clip gradient value",
        description="Gradient clip value.",
    )
    norm_type: float = FLOAT_FIELD(
        value=2.0,
        display_name="clip gradient norm type",
        description="Gradient clip norm type.",
    )


@dataclass
class OneFormerTrainExpConfig(TrainConfig):
    """Train experiment config."""

    num_epochs: int = INT_FIELD(
        value=50,
        display_name="number of epochs",
        description="Number of epochs to train for.",
    )
    num_gpus: int = INT_FIELD(
        value=8,
        display_name="number of gpus",
        description="Number of GPUs to train on.",
    )
    num_nodes: int = INT_FIELD(
        value=1,
        display_name="number of nodes",
        description="Number of nodes to train on.",
    )
    seed: int = INT_FIELD(
        value=123,
        display_name="seed",
        description="Seed for reproducibility.",
    )
    resume_training_checkpoint_path: Optional[str] = STR_FIELD(
        value=None,
        default_type=None,
        description="Path to a pre-trained OneFormer model to initialize the current training from."
    )
    freeze: Optional[List[str]] = LIST_FIELD(
        arrList=[],
        description="""
        List of layer names to freeze.
        Example: ["backbone", "transformer.encoder", "input_proj"].""",
        display_name="freeze"
    )
    pretrained_model: Optional[str] = STR_FIELD(
        value=None,
        default_type=None,
        description="Path to a pre-trained OneFormer model to initialize the current training from."
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
    verbose: bool = BOOL_FIELD(
        value=False,
        display_name="enable verbose logs",
        description="""
        Flag to enable printing of detailed learning rate scaling from the optimizer.
        """
    )
    iters_per_epoch: Optional[int] = INT_FIELD(
        value=None,  # 20210, 118272
        display_name="iteration per epoch",
        description="Number of iteration per epoch.",
    )
    accumulate_grad_batches: int = INT_FIELD(
        value=1,
        display_name="accumulate grad batches",
        description="Number of batches to accumulate gradients over.",
    )
    validation_interval: int = INT_FIELD(
        value=1,
        display_name="validation interval",
        description="Number of epochs to validate.",
    )
    val_check_interval: float = FLOAT_FIELD(
        value=1.0,
        math_cond="> 0.0",
        display_name="val check interval",
        description=(
            "How often to run validation within a training epoch. "
            "A float in (0.0, 1.0] means a fraction of the epoch (e.g. 0.1 = every 10%). "
            "Overrides validation_interval when set to a value < 1.0."
        ),
    )
    pretrained_backbone: Optional[str] = STR_FIELD(
        value=None,
        default_type=None,
        description="Path to a pre-trained backbone to initialize the current training from."
    )
    clip_gradients: ClipGradConfig = DATACLASS_FIELD(
        ClipGradConfig(),
        display_name="clip gradients",
        description="Hyper parameters to configure the gradient clipping."
    )
    checkpoint_interval: int = INT_FIELD(
        value=1,
        display_name="checkpoint interval",
        description="Number of epochs (or steps when checkpoint_interval_unit='step') between checkpoints.",
    )
    checkpoint_interval_unit: str = STR_FIELD(
        value="epoch",
        display_name="checkpoint interval unit",
        description=(
            "Unit for checkpoint_interval. "
            "'epoch' saves every N epochs (default). "
            "'step' saves every N training steps, useful for sub-epoch checkpointing "
            "(e.g. checkpoint_interval=448 with unit='step' saves every ~0.1 epoch when epoch=4480 steps)."
        ),
    )
