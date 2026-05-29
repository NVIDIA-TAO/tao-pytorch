# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        value="val_loss",  # {val_loss, train_loss}
        description="The metric value to be monitored for the :code:`AutoReduce` Scheduler.",
        display_name="monitor_name",
        valid_options=",".join(
            ["val_loss", "train_loss"]
        )
    )
    lr: float = FLOAT_FIELD(
        value=1e-4,
        valid_min=1e-6,
        valid_max=1e-2,
        math_cond="> 0.0",
        display_name="learning rate",
        description="The initial learning rate for training the model, excluding the backbone.",
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
        valid_min=1e-6,
        valid_max=1e-2,
        math_cond="> 0.0",
        display_name="weight decay",
        description="The weight decay coefficient.",
        automl_enabled="TRUE"
    )
    lr_scheduler: str = STR_FIELD(
        value="MultiStepLR",  # {val_loss, train_loss}
        description="""The learning scheduler:
                    * MultiStepLR : Decrease the lr by lr_decay from lr_steps
                    * StepLR : Decrease the lr by lr_decay at every lr_step_size.""",
        display_name="Learning rate scheduler",
        valid_options=",".join(
            ["MultiStep", "StepLR", "CustomMultiStepLRScheduler",
             "LambdaLR", "PolynomialLR", "OneCycleLR", "CosineAnnealingLR"]
        )
    )
    lr_steps: List[int] = LIST_FIELD(
        arrList=[1000],
        description="""The steps at which the learning rate must be decreased.
                    This is applicable only with the MultiStep LR.""",
        display_name="learning rate decay steps"
    )
    lr_step_size: int = INT_FIELD(
        value=1000,
        valid_min=1,
        valid_max=10000,
        math_cond="> 0",
        display_name="learning rate step size",
        description="""The number of steps to decrease the learning rate in the StepLR.""",
        automl_enabled="TRUE"
    )
    lr_decay: float = FLOAT_FIELD(
        value=0.1,
        valid_min=0.0,
        valid_max=1.0,
        math_cond="> 0.0",
        display_name="learning rate decay",
        description="""The decreasing factor for the learning rate scheduler.""",
        automl_enabled="TRUE"
    )
    min_lr: float = FLOAT_FIELD(
        value=1e-7,
        valid_min=1e-8,
        valid_max=1e-3,
        math_cond="> 0.0",
        display_name="minimum learning rate",
        description="""The minimum learning rate value for the learning rate scheduler.""",
        automl_enabled="TRUE"
    )
    warmup_steps: int = INT_FIELD(
        value=20,
        default_value=20,
        description="""The number of steps to perform linear learning rate" \
                    warm-up before engaging a learning rate scheduler""",
        display_name="Warm up steps",
        valid_min=0,
        valid_max="inf"
    )


@dataclass
class DepthNetTrainExpConfig(TrainConfig):
    """Train experiment config."""

    checkpoint_interval_steps: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        description="The number of steps to save the checkpoint.",
        display_name="checkpoint interval steps"
    )
    pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_value='',
        description="Path to a pre-trained DepthNet model to initialize the current training from."
    )
    clip_grad_norm: float = FLOAT_FIELD(
        value=0.1,
        math_cond="> 0.0",
        display_name="clip gradient norm",
        description="""
        Amount to clip the gradient by L2 Norm.
        A value of 0.0 specifies no clipping.""",
    )
    dataloader_visualize: bool = BOOL_FIELD(
        value=False,
        display_name="dataloader visualize",
        description="Whether to visualize the dataloader.",
    )
    vis_step_interval: int = INT_FIELD(
        value=10,
        display_name="visualization interval",
        description="The visualization interval in step.",
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
            "bf16", "fp32", "fp16"
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
        rather than storing activations.""",
    )
    inference_tile: bool = BOOL_FIELD(
        value=False,
        display_name="tile inference",
        description="""Use tiled inference, particularly for transformers
                    which expect fixed size of sequences.
                    """
    )
    tile_wtype: str = STR_FIELD(
        value="gaussian",
        display_name="tile weight type",
        description="Use tiled inference weight type"
    )
    tile_min_overlap: List[int] = LIST_FIELD(
        arrList=[16, 16],
        display_name="tile weight type",
        description="Use tiled inference weight type"
    )
    log_every_n_steps: int = INT_FIELD(
        value=500,
        display_name='log steps',
        description="""
        Interval steps of logging training results and running validation numbers within 1 epoch"""
    )
