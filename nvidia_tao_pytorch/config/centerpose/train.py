# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the trainer."""

from typing import List
from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    BOOL_FIELD,
    FLOAT_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
)
from nvidia_tao_pytorch.config.common.common_config import TrainConfig


@dataclass
class OptimConfig:
    """Optimizer config."""

    lr: float = FLOAT_FIELD(
        value=6e-05,
        default_value=6e-05,
        automl_enabled="True",
        valid_min=0,
        valid_max="inf",
        description="Learning rate.",
        display_name="Learning Rate",
        popular="6e-05"
    )
    lr_scheduler: str = STR_FIELD(
        value="MultiStep",
        description="Learning rate scheduler.",
        display_name="Learning Rate Scheduler",
        popular="MultiStep"
    )
    lr_steps: List[int] = LIST_FIELD(
        arrList=[90, 120],
        description="Learning rate steps.",
        display_name="Learning Rate Steps",
        popular="[90, 120]"
    )
    lr_decay: float = FLOAT_FIELD(
        value=0.1,
        automl_enabled="True",
        default_value=0.1,
        valid_min=0,
        valid_max="inf",
        description="Learning rate decay.",
        display_name="Learning Rate Decay",
        popular="0.1"
    )


@dataclass
class CenterPoseLossConfig:
    """CenterPose loss config."""

    mse_loss: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Use mean squared error loss.",
        display_name="Mean Squared Error Loss"
    )
    dense_hp: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Use dense heatmaps.",
        display_name="Dense Heatmaps"
    )
    reg_loss: str = STR_FIELD(
        value="l1",
        description="Regression loss function.",
        display_name="Regression Loss Function",
        popular="l1"
    )
    num_stacks: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max=1,
        description="Number of stacks.",
        display_name="Number of Stacks",
        popular="1"
    )
    hps_uncertainty: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Use heatmaps uncertainty loss.",
        display_name="Heatmaps Uncertainty"
    )
    wh_weight: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        valid_min=0.1,
        valid_max=0.1,
        description="Weight for width and height loss.",
        display_name="Width and Height Weight",
        popular="0.1"
    )
    reg_bbox: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Use bounding box regression loss.",
        display_name="Bounding Box Regression"
    )
    reg_offset: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Use offset regression loss.",
        display_name="Offset Regression"
    )
    reg_hp_offset: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Use offset regression loss for keypoints.",
        display_name="Offset Regression for Keypoints"
    )
    obj_scale: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Use object scale loss.",
        display_name="Object Scale"
    )
    obj_scale_weight: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max=1,
        description="Weight for object scale loss.",
        display_name="Object Scale Weight",
        popular="1"
    )
    obj_scale_uncertainty: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Use object scale uncertainty loss.",
        display_name="Object Scale Uncertainty"
    )
    use_residual: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Use residual loss.",
        display_name="Residual Loss"
    )
    dimension_ref: str = STR_FIELD(
        value="",
        description="Dimension reference.",
        display_name="Dimension Reference"
    )
    off_weight: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max=1,
        description="Weight for offset loss.",
        display_name="Offset Weight",
        popular="1"
    )
    hm_hp: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Use heatmaps for keypoints.",
        display_name="Heatmaps for Keypoints"
    )
    hm_hp_weight: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max=1,
        description="Weight for heatmaps for keypoints.",
        display_name="Heatmaps for Keypoints Weight",
        popular="1"
    )
    hm_weight: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max=1,
        description="Weight for heatmaps.",
        display_name="Heatmaps Weight",
        popular="1"
    )
    hp_weight: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max=1,
        description="Weight for keypoints.",
        display_name="Keypoints Weight",
        popular="1"
    )


@dataclass
class CenterPoseTrainExpConfig(TrainConfig):
    """Train experiment config."""

    pretrained_model_path: str = STR_FIELD(
        value="",
        description="Path to the pretrained model.",
        display_name="Pretrained Model"
    )
    clip_grad_val: float = FLOAT_FIELD(
        value=100.0,
        default_value=100.0,
        valid_min=1.0,
        valid_max="inf",
        description="Gradient clipping value.",
        display_name="Gradient Clipping Value",
        popular="100.0"
    )
    is_dry_run: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Run a training iteration without model saving.",
        display_name="Dry Run"
    )
    precision: str = STR_FIELD(
        value="fp32",
        default_value="fp32",
        description="Training precision.",
        display_name="Precision",
        popular="fp32"
    )
    optim: OptimConfig = DATACLASS_FIELD(
        OptimConfig(),
        description="Model optimizer configuration.",
        display_name="Optimizer Config",
    )
    loss_config: CenterPoseLossConfig = DATACLASS_FIELD(
        CenterPoseLossConfig(),
        description="Model loss configuration.",
        display_name="Loss Config",
    )
