# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the model."""

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

# @seanf: Is there a better way to do this? Importing is a good idea but it's now separated

# from nvidia_tao_pytorch.cv.rtdetr.model.backbone.resnet import resnet_model_dict
resnet_model_list = [
    'resnet_18',
    'resnet_34',
    'resnet_50',
    'resnet_101',
]

# from nvidia_tao_pytorch.cv.rtdetr.model.backbone.convnext import convnext_model_dict
convnext_model_list = [
    'convnext_tiny',
    'convnext_small',
    'convnext_base',
    'convnext_large',
    'convnext_xlarge',
]

# from nvidia_tao_pytorch.cv.rtdetr.model.backbone.fan import fan_model_dict
fan_model_list = [
    'fan_tiny',
    'fan_small',
    'fan_base',
    'fan_large',
]

# from nvidia_tao_pytorch.cv.rtdetr.model.backbone.efficientvit import efficientvit_model_dict
efficientvit_model_list = [
    "efficientvit_b0",
    "efficientvit_b1",
    "efficientvit_b2",
    "efficientvit_b3",
    "efficientvit_l0",
    "efficientvit_l1",
    "efficientvit_l2",
    "efficientvit_l3",
]

SUPPORTED_BACKBONES = [
    # *list(resnet_model_dict.keys()),
    # *list(convnext_model_dict.keys()),
    # *list(fan_model_dict.keys()),
    # *list(efficientvit_model_dict.keys()),
    *resnet_model_list,
    *convnext_model_list,
    *fan_model_list,
    *efficientvit_model_list
]


@dataclass
class FrozenFMConfig:
    """RT-DETR Frozen FM config."""

    enabled: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Enable frozen FM",
        description="""Flag to enable frozen foundation model to be added to RT-DETR.""",
    )
    backbone: str = STR_FIELD(
        value="radio_v2-l",
        default_value="radio_v2-l",
        valid_options=",".join(["radio_v2-b", "radio_v2-l", "radio_v2-h"]),
        display_name="Name of the frozen foundation model",
        description="Name of the frozen foundation model.",
    )
    checkpoint: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="Pretrained foundation model path or name",
        description="Path to a pretrained foundation model.",
    )


@dataclass
class RTModelConfig:
    """RT-DETR model config."""

    pretrained_backbone_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="pretrained backbone path",
        description="[Optional] Path to a pretrained backbone file.",
    )
    backbone: str = STR_FIELD(
        value="resnet_50",
        default_value="resnet_50",
        display_name="backbone",
        description="""The backbone name of the model.
                    TAO implementation of RT-DETR support ResNet, EfficientViT, FAN, and ConvNext.""",
        valid_options=",".join(SUPPORTED_BACKBONES)
    )
    train_backbone: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="Train backbone",
        description="""Flag to set backbone weights as trainable or frozen.
                    When set to `False`, the backbone weights will be frozen.""",
    )
    load_teacher_enc_dec: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Load teacher's encoder and decoder weights",
        description="""Flag to load teacher's encoder and decoder weights.""",
    )
    num_queries: int = INT_FIELD(
        value=300,
        default_value=300,
        description="The number of queries",
        display_name="number of queries",
        valid_min=1,
        valid_max="inf",
        automl_enabled="TRUE",
        parent_param="TRUE"
    )
    num_select: int = INT_FIELD(
        value=300,
        default_value=300,
        description=(
            "The number of top-K predictions selected during post-process. "
            "Must be < num_queries * num_classes"
        ),
        display_name="num select",
        valid_min=1,
        valid_max=1000,
        automl_enabled="TRUE",
        depends_on="model.num_queries"
    )
    num_feature_levels: int = INT_FIELD(
        value=3,
        default_value=3,
        description="The number of feature levels to use in the model",
        display_name="number of feature levels",
        valid_min=1,
        valid_max=4,
    )
    return_interm_indices: List[int] = LIST_FIELD(
        arrList=[1, 2, 3],
        description="The index of feature levels to use in the model. The length must match `num_feature_levels`.",
        display_name="return interim indices"
    )

    feat_strides: List[int] = LIST_FIELD(
        arrList=[8, 16, 32],
        description="The stride used as grid size of positional embedding at each encoder layer.",
        display_name="feature strides"
    )
    feat_channels: List[int] = LIST_FIELD(
        arrList=[256, 256, 256],
        description="The feature channel sizes in decoder.",
        display_name="feature channels"
    )
    use_encoder_idx: List[int] = LIST_FIELD(
        arrList=[2],
        description="The index of multi-scale backbone features to pass to encoder.",
        display_name="use encoder index"
    )

    hidden_dim: int = INT_FIELD(
        value=256,
        default_value=256,
        description="Dimension of the hidden units.",
        display_unit="hidden dim",
        automl_enabled="FALSE"
    )
    nheads: int = INT_FIELD(
        value=8,
        default_value=8,
        description="Number of heads",
        display_name="nheads",
    )
    dropout_ratio: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        description="The probability to drop hidden units.",
        display_name="drop out ratio",
        valid_min=0.0,
        valid_max=1.0
    )
    enc_layers: int = INT_FIELD(
        value=1,
        default_value=1,
        description="Numer of encoder layers in the transformer",
        valid_min=1,
        automl_enabled="TRUE",
        display_name="encoder layers",
    )
    dim_feedforward: int = INT_FIELD(
        value=1024,
        description="Dimension of the feedforward network",
        display_name="dim feedforward",
        valid_min=1,
    )

    pe_temperature: int = INT_FIELD(
        value=10000,
        default_value=10000,
        description="The temperature applied to the positional sine embedding.",
        display_name="pe_temperature",
        valid_min=1,
        valid_max="inf"
    )
    expansion: float = INT_FIELD(
        value=1.0,
        default_value=1.0,
        description="The expansion raito for hidden dimesnion used in CSPRepLayer.",
        display_name="expansion",
        valid_min=0.0,
        valid_max="inf"
    )
    depth_mult: int = INT_FIELD(
        value=1,
        default_value=1,
        description="The number of RegVGGBlock used in CSPRepLayer.",
        display_name="expansion",
        valid_min=1,
        valid_max="inf"
    )
    enc_act: str = STR_FIELD(
        value="gelu",
        default_value="gelu",
        display_name="encoder activation",
        description="The activation used for the encoder."
    )
    act: str = STR_FIELD(
        value="silu",
        default_value="silu",
        display_name="activation",
        description="The activation used for top-down FPN and bottom-up PAN."
    )

    dec_layers: int = INT_FIELD(
        value=6,
        default_value=6,
        description="Numer of decoder layers in the transformer",
        valid_min=1,
        automl_enabled="TRUE",
        display_name="decoder layers",
    )
    dn_number: int = INT_FIELD(
        value=100,
        default_value=100,
        description="The number of denoising queries.",
        display_name="denoising number",
        valid_min=0,
        valid_max="inf"
    )
    eval_idx: int = INT_FIELD(
        value=-1,
        default_value=-1,
        description="The index of decoder layer to use for evaluation. By default, use the last decoder layer.",
        display_name="evaluation index",
        valid_min=-1,
        valid_max="inf"
    )
    vfl_loss_coef: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the varifocal error in the loss function.",
        display_name="varifocal loss coefficient",
    )
    bbox_loss_coef: float = FLOAT_FIELD(
        value=5.0,
        default_value=5.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the L1 error of the bounding box coordinates in the loss function.",
        display_name="BBox loss coefficient",
    )
    giou_loss_coef: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the GIoU loss of the bounding box in the loss function.",
        display_name="GIoU loss coefficient",
    )
    class_cost: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the classification error in the matching cost.",
        display_name="Class cost coefficient",
    )
    bbox_cost: float = FLOAT_FIELD(
        value=5.0,
        default_value=5.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the L1 error of the bounding box coordinates in the matching cost.",
        display_name="BBox cost coefficient",
    )
    giou_cost: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the GIoU loss of the bounding box in the matching cost.",
        display_name="GIoU cost coefficient",
    )
    alpha: float = FLOAT_FIELD(
        value=0.75,
        description="The alpha value in the varifocal loss.",
        display_name="alpha",
        math_cond="> 0.0"
    )
    gamma: float = FLOAT_FIELD(
        value=2.0,
        description="The gamma value in the varifocal loss.",
        display_name="gamma",
        math_cond="> 0.0"
    )
    clip_max_norm: float = FLOAT_FIELD(
        value=0.1,
        display_name="clip max norm",
        description="",
    )

    aux_loss: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="Auxiliary Loss",
        description="""A flag specifying whether to use auxiliary
                    decoding losses (loss at each decoder layer)""",
    )

    loss_types: List[str] = LIST_FIELD(
        arrList=['vfl', 'boxes'],
        description="Losses to be used during training",
        display_name="loss_types",
    )

    backbone_names: List[str] = LIST_FIELD(
        arrList=["backbone.0"],
        description="Prefix of the tensor names corresponding to the backbone.",
        display_name="Backbone tensor name prefix"
    )

    linear_proj_names: List[str] = LIST_FIELD(
        arrList=['reference_points', 'sampling_offsets'],
        display_name="linear projection names",
        description="Linear projection layer names."
    )
    # Distillation specific
    distillation_loss_coef: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        display_name="distillation loss coefficient",
        description="The coefficient for the distillation loss during distill.",
        valid_min=0.0,
    )
    frozen_fm: FrozenFMConfig = DATACLASS_FIELD(
        FrozenFMConfig,
        description="Configurable parameters to construct the frozen foundation model.",
    )
