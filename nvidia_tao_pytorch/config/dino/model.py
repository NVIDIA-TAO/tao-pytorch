# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the model."""

from dataclasses import dataclass
from typing import List, Optional

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
)

# @seanf: Is there a better way to do this? Importing is a good idea but it's now separated

# from nvidia_tao_pytorch.cv.deformable_detr.model.gc_vit import gc_vit_model_dict
gc_vit_model_list = [
    'gc_vit_xxtiny',
    'gc_vit_xtiny',
    'gc_vit_tiny',
    'gc_vit_small',
    'gc_vit_base',
    'gc_vit_large',
    'gc_vit_large_384',
]

# from nvidia_tao_pytorch.cv.dino.model.fan import fan_model_dict
fan_model_list = [
    'fan_tiny',
    'fan_small',
    'fan_base',
    'fan_large'
]

# from nvidia_tao_pytorch.cv.dino.model.vision_transformer.vit_adapter import vit_model_dict
vit_model_list = [
    'vit_large_nvdinov2',
    'vit_large_dinov2',
]

# from nvidia_tao_pytorch.cv.grounding_dino.model.swin_transformer import swin_model_dict
swin_model_list = [
    'swin_tiny_224_1k',
    'swin_base_224_22k',
    'swin_base_384_22k',
    'swin_large_224_22k',
    'swin_large_384_22k',
]

SUPPORTED_BACKBONES = [
    # *list(fan_model_dict.keys()),
    # *list(gc_vit_model_dict.keys()),
    # *list(vit_model_dict.keys()),
    # *list(swin_model_dict.keys()),
    *fan_model_list,
    *gc_vit_model_list,
    *vit_model_list,
    *swin_model_list,
    *["resnet_34", "resnet_50"],
    *['efficientvit_b0', 'efficientvit_b1', 'efficientvit_b2', 'efficientvit_b3']
]


@dataclass
class DINOModelConfig:
    """DINO model config."""

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
                    TAO implementation of DINO support GCViT, FAN, ResNet50
                    and NVDINOv2.""",
        valid_options=",".join(SUPPORTED_BACKBONES)
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
    num_feature_levels: int = INT_FIELD(
        value=4,
        default_value=4,
        description="The number of feature levels to use in the model",
        display_name="number of feature levels",
        valid_min=1,
        valid_max=5,
    )
    cls_loss_coef: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the classification error in the matching cost.",
        display_name="Class loss coefficient",
    )
    bbox_loss_coef: float = FLOAT_FIELD(
        value=5.0,
        default_value=5.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the L1 error of the bounding box coordinates in the matching cost.",
        display_name="BBox loss coefficient",
    )
    giou_loss_coef: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the GIoU loss of the bounding box in the matching cost.",
        display_name="GIoU loss coefficient",
    )

    # DINO training specific
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
    interm_loss_coef: float = FLOAT_FIELD(
        value=1.0,
        display_name="intermediate loss coefficient",
    )
    no_interm_box_loss: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="No intermediate bbox loss.",
        display_name="no interm bbox loss"
    )

    # DINO model arch specific
    pre_norm: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to add layer norm in the encoder or not.",
        display_name="Pre norm"
    )
    two_stage_type: str = STR_FIELD(
        value="standard",
        default_value="standard",
        valid_options=",".join([
            "standard",
            "no"
        ]),
        description="Type of two stage in DINO",
        display_name="two stage type"
    )
    decoder_sa_type: str = STR_FIELD(
        value="sa",
        default_value="sa",
        description="Type of decoder self attention.",
        valid_options=",".join(['sa', 'ca_label', 'ca_content']),
        display_name="decoder self-attention type"
    )
    embed_init_tgt: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to add target embedding",
        display_name="embed init target"
    )
    fix_refpoints_hw: int = INT_FIELD(
        value=-1,
        default_value=-1,
        valid_min=-2,
        valid_max="inf",
        description="""If this value is -1, width and height are learned seperately for each box.
                    If this value is -2, a shared width and height are learned.
                    A value greater than 0 specifies learning with a fixed number.""",
        math_cond="!= 0",
        display_name="fix refpoints hw"
    )
    pe_temperatureH: int = INT_FIELD(
        value=20,
        default_value=20,
        description="The temperature applied to the height dimension of the positional sine embedding.",
        display_name="pe_temperatureH",
        valid_min=1,
        valid_max="inf"
    )
    pe_temperatureW: int = INT_FIELD(
        value=20,
        default_value=20,
        description="The temperature applied to the width dimension of the positional sine embedding.",
        display_name="pe_temperatureW",
        valid_min=1,
        valid_max="inf"
    )
    return_interm_indices: List[int] = LIST_FIELD(
        arrList=[1, 2, 3, 4],
        description="The index of feature levels to use in the model. The length must match `num_feature_levels`.",
        display_name="return interim indices"
    )

    # for DN
    use_dn: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="A flag specifying whether to enbable contrastive de-noising training in DINO",
        display_name="use denoising",
    )
    dn_number: int = INT_FIELD(
        value=100,
        default_value=100,
        description="The number of denoising queries in DINO.",
        display_name="denoising number",
        valid_min=1,
        valid_max="inf"
    )
    dn_box_noise_scale: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        description=(
            "The scale of noise applied to boxes during contrastive de-noising. "
            "If this value is 0, noise is not applied."
        ),
        display_name="Denoised boxes noise scaling",
        valid_min=0.0,
        valid_max="inf",
    )
    dn_label_noise_ratio: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="""The scale of the noise applied to labels during
                       contrastive denoising. If this value is 0, then noise is
                       no applied.""",
        display_name="denoise label noise ratio",
        valid_min=0.0
    )

    focal_alpha: float = FLOAT_FIELD(
        value=0.25,
        description="The alpha value in the focal loss.",
        display_name="focal alpha",
        math_cond="> 0.0"
    )
    clip_max_norm: float = FLOAT_FIELD(
        value=0.1,
        display_name="clip max norm",
        description="",
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
    hidden_dim: int = INT_FIELD(
        value=256,
        default_value=256,
        description="Dimension of the hidden units.",
        display_unit="hidden dim",
        automl_enabled="FALSE"
    )
    enc_layers: int = INT_FIELD(
        value=6,
        default_value=6,
        description="Numer of encoder layers in the transformer",
        valid_min=1,
        automl_enabled="TRUE",
        display_name="encoder layers",
    )
    dec_layers: int = INT_FIELD(
        value=6,
        default_value=6,
        description="Numer of decoder layers in the transformer",
        valid_min=1,
        automl_enabled="TRUE",
        display_name="decoder layers",
    )
    dim_feedforward: int = INT_FIELD(
        value=2048,
        description="Dimension of the feedforward network",
        display_name="dim feedforward",
        valid_min=1,
    )
    dec_n_points: int = INT_FIELD(
        value=4,
        display_name="decoder n points",
        description="Number of reference points in the decoder.",
        valid_min=1,
    )
    enc_n_points: int = INT_FIELD(
        value=4,
        display_name="encoder n points",
        description="Number of reference points in the encoder.",
        valid_min=1,
    )
    aux_loss: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="Train backbone",
        description="""A flag specifying whether to use auxiliary
                    decoding losses (loss at each decoder layer)""",
    )
    dilation: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Dilation enabled.",
        description="""A flag specifying whether enable dilation or not in the backbone.""",
    )
    train_backbone: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="Train backbone",
        description="""Flag to set backbone weights as trainable or frozen.
                    When set to `False`, the backbone weights will be frozen.""",
    )
    loss_types: List[str] = LIST_FIELD(
        arrList=['labels', 'boxes'],
        description="Losses to be used during training",
        display_name="loss_types",
    )
    backbone_names: List[str] = LIST_FIELD(
        arrList=["backbone.0"],
        description="Prefix of the tensor names corresponding to the backbone.",
        display_name="Backbone tensor name prefix")
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
