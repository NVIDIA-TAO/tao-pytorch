# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the CoDETR model."""

from dataclasses import dataclass
from typing import List, Optional

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
)

swin_model_list = [
    'swin_large_224_22k',
    'swin_large_384_22k',
    'swin_large_patch4_window7_224',
    'swin_large_patch4_window12_384',
    'swin_base_224_22k',
    'swin_base_384_22k',
    'swin_tiny_224_1k',
]

fan_model_list = [
    'fan_tiny',
    'fan_small',
    'fan_base',
    'fan_large',
]

vit_model_list = [
    'vit_large_nvdinov2',
    'vit_large_dinov2',
]

SUPPORTED_BACKBONES = [
    *swin_model_list,
    *fan_model_list,
    *vit_model_list,
    *["resnet_34", "resnet_50"],
    *['efficientvit_b0', 'efficientvit_b1', 'efficientvit_b2', 'efficientvit_b3'],
]


@dataclass
class CoDETRModelConfig:
    """CoDETR model config."""

    pretrained_backbone_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="pretrained backbone path",
        description="[Optional] Path to a pretrained backbone file.",
    )
    backbone: str = STR_FIELD(
        value="swin_large_patch4_window7_224",
        default_value="swin_large_patch4_window7_224",
        display_name="backbone",
        description="The backbone name. CoDETR defaults to Swin-L.",
        valid_options=",".join(SUPPORTED_BACKBONES)
    )
    num_queries: int = INT_FIELD(
        value=900,
        default_value=900,
        description="Number of object queries (detection slots).",
        display_name="number of queries",
        valid_min=1,
        valid_max=2000,
    )
    num_feature_levels: int = INT_FIELD(
        value=4,
        default_value=4,
        description="Number of feature levels for multi-scale deformable attention.",
        display_name="number of feature levels",
        valid_min=1,
        valid_max=5,
    )
    hidden_dim: int = INT_FIELD(
        value=256,
        default_value=256,
        description="Dimension of the transformer hidden units.",
        display_name="hidden dim",
    )
    nheads: int = INT_FIELD(
        value=8,
        default_value=8,
        description="Number of attention heads.",
        display_name="nheads",
    )
    enc_layers: int = INT_FIELD(
        value=6,
        default_value=6,
        description="Number of transformer encoder layers.",
        display_name="encoder layers",
        valid_min=1,
    )
    dec_layers: int = INT_FIELD(
        value=6,
        default_value=6,
        description="Number of transformer decoder layers.",
        display_name="decoder layers",
        valid_min=1,
    )
    dim_feedforward: int = INT_FIELD(
        value=2048,
        description="Dimension of the transformer FFN.",
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
    dropout_ratio: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        description="Dropout probability.",
        display_name="dropout ratio",
        valid_min=0.0,
        valid_max=1.0,
    )
    aux_loss: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="auxiliary decoder losses",
        description="Whether to apply auxiliary losses at each decoder layer.",
    )
    dilation: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="dilation",
        description="Enable dilation in backbone (ResNet only).",
    )
    train_backbone: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="train backbone",
        description="Whether to train the backbone weights.",
    )
    pre_norm: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Add LayerNorm before the encoder.",
        display_name="pre norm",
    )
    two_stage_type: str = STR_FIELD(
        value="standard",
        default_value="standard",
        valid_options=",".join(["standard", "no"]),
        description="Two-stage detection type.",
        display_name="two stage type",
    )
    decoder_sa_type: str = STR_FIELD(
        value="sa",
        default_value="sa",
        valid_options=",".join(['sa', 'ca_label', 'ca_content']),
        description="Type of decoder self-attention.",
        display_name="decoder sa type",
    )
    embed_init_tgt: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Add target embedding.",
        display_name="embed init target",
    )
    fix_refpoints_hw: int = INT_FIELD(
        value=-1,
        default_value=-1,
        valid_min=-2,
        valid_max=1000,
        description="-1: learn w/h per box; -2: share w/h; >0: fixed.",
        display_name="fix refpoints hw",
    )
    pe_temperatureH: int = INT_FIELD(
        value=20,
        default_value=20,
        description="Height positional embedding temperature.",
        display_name="pe temperatureH",
        valid_min=1,
    )
    pe_temperatureW: int = INT_FIELD(
        value=20,
        default_value=20,
        description="Width positional embedding temperature.",
        display_name="pe temperatureW",
        valid_min=1,
    )
    return_interm_indices: List[int] = LIST_FIELD(
        arrList=[1, 2, 3, 4],
        description="Feature level indices to use. Length must match num_feature_levels.",
        display_name="return interm indices",
    )
    # DN (denoising) params
    use_dn: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Enable contrastive denoising training.",
        display_name="use denoising",
    )
    dn_number: int = INT_FIELD(
        value=100,
        default_value=100,
        description="Number of denoising queries.",
        display_name="denoising number",
        valid_min=1,
    )
    dn_box_noise_scale: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        description="Scale of noise applied to boxes during denoising.",
        display_name="dn box noise scale",
        valid_min=0.0,
    )
    dn_label_noise_ratio: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="Scale of noise applied to labels during denoising.",
        display_name="dn label noise ratio",
        valid_min=0.0,
    )
    # DETR head loss coefficients
    cls_loss_coef: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        description="Classification loss coefficient.",
        display_name="cls loss coef",
        valid_min=0.0,
    )
    bbox_loss_coef: float = FLOAT_FIELD(
        value=5.0,
        default_value=5.0,
        description="BBox L1 loss coefficient.",
        display_name="bbox loss coef",
        valid_min=0.0,
    )
    giou_loss_coef: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        description="GIoU loss coefficient.",
        display_name="giou loss coef",
        valid_min=0.0,
    )
    focal_alpha: float = FLOAT_FIELD(
        value=0.25,
        description="Alpha value in focal loss.",
        display_name="focal alpha",
    )
    num_select: int = INT_FIELD(
        value=300,
        default_value=300,
        description="Number of top-K predictions selected during post-processing.",
        display_name="num select",
        valid_min=1,
    )
    interm_loss_coef: float = FLOAT_FIELD(
        value=1.0,
        display_name="intermediate loss coefficient",
        description="Coefficient for intermediate (encoder) outputs loss.",
    )
    no_interm_box_loss: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Disable intermediate bbox loss.",
        display_name="no interm bbox loss",
    )
    loss_types: List[str] = LIST_FIELD(
        arrList=['labels', 'boxes'],
        description="Loss types to apply.",
        display_name="loss types",
    )
    backbone_names: List[str] = LIST_FIELD(
        arrList=["backbone.0"],
        description="Prefix of tensor names corresponding to the backbone.",
        display_name="backbone tensor name prefix",
    )
    linear_proj_names: List[str] = LIST_FIELD(
        arrList=['reference_points', 'sampling_offsets'],
        display_name="linear projection names",
        description="Linear projection layer names.",
    )
    # CoDETR-specific: collaborative auxiliary head
    num_co_heads: int = INT_FIELD(
        value=1,
        default_value=1,
        description="Number of collaborative auxiliary ATSS heads.",
        display_name="num co heads",
        valid_min=1,
        valid_max=3,
    )
    co_head_loss_weight: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        description="Loss weight for collaborative auxiliary head losses.",
        display_name="co head loss weight",
        valid_min=0.0,
    )
    co_head_num_convs: int = INT_FIELD(
        value=4,
        default_value=4,
        description="Number of conv layers in the collaborative ATSS head towers.",
        display_name="co head num convs",
        valid_min=1,
    )
    # Post-processing NMS
    soft_nms_enabled: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Apply per-class soft-NMS after top-K selection.",
        display_name="soft nms enabled",
    )
    soft_nms_method: str = STR_FIELD(
        value="linear",
        default_value="linear",
        description="Soft-NMS method: 'linear' or 'gaussian'.",
        display_name="soft nms method",
    )
    soft_nms_iou_threshold: float = FLOAT_FIELD(
        value=0.8,
        default_value=0.8,
        description="IoU threshold for linear soft-NMS; "
                    "boxes with IoU <= threshold are not suppressed.",
        display_name="soft nms iou threshold",
        valid_min=0.0,
        valid_max=1.0,
    )
    soft_nms_sigma: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="Gaussian sigma for soft-NMS score decay (gaussian method only).",
        display_name="soft nms sigma",
        valid_min=0.01,
    )
    # Distillation compatibility (unused but keeps config shape consistent with DINO)
    distillation_loss_coef: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        display_name="distillation loss coefficient",
        description="Coefficient for distillation loss.",
        valid_min=0.0,
    )
