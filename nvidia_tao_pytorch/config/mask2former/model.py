# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the model."""

from typing import Optional, List
from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
    DATACLASS_FIELD
)


@dataclass
class SemanticSegmentationHead:
    """Semantic Segmentation Head config."""

    common_stride: int = INT_FIELD(
        value=4,
        default_value=4,
        description="Common stride.",
        display_name="Common stride",
        valid_min=2,
    )
    transformer_enc_layers: int = INT_FIELD(
        value=6,
        default_value=6,
        description="Number of transformer encoder layers.",
        display_name="Number of transformer encoder layers.",
        valid_min=1,
        popular="yes",
    )
    convs_dim: int = INT_FIELD(
        value=256,
        default_value=256,
        description="Convolutional layer dimension.",
        display_name="conv layer dim.",
        valid_min=1,
        popular="yes",
    )
    mask_dim: int = INT_FIELD(
        value=256,
        default_value=256,
        description="Mask head dimension.",
        display_name="mask head dim.",
        valid_min=1,
        popular="yes",
    )
    deformable_transformer_encoder_in_features: List[str] = LIST_FIELD(
        arrList=["res3", "res4", "res5"],
        default_value=["res3", "res4", "res5"],
        description="List of feature names for deformable transformer encoder input.",
        display_name="transformer encoder in_features"
    )
    num_classes: int = INT_FIELD(
        value=200,
        default_value=200,
        description="Number of classes.",
        display_name="number of classes.",
        valid_min=1,
    )
    norm: str = STR_FIELD(
        value="GN",
        description="""Norm layer type.""",
        display_name="norm type"
    )


@dataclass
class MaskFormer:
    """MaskFormer config."""

    dropout: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        description="The probability to drop out.",
        display_name="drop out ratio",
        valid_min=0.0,
        valid_max=1.0
    )
    nheads: int = INT_FIELD(
        value=8,
        default_value=8,
        description="Number of heads",
        display_name="nheads",
        popular="yes",
    )
    num_object_queries: int = INT_FIELD(
        value=100,
        default_value=100,
        description="The number of queries",
        display_name="number of queries",
        valid_min=1,
        valid_max="inf",
        automl_enabled="True",
        popular="yes",
    )
    hidden_dim: int = INT_FIELD(
        value=256,
        default_value=256,
        description="Dimension of the hidden units.",
        display_unit="hidden dim",
        valid_min=64,
        valid_max=1024,
        math_cond="/ 8",
        automl_enabled="True",
        popular="yes",
    )
    dim_feedforward: int = INT_FIELD(
        value=2048,
        description="Dimension of the feedforward network",
        display_name="dim feedforward",
        valid_min=1,
    )
    dec_layers: int = INT_FIELD(
        value=10,
        default_value=10,
        description="Numer of decoder layers in the transformer",
        valid_min=1,
        valid_max=50,
        automl_enabled="TRUE",
        display_name="decoder layers",
    )
    pre_norm: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to add layer norm in the encoder or not.",
        display_name="Pre norm"
    )  # Add layer norm in encoder or not
    class_weight: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the classification error in the matching cost.",
        display_name="Class loss coefficient",
        popular="yes",
    )
    dice_weight: float = FLOAT_FIELD(
        value=5.0,
        default_value=5.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the focal loss of the binary mask in the matching cost.",
        display_name="focal loss coefficient",
        popular="yes",
    )
    mask_weight: float = FLOAT_FIELD(
        value=5.0,
        default_value=5.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the dice loss of the binary mask in the matching cost",
        display_name="mask loss coefficient",
        popular="yes",
    )
    train_num_points: int = INT_FIELD(
        value=12544,
        default_value=12544,
        description="The number of points P to sample.",
        display_name="number of points",
    )
    oversample_ratio: float = FLOAT_FIELD(
        value=3.0,
        default_value=3.0,
        description="Oversampling parameter.",
        display_name="oversampling ratio",
    )
    importance_sample_ratio: float = FLOAT_FIELD(
        value=0.75,
        default_value=0.75,
        description="Ratio of points that are sampled via importnace sampling.",
        display_name="importance sampling ratio",
        popular="yes",
    )
    deep_supervision: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to enable deep supervision.",
        display_name="deep supervision"
    )
    no_object_weight: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        description="The relative classification weight applied to the no-object category.",
        display_name="no object coefficient",
    )


@dataclass
class Swin:
    """Swin Transformer config."""

    type: str = STR_FIELD(
        value="tiny",
        default_value="tiny",
        display_name="swin transformer type",
        description="Swin Transformer type"
    )
    embed_dim: int = INT_FIELD(
        value=96,
        default_value=96,
        display_name="embedding dimensions",
        description="Number of input channels."
    )
    depths: List[int] = LIST_FIELD(
        arrList=[2, 2, 6, 2],
        display_name="swin transformer depth",
        description="Depths of each Swin Transformer stage."
    )
    num_heads: List[int] = LIST_FIELD(
        arrList=[3, 6, 12, 24],
        display_name="number of heads",
        description="Number of attention head of each stage."
    )
    patch_size: int = INT_FIELD(
        value=4,
        default_value=4,
        display_name="patch size",
        description="Patch size for swin transformer."
    )
    window_size: int = INT_FIELD(
        value=7,
        default_value=7,
        display_name="window size",
        description="Window size for Swin Transformer."
    )
    mlp_ratio: float = FLOAT_FIELD(
        value=4.0,
        default_value=4.0,
        display_name="mlp ratio",
        description="Ratio of mlp hidden dim to embedding dim."
    )
    qkv_bias: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="qkv bias",
        description="If True, add a learnable bias to query, key, value."
    )
    qk_scale: Optional[float] = FLOAT_FIELD(
        value=None,
        default_value=None,
        display_name="qk scale",
        description="Override default qk scale of head_dim ** -0.5 if set."
    )
    drop_rate: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        display_name="dropout rate",
        description="Dropout rate."
    )
    attn_drop_rate: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        display_name="attention dropout rate",
        description="Attention dropout rate."
    )
    drop_path_rate: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        display_name="stochastic drop rate",
        description="Stochastic drop rate"
    )
    ape: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="absolute position embedding",
        description="If True, add absolute position embedding to the patch embedding."
    )
    patch_norm: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="patch normalization",
        description="If True, add normalization after patch embedding."
    )
    out_indices: List[int] = LIST_FIELD(
        arrList=[0, 1, 2, 3],
        display_name="output indices",
        description="Output from which stages."
    )
    pretrain_img_size: int = INT_FIELD(
        value=384,
        default_value=384,
        display_name="pretrained image size",
        description="Input image size for training the pretrained model."
    )  # TODO
    use_checkpoint: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="use checkpointing",
        description="Whether to use checkpointing to save memory."
    )  # TODO
    out_features: List[str] = LIST_FIELD(
        arrList=["res2", "res3", "res4", "res5"],
        default_value=["res2", "res3", "res4", "res5"],
        description="List of output feature names for swin backbone.",
        display_name="output features"
    )


@dataclass
class EfficientViT:
    """EfficientViT config."""

    name: str = STR_FIELD(
        value="l0",  # b0-3; l0-3
        description="""efficient vit name.""",
        display_name="efficient vit name"
    )
    out_indices: List[int] = LIST_FIELD(
        arrList=[1, 2, 3],
        display_name="output indices",
        description="Output from which stages."
    )
    pretrain_img_size: int = INT_FIELD(
        value=384,
        default_value=384,
        display_name="pretrained imaeg size",
        description="Input image size for training the pretrained model."
    )  # TODO
    use_checkpoint: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="use checkpointing",
        description="Whether to use checkpointing to save memory."
    )  # TODO
    out_features: List[str] = LIST_FIELD(
        arrList=["res2", "res3", "res4", "res5"],
        default_value=["res2", "res3", "res4", "res5"],
        description="List of output feature names for swin backbone.",
        display_name="output features"
    )


@dataclass
class Backbone:
    """Backbone config."""

    type: str = STR_FIELD(
        value="swin",
        description="""backbone name.""",
        display_name="backbone name"
    )
    pretrained_weights: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="pretrained backbone path",
        description="[Optional] Path to a pretrained backbone file.",
    )
    swin: Swin = DATACLASS_FIELD(
        Swin(),
        descripton="Configuration hyper parameters for the Swin Transformer Backbone.",
        display_name="swin"
    )
    efficientvit: EfficientViT = DATACLASS_FIELD(
        EfficientViT(),
        descripton="Configuration hyper parameters for the Efficient-ViT Backbone.",
        display_name="efficient vit"
    )


@dataclass
class Mask2FormerModelConfig:
    """Mask2former model config."""

    export: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="export",
        description="A flag to enable export mode."
    )
    backbone: Backbone = DATACLASS_FIELD(
        Backbone(),
        descripton="Configuration hyper parameters for the Mask2Former Backbone.",
        display_name="backbone"
    )
    sem_seg_head: SemanticSegmentationHead = DATACLASS_FIELD(
        SemanticSegmentationHead(),
        descripton="Configuration hyper parameters for the Mask2Former Semantic Segmentation Head.",
        display_name="head"
    )
    mask_former: MaskFormer = DATACLASS_FIELD(
        MaskFormer(),
        descripton="Configuration hyper parameters for the Mask2Former model.",
        display_name="mask2former"
    )
    mode: str = STR_FIELD(
        value="panoptic",
        default_value="panoptic",
        display_name="segmentation mode",
        description="Segmentation mode.",
        valid_options=",".join(['panoptic', 'instance', 'semantic'])
    )
    object_mask_threshold: float = FLOAT_FIELD(
        value=0.4,
        default_value=0.4,
        description="""The value of the threshold to be used when
                    filtering out the object mask.""",
        display_name="object mask threshold"
    )
    overlap_threshold: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="""The value of the threshold to be used when
                    evaluating overlap.""",
        display_name="overlap threshold"
    )
    test_topk_per_image: int = INT_FIELD(
        value=100,
        default_value=100,
        description=" keep topk instances per image for instance segmentation.",
        display_name="top k per image",
    )
