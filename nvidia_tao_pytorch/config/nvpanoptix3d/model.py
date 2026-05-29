# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    depth_dim: int = INT_FIELD(
        value=256,
        default_value=256,
        description="Depth head dimension.",
        display_name="depth head dim.",
        valid_min=1,
        popular="yes",
    )
    ignore_value: int = INT_FIELD(
        value=255,
        default_value=255,
        description="Ignore value.",
        display_name="ignore value",
        valid_min=0,
        valid_max=255,
    )
    deformable_transformer_encoder_in_features: List[str] = LIST_FIELD(
        arrList=["res3", "res4", "res5"],
        default_value=["res3", "res4", "res5"],
        description="List of feature names for deformable transformer encoder input.",
        display_name="transformer encoder in_features"
    )
    num_classes: int = INT_FIELD(
        value=13,
        default_value=13,
        description="Number of classes.",
        display_name="number of classes.",
        valid_min=1,
    )
    norm: str = STR_FIELD(
        value="GN",
        default_value="GN",
        description="""Norm layer type.""",
        display_name="norm type"
    )
    in_features: List[str] = LIST_FIELD(
        arrList=["res2", "res3", "res4", "res5"],
        default_value=["res2", "res3", "res4", "res5"],
        description="List of input feature names.",
        display_name="transformer encoder in_features"
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
        popular="yes",
    )
    hidden_dim: int = INT_FIELD(
        value=256,
        default_value=256,
        description="Dimension of the hidden units.",
        display_unit="hidden dim",
        popular="yes",
    )
    transformer_dim_feedforward: int = INT_FIELD(
        value=1024,
        default_value=1024,
        description="Dimension of the feedforward network in the transformer",
        display_name="transformer dim feedforward",
        valid_min=1,
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
        display_name="decoder layers",
    )
    pre_norm: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to add layer norm in the encoder or not.",
        display_name="Pre norm"
    )
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
    depth_weight: float = FLOAT_FIELD(
        value=5.0,
        default_value=5.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the depth loss in the matching cost.",
        display_name="depth loss coefficient",
    )
    mp_occ_weight: float = FLOAT_FIELD(
        value=5.0,
        default_value=5.0,
        valid_min=0.0,
        valid_max="inf",
        description="The relative weight of the mp occ loss in the matching cost.",
        display_name="mp occ loss coefficient",
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
        description="Ratio of points that are sampled via important sampling.",
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
    size_divisibility: int = INT_FIELD(
        value=32,
        default_value=32,
        description="Size divisibility.",
        display_name="size divisibility",
    )


@dataclass
class Backbone:
    """Backbone config."""

    backbone_type: str = STR_FIELD(
        value="vggt",
        default_value="vggt",
        description="Type of backbone to use. Available backbone: vggt.",
        display_name="backbone name",
        valid_options=",".join(["vggt"])
    )
    pretrained_model_path: Optional[str] = STR_FIELD(
        value="",
        default_value="",
        display_name="pretrained backbone path",
        description="Path to a pretrained backbone file.",
    )


@dataclass
class Frustum3D:
    """Frustum3D config."""

    truncation: float = FLOAT_FIELD(
        value=3.0,
        default_value=3.0,
        description="The truncation value.",
        display_name="truncation",
    )
    iso_recon_value: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        description="The iso recon value.",
        display_name="iso recon value",
    )
    panoptic_weight: float = FLOAT_FIELD(
        value=25.0,
        default_value=25.0,
        description="The weight of the panoptic loss.",
        display_name="panoptic weight",
    )
    completion_weights: List[float] = LIST_FIELD(
        arrList=[50.0, 25.0, 10.0],
        default_value=[50.0, 25.0, 10.0],
        description="The weights of the completion loss.",
        display_name="completion weights",
    )
    surface_weight: float = FLOAT_FIELD(
        value=5.0,
        default_value=5.0,
        description="The weight of the surface loss.",
        display_name="surface weight",
    )
    unet_output_channels: int = INT_FIELD(
        value=16,
        default_value=16,
        description="The number of output channels of the UNet.",
        display_name="unet output channels",
    )
    unet_features: int = INT_FIELD(
        value=16,
        default_value=16,
        description="The number of features of the UNet.",
        display_name="unet features",
    )
    use_multi_scale: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Whether to use multi-scale.",
        display_name="use multi-scale",
    )
    grid_dimensions: int = INT_FIELD(
        value=256,
        default_value=256,
        description="The number of grid dimensions.",
        display_name="grid dimensions",
    )
    frustum_dims: int = INT_FIELD(
        value=256,
        default_value=256,
        description="The number of frustum dimensions.",
        display_name="frustum dimensions",
    )
    signed_channel: int = INT_FIELD(
        value=3,
        default_value=3,
        description="The number of signed channel.",
        display_name="signed channel",
    )


@dataclass
class Projection:
    """Projection config."""

    voxel_size: float = FLOAT_FIELD(
        value=0.03,
        default_value=0.03,
        description="The size of the voxel.",
        display_name="voxel size",
    )
    sign_channel: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Whether to use signed channel.",
        display_name="sign channel",
    )
    depth_feature_dim: int = INT_FIELD(
        value=256,
        default_value=256,
        description="The dimension of the depth feature.",
        display_name="depth feature dim",
    )


@dataclass
class NVPanoptix3DModelConfig:
    """NVPanoptix3D model config."""

    backbone: Backbone = DATACLASS_FIELD(
        Backbone(),
        description="Configuration hyper parameters for the NVPanoptix3D Backbone.",
        display_name="backbone"
    )
    sem_seg_head: SemanticSegmentationHead = DATACLASS_FIELD(
        SemanticSegmentationHead(),
        description="Configuration hyper parameters for the Mask2Former Semantic Segmentation Head.",
        display_name="segmentation head configs"
    )
    mask_former: MaskFormer = DATACLASS_FIELD(
        MaskFormer(),
        description="Configuration hyper parameters for the Mask2Former model.",
        display_name="mask2former"
    )
    frustum3d: Frustum3D = DATACLASS_FIELD(
        Frustum3D(),
        description="Configuration hyper parameters for the Frustum3D model.",
        display_name="frustum3d"
    )
    projection: Projection = DATACLASS_FIELD(
        Projection(),
        description="Configuration hyper parameters for the Projection model.",
        display_name="projection"
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
        description="The value of the threshold to be used when filtering out the object mask.",
        display_name="object mask threshold"
    )
    overlap_threshold: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="The value of the threshold to be used when evaluating overlap.",
        display_name="overlap threshold"
    )
    test_topk_per_image: int = INT_FIELD(
        value=100,
        default_value=100,
        description="Keep topk instances per image for instance segmentation.",
        display_name="top k per image",
    )
