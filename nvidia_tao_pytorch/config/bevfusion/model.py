# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the model."""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    BOOL_FIELD,
    LIST_FIELD,
    DICT_FIELD,
    DATACLASS_FIELD
)
from nvidia_tao_pytorch.config.bevfusion.dataset import BEVFusionDataPreprocessorConfig


@dataclass
class ImageBackboneConfig:
    """Configuration parameters for Image Backbone."""

    type: str = STR_FIELD(
        value="mmdet.SwinTransformer",
        default_value="mmdet.SwinTransformer",
        display_name="Image Backbone Type",
        description="Name of Image Backbone for 3D Fusion"
    )
    embed_dims: int = INT_FIELD(
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
    window_size: int = INT_FIELD(
        value=7,
        default_value=7,
        display_name="window size",
        description="Window size for Swin Transformer."
    )
    mlp_ratio: int = INT_FIELD(
        value=4,
        default_value=4,
        display_name="mlp ratio",
        description="Ratio of mlp hidden dim to embedding dim."
    )
    qkv_bias: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="qkv bias",
        description="If True, add a learnable bias to query, key, value."
    )
    qk_scale: Optional[str] = STR_FIELD(
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
        value=0.2,
        default_value=0.2,
        display_name="stochastic drop rate",
        description="Stochastic drop rate"
    )
    patch_norm: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="patch normalization",
        description="If True, add normalization after patch embedding."
    )
    out_indices: List[int] = LIST_FIELD(
        arrList=[1, 2, 3],
        display_name="output indices",
        description="Output from which stages."
    )
    with_cp: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="with checkpoint",
        description=(
            "Use checkpoint or not. Using checkpoint will save some memory "
            "while slowing down the training speed."
        )
    )
    convert_weights: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="convert weights",
        description=(
            "The flag indicates whether the pre-trained model is from the original repo."
        )
    )
    init_cfg: Optional[Dict[Any, Any]] = DICT_FIELD(
        hashMap={
            "type": "Pretrained",
            "checkpoint": (
                "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"
            )
        },
        default_value=None,
        description="Configuration for initialzation."
    )


@dataclass
class ImageNeckConfig:
    """Configuration parameters for Image Neck."""

    type: str = STR_FIELD(
        value="GeneralizedLSSFPN",
        default_value="GeneralizedLSSFPN",
        display_name="Image neck name",
        description="Image Neck Name"
    )
    in_channels: List[int] = LIST_FIELD(
        arrList=[192, 384, 768],
        display_name="input channels",
        description="The number of input channels for image neck."
    )
    out_channels: int = INT_FIELD(
        value=256,
        default_value=256,
        display_name="output channels",
        description="The number of output channels for image neck."
    )
    start_level: int = INT_FIELD(
        value=0,
        default_value=0,
        display_name="starting level",
        description="Starting level for image neck."
    )
    num_outs: int = INT_FIELD(
        value=3,
        default_value=0,
        display_name="number of output",
        description="The number of outputput for image neck."
    )
    norm_cfg: Dict[Any, Any] = DICT_FIELD(
        hashMap={"type": 'BN2d', "requires_grad": True},
        display_name="normalization config",
        description="The configuration of normalization for image neck."
    )
    act_cfg: Dict[Any, Any] = DICT_FIELD(
        hashMap={"type": 'ReLU', "inplace": True},
        display_name="activation config",
        description="The configuration of activation for image neck."
    )
    upsample_cfg: Dict[Any, Any] = DICT_FIELD(
        hashMap={"mode": 'bilinear', "align_corners": False},
        display_name="upsampling config",
        description="The configuration of upsampling for image neck."
    )


@dataclass
class ViewTransformConfig:
    """Configuration parameters for View Transform."""

    type: str = STR_FIELD(
        value="DepthLSSTransform",
        default_value="DepthLSSTransform",
        display_name="view transform Name",
        description="Image view transform name.",
        valid_options=",".join(["DepthLSSTransform", "LSSTransform"])

    )
    in_channels: int = INT_FIELD(
        value=256,
        default_value=256,
        display_name="input channels",
        description="The number of input channels for view transform."
    )
    out_channels: int = INT_FIELD(
        value=80,
        default_value=80,
        display_name="output channels",
        description="The number of output channels for view transform."
    )
    image_size: List[int] = LIST_FIELD(
        arrList=[256, 704],
        display_name="image size",
        description="Image size for view transform."
    )
    feature_size: List[int] = LIST_FIELD(
        arrList=[32, 88],
        display_name="feature size",
        description="Feature size for view transform."
    )
    xbound: List[float] = LIST_FIELD(
        arrList=[-54.0, 54.0, 0.3],
        display_name="x range",
        description="The grid range for x-axis."
    )
    ybound: List[float] = LIST_FIELD(
        arrList=[-54.0, 54.0, 0.3],
        display_name="y range",
        description="The grid range for y-axis."
    )
    zbound: List[float] = LIST_FIELD(
        arrList=[-10.0, 10.0, 20.0],
        display_name="z range",
        description="The grid range for z-axis."
    )
    dbound: List[float] = LIST_FIELD(
        arrList=[1.0, 60.0, 0.5],
        display_name="depth range",
        description="The grid range for depth."
    )
    downsample: int = INT_FIELD(
        value=2,
        default_value=2,
        display_name="downsample ratio",
        description="The ratio for downsampling."
    )


@dataclass
class FusionLayerConfig:
    """Configuration parameters for Fusion Layer."""

    type: str = STR_FIELD(
        value="ConvFuser",
        default_value="ConvFuser",
        display_name="fusion layer name",
        description="The fusion layer name."
    )
    in_channels: List[int] = LIST_FIELD(
        arrList=[80, 256],
        display_name="input channels",
        description="The number of input channels for fusion layer."
    )
    out_channels: int = INT_FIELD(
        value=256,
        display_name="output channels",
        description="The number of output channels for fusion layer."
    )


@dataclass
class LidarBackboneConfig:
    """Configuration parameters for Lidar Backbone."""

    type: str = STR_FIELD(
        value="SECOND",
        default_value="SECOND",
        display_name="lidar backbone name",
        description="The lidar backbone name."
    )
    in_channels: int = INT_FIELD(
        value=256,
        default_value=256,
        display_name="input channels",
        description="The number of input channels for lidar backbone."
    )
    out_channels: List[int] = LIST_FIELD(
        arrList=[128, 256],
        display_name="output channels",
        description="The number of output channels for lidar backbone."
    )
    layer_nums: List[int] = LIST_FIELD(
        arrList=[5, 5],
        display_name="number of layer",
        description="The number of layer in each stage for lidar backbone."
    )
    layer_strides: List[int] = LIST_FIELD(
        arrList=[1, 2],
        display_name="number of layer",
        description="Number of layers in each stage for lidar backbone."
    )
    norm_cfg: Dict[Any, Any] = DICT_FIELD(
        hashMap={"type": "BN", "eps": 0.001, "momentum": 0.01},
        display_name="normalization config",
        description="The configuration of normalization for lidar backbone."
    )
    conv_cfg: Dict[Any, Any] = DICT_FIELD(
        hashMap={"type": "Conv2d", "bias": False},
        display_name="convolution config",
        description="The configuration of convolution layers for lidar backbone."
    )


@dataclass
class LidarNeckConfig:
    """Configuration parameters for Lidar Neck."""

    type: str = STR_FIELD(
        value="SECONDFPN",
        default_value="SECONDFPN",
        display_name="lidar neck name",
        description="The lidar neck name."
    )
    in_channels: List[int] = LIST_FIELD(
        arrList=[128, 256],
        display_name="input channels",
        description="The number of input channels for lidar neck."
    )
    out_channels: List[int] = LIST_FIELD(
        arrList=[256, 256],
        display_name="output channels",
        description="The number of output channels for lidar neck."
    )
    upsample_strides: List[int] = LIST_FIELD(
        arrList=[1, 2],
        display_name="upsample strides",
        description="Strides used to upsample the feature map for lidar neck."
    )
    norm_cfg: Dict[Any, Any] = DICT_FIELD(
        hashMap={"type": "BN", "eps": 0.001, "momentum": 0.01},
        display_name="normalization config",
        description="The configuration of normalization for lidar neck."
    )
    upsample_cfg: Dict[Any, Any] = DICT_FIELD(
        hashMap={"type": "deconv", "bias": False},
        display_name="upsample configuration",
        description="The configuration of upsample layers for lidar neck."
    )
    use_conv_for_no_stride: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="use convolution for stride 1",
        description="Whether to use conv when stride is 1."
    )


@dataclass
class LidarEncoderConfig:
    """Configuration parameters for Lidar Encoder."""

    type: str = STR_FIELD(
        value="BEVFusionSparseEncoder",
        default_value="BEVFusionSparseEncoder",
        display_name="lidar encoder name",
        description="The lidar encoder name."
    )
    in_channels: int = INT_FIELD(
        value=4,
        default_value=4,
        display_name="input channels",
        description="The number of input channels for lidar encoder."
    )
    sparse_shape: List[int] = LIST_FIELD(
        arrList=[1440, 1440, 41],
        display_name="sparse shape",
        description="The sparse shape of input tensor."
    )
    order: List[str] = LIST_FIELD(
        arrList=['conv', 'norm', 'act'],
        display_name="convolution module order",
        description="Order of conv module."
    )
    norm_cfg: Dict[Any, Any] = DICT_FIELD(
        hashMap={"type": "BN1d", "eps": 0.001, "momentum": 0.01},
        display_name="normalization config",
        description="The configuration of normalization for lidar encoder."
    )
    encoder_channels: Tuple = ((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128))
    encoder_paddings: Tuple = ((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0))
    block_type: str = STR_FIELD(
        value='basicblock',
        default_value='basicblock',
        display_name="block type",
        description="Type of the block to use."
    )


@dataclass
class AttentionLayerConfig:
    """Configuration parameters for Attention Layer."""

    embed_dims: int = INT_FIELD(
        value=128,
        default_value=128,
        display_name="embedding dimensions",
        description="Number of input channels for attention layer."
    )
    num_heads: int = INT_FIELD(
        value=8,
        default_value=8,
        display_name="number of heads",
        description="Number of attention heads."
    )
    dropout: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        display_name="dropout probability",
        description="Dropout probability on attention weights."
    )


@dataclass
class DecoderLayerConfig:
    """Configuration parameters for Decoder Layer."""

    type: str = STR_FIELD(
        value='TransformerDecoderLayer',
        default_value="TransformerDecoderLayer",
        display_name="decoder layer name",
        description="Transformer decoder layer name."
    )
    self_attn_cfg: AttentionLayerConfig = DATACLASS_FIELD(
        AttentionLayerConfig(),
        description="The configuration for self attention module."
    )
    cross_attn_cfg: AttentionLayerConfig = DATACLASS_FIELD(
        AttentionLayerConfig(),
        description="The configuration for cross attention module."
    )
    ffn_cfg: Dict[Any, Any] = DICT_FIELD(
        hashMap={"embed_dims": 128, "feedforward_channels": 256,
                 "num_fcs": 2, "ffn_drop": 0.1,
                 "act_cfg": {"type": 'ReLU', "inplace": True}},
        display_name="ffn config",
        description="The configuration for ffn module."
    )
    norm_cfg: Dict[Any, Any] = DICT_FIELD(
        hashMap={"type": 'LN'},
        display_name="normalization config",
        description="The configuration of normalization for transformer decoder layer."
    )
    pos_encoding_cfg: Dict[Any, Any] = DICT_FIELD(
        hashMap={"input_channel": 2, "num_pos_feats": 128},
        display_name="position encoding config",
        description="Position Encoding parameters."
    )


@dataclass
class BboxCoderConfig:
    """Configuration parameters for Bbox Coder."""

    type: str = STR_FIELD(
        value='TAO3DBBoxCoder',
        default_value="TAO3DBBoxCoder",
        display_name="bounding box coder",
        description="Boudning box encoder.",
        valid_options=",".join(["TAO3DBBoxCoder"])
    )
    score_threshold: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        display_name="score threshold",
        description="Score threshold to filter bounding boxes in box encoder."
    )
    code_size: int = INT_FIELD(
        value=12,
        default_value=12,
        display_name="code size",
        description="Bounding box encoding size."
    )


@dataclass
class BboxHeadConfig:
    """Configuration parameters for Bbox Head."""

    type: str = STR_FIELD(
        value="BEVFusionHead",
        default_value="BEVFusionHead",
        display_name="Bounding box prediction head name",
        description="Prediction head name.",
        valid_options=",".join(["BEVFusionHead"])
    )
    num_proposals: int = INT_FIELD(
        value=200,
        default_value=200,
        display_name="number of proposals",
        description="Number of proposals."
    )
    auxiliary: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="is auxiliary",
        description="Whether to enable auxiliary training."
    )
    in_channels: int = INT_FIELD(
        value=512,
        default_value=512,
        display_name="input channels",
        description="Number of channels in the input feature map."
    )
    hidden_channel: int = INT_FIELD(
        value=128,
        default_value=128,
        display_name="hidden channels",
        description="Number of hiden channel."
    )
    num_classes: int = INT_FIELD(
        value=1,
        default_value=1,
        display_name="class numbers",
        description="Number of classes."
    )
    nms_kernel_size: int = INT_FIELD(
        value=3,
        default_value=3,
        display_name="nms kernel size",
        description="NMS kernel size."
    )
    bn_momentum: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        display_name="batch norm momentum",
        description="Batch Norm momentum."
    )
    num_decoder_layers: int = INT_FIELD(
        value=1,
        default_value=1,
        display_name="decoder layer number",
        description="Number of decoder layer."
    )
    out_size_factor: int = INT_FIELD(
        value=8,
        default_value=8,
        display_name="output size factor",
        description="Output size factor."
    )
    bbox_coder: BboxCoderConfig = DATACLASS_FIELD(
        BboxCoderConfig(),
        description="The configuration for bounding box encoder."
    )
    decoder_layer: DecoderLayerConfig = DATACLASS_FIELD(
        DecoderLayerConfig(),
        description="The configuration for decoder layer."
    )
    code_weights: List[float] = LIST_FIELD(
        arrList=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        display_name="code weights",
        description="Weights for box encoder."
    )
    nms_type: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="nms type",
        description="The type of NMS."
    )
    assigner: Dict[Any, Any] = DICT_FIELD(
        hashMap={"type": "HungarianAssigner3D",
                 "iou_calculator": {'type': 'BboxOverlaps3D', 'coordinate': 'lidar'},
                 "cls_cost": {'type': 'mmdet.FocalLossCost', 'gamma': 2.0, 'alpha': 0.25, 'weight': 0.15},
                 "reg_cost": {'type': 'BBoxBEVL1Cost', 'weight': 0.25},
                 "iou_cost": {'type': 'IoU3DCost', 'weight': 0.25}
                 },
        display_name="assigner configuration",
        description="The configuration for assginer."
    )
    common_heads: Optional[Dict[Any, Any]] = DICT_FIELD(
        hashMap={'center': [2, 2], 'height': [1, 2], 'dim': [3, 2], 'rot': [6, 2]},
        display_name="common heads configuration",
        description="The configuration for common heads."
    )
    loss_cls: Optional[Dict[Any, Any]] = DICT_FIELD(
        hashMap={'type': 'mmdet.FocalLoss',
                 'use_sigmoid': True, 'gamma': 2.0,
                 'alpha': 0.25, 'reduction': 'mean', 'loss_weight': 1.0},
        display_name="classification loss configuration",
        description="The configuration for classification loss."
    )
    loss_heatmap: Optional[Dict[Any, Any]] = DICT_FIELD(
        hashMap={'type': 'mmdet.GaussianFocalLoss',
                 'reduction': 'mean', 'loss_weight': 1.0},
        display_name="heatmap loss configuration",
        description="The configuration for heatmap loss."
    )
    loss_bbox: Optional[Dict[Any, Any]] = DICT_FIELD(
        hashMap={'type': 'mmdet.L1Loss',
                 'reduction': 'mean', 'loss_weight': 0.25},
        display_name="bounding box loss configuration",
        description="The configuration for bounding box loss."
    )


@dataclass
class BEVFusionModelConfig:
    """Configuration parameters for BEVFusion Model."""

    type: str = STR_FIELD(
        value="BEVFusion",
        default_value="BEVFusion",
        display_name="model name",
        description="Model name",
        valid_options=",".join(["BEVFusion"])
    )
    point_cloud_range: List[float] = LIST_FIELD(
        arrList=[0, -40, -3, 70.4, 40, 1],
        display_name="point cloud range",
        description="point cloud range"
    )
    voxel_size: List[float] = LIST_FIELD(
        arrList=[0.05, 0.05, 0.1],
        display_name="voxel size",
        description="voxel size in voxelization"
    )
    post_center_range: List[float] = LIST_FIELD(
        arrList=[-61.2, -61.2, -20.0, 61.2, 61.2, 20.0],
        display_name="post center range",
        description="post processing center filter range"
    )
    grid_size: List[int] = LIST_FIELD(
        arrList=[1440, 1440, 41],
        display_name="grid size",
        description="Grid size for bevfusion model"
    )
    data_preprocessor: BEVFusionDataPreprocessorConfig = DATACLASS_FIELD(
        BEVFusionDataPreprocessorConfig(),
        description="Configurable parameters to construct the preprocessor for the bevfusion model."
    )
    img_backbone: Optional[ImageBackboneConfig] = DATACLASS_FIELD(
        ImageBackboneConfig(),
        description="Configurable parameters to construct the camera image backbone for the bevfusion model."
    )
    img_neck: Optional[ImageNeckConfig] = DATACLASS_FIELD(
        ImageNeckConfig(),
        description="Configurable parameters to construct the camera image neck for the bevfusion model."
    )
    view_transform: Optional[ViewTransformConfig] = DATACLASS_FIELD(
        ViewTransformConfig(),
        description="Configurable parameters to construct the camera view transform for the bevfusion model."
    )
    pts_backbone: LidarBackboneConfig = DATACLASS_FIELD(
        LidarBackboneConfig(),
        description="Configurable parameters to construct the lidar pofort cloud backbone for the bevfusion model."
    )
    pts_voxel_encoder: Optional[Dict[Any, Any]] = DICT_FIELD(
        hashMap={"type": "HardSimpleVFE", "num_features": 4},
        description="Configurable parameters to construct the lidar pofort cloud voxel encoder for the bevfusion model."
    )
    pts_middle_encoder: LidarEncoderConfig = DATACLASS_FIELD(
        LidarEncoderConfig(),
        description="Configurable parameters to construct the lidar encoder for the bevfusion model."
    )
    pts_neck: LidarNeckConfig = DATACLASS_FIELD(
        LidarNeckConfig(),
        description="Configurable parameters to construct the lidar neck for the bevfusion model."
    )
    fusion_layer: Optional[FusionLayerConfig] = DATACLASS_FIELD(
        FusionLayerConfig(),
        description="Configurable parameters to construct the fusion layer for the bevfusion model."
    )
    bbox_head: BboxHeadConfig = DATACLASS_FIELD(
        BboxHeadConfig(),
        description="Configurable parameters to construct the bounding box head for the bevfusion model."
    )
