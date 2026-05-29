# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the model."""

from typing import Optional, List, Any
from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    BOOL_FIELD,
    FLOAT_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD
)


@dataclass
class Sparse4DBackboneConfig:
    """Backbone configuration for Sparse4D, aligning with timm.ResNet and BackboneBase expectations."""

    type: str = STR_FIELD(
        value="resnet_101",
        default_value="resnet_101",
        description="Backbone type",
        display_name="Backbone type"
    )


@dataclass
class Sparse4DNeckConfig:
    """Neck configuration for Sparse4D."""

    type: str = STR_FIELD(
        value="FPN",
        default_value="FPN",
        description="Neck type",
        valid_options="FPN",
        display_name="Neck type"
    )
    num_outs: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=1,
        valid_max="inf",
        display_name="Number of output levels"
    )
    start_level: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max="inf",
        description="Start level for FPN",
        display_name="Start level for FPN"
    )
    out_channels: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=1,
        valid_max="inf",
        description="Output channels",
        display_name="Output channels"
    )
    in_channels: List[int] = LIST_FIELD(
        arrList=[256, 512, 1024, 2048],
        default_value=[256, 512, 1024, 2048],
        description="Input channels",
        display_name="Input channels"
    )
    add_extra_convs: str = STR_FIELD(
        value="on_output",
        default_value="on_output",
        description="Type of extra conv",
        valid_options="on_input,on_lateral,on_output,False",
        display_name="Type of extra conv"
    )
    relu_before_extra_convs: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Apply ReLU before extra convs",
        display_name="Apply ReLU before extra convs"
    )


@dataclass
class Sparse4DDepthBranchConfig:
    """Depth branch configuration for Sparse4D."""

    type: str = STR_FIELD(
        value="dense_depth",
        default_value="dense_depth",
        description="Depth branch type",
        display_name="Depth branch type"
    )
    embed_dims: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=1,
        valid_max="inf",
        description="Embedding dimensions",
        display_name="Embedding dimensions"
    )
    num_depth_layers: int = INT_FIELD(
        value=3,
        default_value=3,
        valid_min=1,
        valid_max="inf",
        description="Number of depth layers",
        display_name="Number of depth layers"
    )
    loss_weight: float = FLOAT_FIELD(
        value=0.2,
        default_value=0.2,
        valid_min=0,
        valid_max="inf",
        description="Weight for depth loss",
        display_name="Weight for depth loss"
    )


@dataclass
class Sparse4DInstanceBankConfig:
    """Instance bank configuration for Sparse4D."""

    num_anchor: int = INT_FIELD(
        value=900,
        default_value=900,
        valid_min=1,
        valid_max="inf",
        description="Number of anchors",
        display_name="Number of anchors"
    )
    anchor: str = STR_FIELD(
        value="",
        default_value="",
        description="Path to anchor file",
        display_name="Path to anchor file"
    )
    num_temp_instances: int = INT_FIELD(
        value=600,
        default_value=600,
        valid_min=0,
        valid_max="inf",
        description="Number of temporal instances",
        display_name="Number of temporal instances"
    )
    confidence_decay: float = FLOAT_FIELD(
        value=0.8,
        default_value=0.8,
        valid_min=0,
        valid_max=1,
        description="Confidence decay factor",
        display_name="Confidence decay factor"
    )
    feat_grad: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable gradients for features",
        display_name="Enable gradients for features"
    )
    default_time_interval: float = FLOAT_FIELD(
        value=0.033333,
        default_value=0.033333,
        valid_min=0,
        valid_max="inf",
        description="Default time interval",
        display_name="Default time interval"
    )
    embed_dims: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=1,
        valid_max="inf",
        description="Embedding dimensions",
        display_name="Embedding dimensions"
    )
    use_temporal_align: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Use temporal alignment",
        display_name="Use temporal alignment"
    )
    grid_size: Optional[float] = FLOAT_FIELD(
        value=None,
        default_value=None,
        description="Grid size",
        display_name="Grid size"
    )


@dataclass
class Sparse4DAnchorEncoderConfig:
    """Anchor encoder configuration for Sparse4D."""

    type: str = STR_FIELD(
        value="SparseBox3DEncoder",
        default_value="SparseBox3DEncoder",
        description="Anchor encoder type",
        display_name="Anchor encoder type"
    )
    vel_dims: int = INT_FIELD(
        value=3,
        default_value=3,
        valid_min=1,
        valid_max="inf",
        description="Velocity dimensions",
        display_name="Velocity dimensions"
    )
    embed_dims: List[int] = LIST_FIELD(
        arrList=[128, 32, 32, 64],
        default_value=[128, 32, 32, 64],
        description="Embedding dimensions",
        display_name="Embedding dimensions"
    )
    mode: str = STR_FIELD(
        value="cat",
        default_value="cat",
        description="Mode",
        valid_options="cat,add",
        display_name="Mode"
    )
    output_fc: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Output FC",
        display_name="Output FC"
    )
    in_loops: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        description="In loops",
        display_name="In loops"
    )
    out_loops: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=1,
        valid_max="inf",
        description="Out loops",
        display_name="Out loops"
    )
    pos_embed_only: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Pos embed only",
        display_name="Pos embed only"
    )


@dataclass
class Sparse4DKpsGeneratorConfig:
    """KPS generator configuration for Sparse4D."""

    embed_dims: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=1,
        valid_max="inf",
        description="Embedding dimensions",
        display_name="Embedding dimensions"
    )
    num_learnable_pts: int = INT_FIELD(
        value=6,
        default_value=6,
        valid_min=1,
        valid_max="inf",
        description="Number of learnable points",
        display_name="Number of learnable points"
    )
    fix_scale: List[List[Any]] = LIST_FIELD(
        arrList=[
            [0, 0, 0],
            [0.45, 0, 0],
            [-0.45, 0, 0],
            [0, 0.45, 0],
            [0, -0.45, 0],
            [0, 0, 0.45],
            [0, 0, -0.45]
        ],
        default_value=[
            [0, 0, 0],
            [0.45, 0, 0],
            [-0.45, 0, 0],
            [0, 0.45, 0],
            [0, -0.45, 0],
            [0, 0, 0.45],
            [0, 0, -0.45]
        ],
        description="Fixed scale",
        display_name="Fixed scale"
    )


@dataclass
class Sparse4DLossClsConfig:
    """Classification loss configuration for Sparse4D."""

    type: str = STR_FIELD(
        value="focal",
        default_value="focal",
        description="Classification loss type",
        display_name="Classification loss type"
    )
    use_sigmoid: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Use sigmoid",
        display_name="Use sigmoid"
    )
    gamma: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        valid_min=0,
        valid_max="inf",
        description="Focal loss gamma",
        display_name="Focal loss gamma"
    )
    alpha: float = FLOAT_FIELD(
        value=0.25,
        default_value=0.25,
        valid_min=0,
        valid_max=1,
        description="Focal loss alpha",
        display_name="Focal loss alpha"
    )
    loss_weight: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        valid_min=0,
        valid_max="inf",
        description="Loss weight",
        display_name="Loss weight"
    )


@dataclass
class Sparse4DLossRegConfig:
    """Regression loss configuration for Sparse4D."""

    type: str = STR_FIELD(
        value="sparse_box_3d",
        default_value="sparse_box_3d",
        description="Regression loss type",
        display_name="Regression loss type"
    )
    box_weight: float = FLOAT_FIELD(
        value=0.25,
        default_value=0.25,
        valid_min=0,
        valid_max="inf",
        description="Box loss weight",
        display_name="Box loss weight"
    )
    cls_allow_reverse: list = LIST_FIELD(
        arrList=[],
        default_value=[],
        description="Class allow reverse",
        display_name="Class allow reverse"
    )


@dataclass
class Sparse4DLossIDConfig:
    """ID loss configuration for Sparse4D."""

    type: str = STR_FIELD(
        value="cross_entropy_label_smooth",
        default_value="cross_entropy_label_smooth",
        description="ID loss type",
        display_name="ID loss type"
    )
    num_ids: int = INT_FIELD(
        value=70,
        default_value=70,
        valid_min=1,
        valid_max="inf",
        description="Number of IDs",
        display_name="Number of IDs"
    )


@dataclass
class Sparse4DLossConfig:
    """Loss configuration for Sparse4D."""

    cls: Sparse4DLossClsConfig = DATACLASS_FIELD(
        Sparse4DLossClsConfig(),
        description="Classification loss config",
        display_name="Classification loss config"
    )
    reg: Sparse4DLossRegConfig = DATACLASS_FIELD(
        Sparse4DLossRegConfig(),
        description="Regression loss config",
        display_name="Regression loss config"
    )
    id: Sparse4DLossIDConfig = DATACLASS_FIELD(
        Sparse4DLossIDConfig(),
        description="ID loss config",
        display_name="ID loss config"
    )


@dataclass
class Sparse4DDecoderConfig:
    """Decoder configuration for Sparse4D."""

    type: str = STR_FIELD(
        value="SparseBox3DDecoder",
        default_value="SparseBox3DDecoder",
        description="Decoder type",
        display_name="Decoder type"
    )
    score_threshold: float = FLOAT_FIELD(
        value=0.05,
        default_value=0.05,
        valid_min=0,
        valid_max=1,
        description="Score threshold",
        display_name="Score threshold"
    )


@dataclass
class Sparse4DBNNeckConfig:
    """BNNeck configuration for Sparse4D."""

    type: str = STR_FIELD(
        value="bnneck",
        default_value="bnneck",
        description="BNNeck type",
        display_name="BNNeck type"
    )
    feat_dim: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=1,
        valid_max="inf",
        description="Feature dimension",
        display_name="Feature dimension"
    )
    num_ids: int = INT_FIELD(
        value=70,
        default_value=70,
        valid_min=1,
        valid_max="inf",
        description="Number of IDs",
        display_name="Number of IDs"
    )


@dataclass
class Sparse4DVisibilityNetConfig:
    """VisibilityNet configuration for Sparse4D."""

    type: str = STR_FIELD(
        value="visibility_net",
        default_value="visibility_net",
        description="VisibilityNet type",
        display_name="VisibilityNet type"
    )
    embedding_dim: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=1,
        valid_max="inf",
        description="Embedding dimension",
        display_name="Embedding dimension"
    )
    hidden_channels: int = INT_FIELD(
        value=32,
        default_value=32,
        valid_min=1,
        valid_max="inf",
        description="Hidden channels",
        display_name="Hidden channels"
    )


@dataclass
class Sparse4DSamplerConfig:
    """Sampler configuration for Sparse4D."""

    num_dn_groups: int = INT_FIELD(
        value=5,
        default_value=5,
        valid_min=1,
        valid_max="inf",
        description="Number of DN groups",
        display_name="Number of DN groups"
    )
    num_temp_dn_groups: int = INT_FIELD(
        value=3,
        default_value=3,
        valid_min=0,
        valid_max="inf",
        description="Number of temporal DN groups",
        display_name="Number of temporal DN groups"
    )
    dn_noise_scale: List[float] = LIST_FIELD(
        arrList=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        default_value=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        description="DN noise scale",
        display_name="DN noise scale"
    )
    max_dn_gt: int = INT_FIELD(
        value=128,
        default_value=128,
        valid_min=1,
        valid_max="inf",
        description="Maximum DN ground truth",
        display_name="Maximum DN ground truth"
    )
    add_neg_dn: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Add negative DN",
        display_name="Add negative DN"
    )
    cls_weight: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        valid_min=0,
        valid_max="inf",
        description="Classification weight",
        display_name="Classification weight"
    )
    box_weight: float = FLOAT_FIELD(
        value=0.25,
        default_value=0.25,
        valid_min=0,
        valid_max="inf",
        description="Box weight",
        display_name="Box weight"
    )
    reg_weights: List[float] = LIST_FIELD(
        arrList=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        default_value=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        description="Regression weights",
        display_name="Regression weights"
    )
    use_temporal_align: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Use temporal alignment",
        display_name="Use temporal alignment"
    )
    gt_assign_threshold: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0,
        valid_max=1,
        description="GT assign threshold",
        display_name="GT assign threshold"
    )


@dataclass
class Sparse4DDeformableModelConfig:
    """Deformable model configuration for Sparse4D."""

    embed_dims: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=1,
        valid_max="inf",
        description="Embedding dimensions"
    )
    num_groups: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="Number of groups"
    )
    num_levels: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=1,
        valid_max="inf",
        description="Number of levels",
        display_name="Number of levels"
    )
    attn_drop: float = FLOAT_FIELD(
        value=0.15,
        default_value=0.15,
        valid_min=0,
        valid_max=1,
        description="Attention dropout",
        display_name="Attention dropout"
    )
    use_deformable_func: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Use deformable function",
        display_name="Use deformable function"
    )
    use_camera_embed: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Use camera embedding",
        display_name="Use camera embedding"
    )
    residual_mode: str = STR_FIELD(
        value="cat",
        default_value="cat",
        description="Residual mode",
        valid_options="cat,add",
        display_name="Residual mode"
    )
    num_cams: int = INT_FIELD(
        value=6,
        default_value=6,
        valid_min=1,
        valid_max="inf",
        description="Number of cameras",
        display_name="Number of cameras"
    )
    max_num_cams: int = INT_FIELD(
        value=20,
        default_value=20,
        valid_min=1,
        valid_max="inf",
        description="Maximum number of cameras",
        display_name="Maximum number of cameras"
    )
    proj_drop: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0,
        valid_max=1,
        description="Projection dropout",
        display_name="Projection dropout"
    )
    kps_generator: Sparse4DKpsGeneratorConfig = DATACLASS_FIELD(
        Sparse4DKpsGeneratorConfig(),
        description="KPS generator config",
        display_name="KPS generator config"
    )


@dataclass
class Sparse4DRefineLayerConfig:
    """Refine layer configuration for Sparse4D."""

    type: str = STR_FIELD(
        value="sparse_box_3d_refinement_module",
        default_value="sparse_box_3d_refinement_module",
        description="Refine layer type",
        display_name="Refine layer type"
    )
    embed_dims: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=1,
        valid_max="inf",
        description="Embedding dimensions",
        display_name="Embedding dimensions"
    )
    refine_yaw: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Refine yaw",
        display_name="Refine yaw"
    )
    with_quality_estimation: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="With quality estimation",
        display_name="With quality estimation"
    )


@dataclass
class Sparse4DGraphModelConfig:
    """Graph model configuration for Sparse4D."""

    type: str = STR_FIELD(
        value="MultiheadAttention",
        default_value="MultiheadAttention",
        description="Graph model type",
        display_name="Graph model type"
    )
    embed_dims: int = INT_FIELD(
        value=512,
        default_value=512,
        valid_min=1,
        valid_max="inf",
        description="Embedding dimensions",
        display_name="Embedding dimensions"
    )
    num_heads: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="Number of heads",
        display_name="Number of heads"
    )
    batch_first: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Batch first",
        display_name="Batch first"
    )
    dropout: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        valid_min=0,
        valid_max=1,
        description="Dropout rate",
        display_name="Dropout rate"
    )


@dataclass
class Sparse4DNormLayerConfig:
    """Norm layer configuration for Sparse4D."""

    type: str = STR_FIELD(
        value="LN",
        default_value="LN",
        description="Norm layer type",
        display_name="Norm layer type"
    )
    normalized_shape: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=1,
        valid_max="inf",
        description="Normalized shape",
        display_name="Normalized shape"
    )


@dataclass
class Sparse4DActConfig:
    """Activation configuration for Sparse4D."""

    type: str = STR_FIELD(
        value="ReLU",
        default_value="ReLU",
        description="Activation type",
        display_name="Activation type"
    )
    inplace: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Inplace",
        display_name="Inplace"
    )


@dataclass
class Sparse4DFFNConfig:
    """FFN configuration for Sparse4D."""

    type: str = STR_FIELD(
        value="AsymmetricFFN",
        default_value="AsymmetricFFN",
        description="FFN type",
        display_name="FFN type"
    )
    in_channels: int = INT_FIELD(
        value=512,
        default_value=512,
        valid_min=1,
        valid_max="inf",
        description="In channels",
        display_name="In channels"
    )
    pre_norm: Sparse4DNormLayerConfig = DATACLASS_FIELD(
        Sparse4DNormLayerConfig(),
        description="Pre-norm config",
        display_name="Pre-norm config"
    )
    embed_dims: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=1,
        valid_max="inf",
        description="Embedding dimensions",
        display_name="Embedding dimensions"
    )
    feedforward_channels: int = INT_FIELD(
        value=1024,
        default_value=1024,
        valid_min=1,
        valid_max="inf",
        description="Feedforward channels",
        display_name="Feedforward channels"
    )
    num_fcs: int = INT_FIELD(
        value=2,
        default_value=2,
        valid_min=1,
        valid_max="inf",
        description="Number of feedforward channels",
        display_name="Number of feedforward channels"
    )
    ffn_drop: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        valid_min=0,
        valid_max=1,
        description="FFN dropout",
        display_name="FFN dropout"
    )
    act_cfg: Sparse4DActConfig = DATACLASS_FIELD(
        Sparse4DActConfig(),
        description="Activation config",
        display_name="Activation config"
    )


@dataclass
class Sparse4DHeadConfig:
    """Head configuration for Sparse4D."""

    type: str = STR_FIELD(
        value="sparse4d",
        default_value="sparse4d",
        description="Head type",
        display_name="Head type"
    )
    num_output: int = INT_FIELD(
        value=300,
        default_value=300,
        valid_min=1,
        valid_max="inf",
        description="Number of output instances",
        display_name="Number of output instances"
    )
    cls_threshold_to_reg: float = FLOAT_FIELD(
        value=0.05,
        default_value=0.05,
        valid_min=0,
        valid_max=1,
        description="Classification threshold for regression",
        display_name="Classification threshold for regression"
    )
    decouple_attn: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Decouple attention",
        display_name="Decouple attention"
    )
    return_feature: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Return instance features",
        display_name="Return instance features"
    )
    use_reid_sampling: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Use Re-ID sampling",
        display_name="Use Re-ID sampling"
    )
    embed_dims: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=1,
        valid_max="inf",
        description="Embedding dimensions",
        display_name="Embedding dimensions"
    )
    reid_dims: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max="inf",
        description="Re-ID dimensions",
        display_name="Re-ID dimensions"
    )
    num_groups: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="Number of groups",
        display_name="Number of groups"
    )
    num_decoder: int = INT_FIELD(
        value=6,
        default_value=6,
        valid_min=1,
        valid_max="inf",
        description="Number of decoder layers",
        display_name="Number of decoder layers"
    )
    num_single_frame_decoder: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        description="Number of single-frame decoder layers",
        display_name="Number of single-frame decoder layers"
    )
    drop_out: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        valid_min=0,
        valid_max=1,
        description="Dropout rate",
        display_name="Dropout rate"
    )
    temporal: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Enable temporal modeling",
        display_name="Enable temporal modeling"
    )
    with_quality_estimation: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Enable quality estimation",
        display_name="Enable quality estimation"
    )
    operation_order: List[str] = LIST_FIELD(
        arrList=[
            "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
            "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
            "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
            "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
            "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
            "deformable", "ffn", "norm", "refine"
        ],
        default_value=[
            "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
            "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
            "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
            "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
            "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm",
            "deformable", "ffn", "norm", "refine"
        ],
        description="Operation order",
        display_name="Operation order"
    )
    visibility_net: Sparse4DVisibilityNetConfig = DATACLASS_FIELD(
        Sparse4DVisibilityNetConfig(),
        description="Visibility net config",
        display_name="Visibility net config"
    )
    instance_bank: Sparse4DInstanceBankConfig = DATACLASS_FIELD(
        Sparse4DInstanceBankConfig(),
        description="Instance bank config",
        display_name="Instance bank config"
    )
    anchor_encoder: Sparse4DAnchorEncoderConfig = DATACLASS_FIELD(
        Sparse4DAnchorEncoderConfig(),
        description="Anchor encoder config",
        display_name="Anchor encoder config"
    )
    sampler: Sparse4DSamplerConfig = DATACLASS_FIELD(
        Sparse4DSamplerConfig(),
        description="Sampler config",
        display_name="Sampler config"
    )
    reg_weights: List[float] = LIST_FIELD(
        arrList=[2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        default_value=[2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        description="Regression weights",
        display_name="Regression weights"
    )
    loss: Sparse4DLossConfig = DATACLASS_FIELD(
        Sparse4DLossConfig(),
        description="Loss config",
        display_name="Loss config"
    )
    bnneck: Sparse4DBNNeckConfig = DATACLASS_FIELD(
        Sparse4DBNNeckConfig(),
        description="BN neck config",
        display_name="BN neck config"
    )
    deformable_model: Sparse4DDeformableModelConfig = DATACLASS_FIELD(
        Sparse4DDeformableModelConfig(),
        description="Deformable model config",
        display_name="Deformable model config"
    )
    refine_layer: Sparse4DRefineLayerConfig = DATACLASS_FIELD(
        Sparse4DRefineLayerConfig(),
        description="Refine layer config",
        display_name="Refine layer config"
    )
    valid_vel_weight: float = FLOAT_FIELD(
        value=-1,
        default_value=-1,
        valid_min=-1,
        valid_max="inf",
        description="Valid velocity weight",
        display_name="Valid velocity weight"
    )
    graph_model: Sparse4DGraphModelConfig = DATACLASS_FIELD(
        Sparse4DGraphModelConfig(),
        description="Graph model config",
        display_name="Graph model config"
    )
    temp_graph_model: Sparse4DGraphModelConfig = DATACLASS_FIELD(
        Sparse4DGraphModelConfig(),
        description="Temp graph model config",
        display_name="Temp graph model config"
    )
    decoder: Sparse4DDecoderConfig = DATACLASS_FIELD(
        Sparse4DDecoderConfig(),
        description="Decoder config",
        display_name="Decoder config"
    )
    norm_layer: Sparse4DNormLayerConfig = DATACLASS_FIELD(
        Sparse4DNormLayerConfig(),
        description="Norm layer config",
        display_name="Norm layer config"
    )
    ffn: Sparse4DFFNConfig = DATACLASS_FIELD(
        Sparse4DFFNConfig(),
        description="FFN config",
        display_name="FFN config"
    )


@dataclass
class Sparse4DModelConfig:
    """Model configuration for Sparse4D."""

    type: str = STR_FIELD(
        value="sparse4d",
        default_value="sparse4d",
        description="Model type",
        display_name="Model type"
    )
    embed_dims: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=1,
        valid_max="inf",
        description="Embedding dimensions",
        display_name="Embedding dimensions"
    )
    use_grid_mask: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Use grid mask",
        display_name="Use grid mask"
    )
    use_deformable_func: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Use deformable function",
        display_name="Use deformable function"
    )
    input_shape: List[int] = LIST_FIELD(
        arrList=[1408, 512],
        default_value=[1408, 512],
        description="Input image shape",
        display_name="Input image shape"
    )
    backbone: Sparse4DBackboneConfig = DATACLASS_FIELD(
        Sparse4DBackboneConfig(),
        description="Backbone config",
        display_name="Backbone config"
    )
    neck: Sparse4DNeckConfig = DATACLASS_FIELD(
        Sparse4DNeckConfig(),
        description="Neck config",
        display_name="Neck config"
    )
    depth_branch: Sparse4DDepthBranchConfig = DATACLASS_FIELD(
        Sparse4DDepthBranchConfig(),
        description="Depth branch config",
        display_name="Depth branch config"
    )
    head: Sparse4DHeadConfig = DATACLASS_FIELD(
        Sparse4DHeadConfig(),
        description="Head config",
        display_name="Head config"
    )
    use_temporal_align: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Use temporal alignment",
        display_name="Use temporal alignment"
    )
