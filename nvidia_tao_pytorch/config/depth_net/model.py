# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the model."""

from dataclasses import dataclass
from typing import List, Optional

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class MonoBackBone:
    """Define MonoBackBone dependency config"""

    pretrained_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="Pretrained path for mono backbone",
        description="""Path to load depth anything v2 as an encoder for Monocular DepthNet""",
    )
    use_bn: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Batch normalization in Monocular DepthNet",
        description="""A flag specifying whether to use batch normalization in Monocular DepthNet""",
    )
    use_clstoken: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Class token in Monocular DepthNet",
        description="""A flag specifying whether to use class token""",
    )


@dataclass
class StereoBackBone:
    """Define StereoBackBone dependency config"""

    depth_anything_v2_pretrained_path: Optional[str] = STR_FIELD(
        value="",
        default_value="",
        description="""Path to load depth anything v2 as an encoder for Stereo DepthNet (FoundationStereo)""",
    )
    edgenext_pretrained_path: Optional[str] = STR_FIELD(
        value="",
        default_value="",
        description="""Path to load edgenext encoder for Stereo DepthNet (FoundationStereo)""",
    )
    use_bn: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="batch normalization in DepthAnythingV2",
        description="""A flag specifying whether to use batch normalization in DepthAnythingV2""",
    )
    use_clstoken: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="class token in DepthAnythingV2",
        description="""A flag specifying whether to use class token""",
    )


@dataclass
class DepthNetModelConfig:
    """DepthNet model config."""

    model_type: str = STR_FIELD(
        value="MetricDepthAnything",
        default_value="MetricDepthAnything",
        description="Network name",
        valid_options=",".join([
            "FoundationStereo", "FastFoundationStereo",
            "MetricDepthAnything", "RelativeDepthAnything"
        ])
    )
    mono_backbone: MonoBackBone = DATACLASS_FIELD(
        MonoBackBone(),
        value="",
        default_value="",
        display_name="Mono backbone configuration",
        description="Network defined paths for Monocular DepthNet Backbone",
    )
    stereo_backbone: StereoBackBone = DATACLASS_FIELD(
        StereoBackBone(),
        value="",
        default_value="",
        display_name="Stereo backbone configuration",
        description="Network defined paths for Edgenext and Depthanythingv2",
    )
    hidden_dims: List[int] = LIST_FIELD(
        arrList=[128, 128, 128],
        description="The hidden dimensions.",
        display_name="The hidden dimensions."
    )
    corr_radius: int = INT_FIELD(
        value=4,
        default_value=4,
        description="The width of the correlation pyramid",
        display_name="correlation pyramid width",
        valid_min=2,
        valid_max=8,
        automl_enabled="TRUE"
    )
    cv_group: int = INT_FIELD(
        value=8,
        default_value=8,
        description="cv group",
        display_name="cv group",
        valid_min=4,
        valid_max=16,
        automl_enabled="TRUE"
    )
    train_iters: int = INT_FIELD(
        value=22,
        default_value=22,
        description="Train Iteration",
        display_name="train iteration",
        valid_min=1,
    )
    valid_iters: int = INT_FIELD(
        value=22,
        default_value=22,
        description="Validation Iteration",
        display_name="Validation iteration",
        valid_min=1,
    )
    volume_dim: int = INT_FIELD(
        value=32,
        default_value=32,
        description="Volume dimension",
        display_name="volume dimension",
        valid_min=16,
        valid_max=64,
        automl_enabled="TRUE"
    )
    low_memory: int = INT_FIELD(
        value=0,
        default_value=0,
        description="reduce memory usage",
        display_name="reduce memory usage",
        valid_min=0,
        valid_max=4,
    )
    mixed_precision: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Mixed Precision Training",
        description="""A flag specifying whether to use mixed precision training""",
    )
    gwc_feature_normalize: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="GWC feature L2-normalize",
        description="""L2-normalize features before group-wise correlation in
        build_gwc_volume. Matches FFS bp2 training-time intent; without it
        activations accumulate to ~10^3-10^4 magnitude causing physically
        invalid disparity output. FoundationStereo path is unaffected because
        it does not consume this field; FastFoundationStereo reads it from
        ``self.args.gwc_feature_normalize``.""",
    )
    n_gru_layers: int = INT_FIELD(
        value=3,
        valid_min=1,
        valid_max=3,
        description="The number of hidden GRU levels",
        display_name="number of hidden GRU levels"
    )
    corr_levels: int = INT_FIELD(
        value=2,
        valid_min=1,
        valid_max=2,
        description="The number of levels in the correlation pyramid",
        display_name="number of correlation pyramid levels"
    )
    n_downsample: int = INT_FIELD(
        value=2,
        valid_min=1,
        valid_max=2,
        description="resolution of the disparity field (1/2^K)",
        display_name="disparity field resoultion"
    )
    motion_encoder_widths: List[int] = LIST_FIELD(
        arrList=[256, 256, 64, 64],
        description="BasicMotionEncoder convc1/convc2/convd1/convd2 outputs.",
        display_name="Motion encoder widths"
    )
    motion_encoder_final: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        description="BasicMotionEncoder final conv output. None -> hidden_dims[0]-1.",
        display_name="Motion encoder final"
    )
    gru_hidden: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        description="SelectiveConvGRU hidden width. None -> hidden_dim arg.",
        display_name="GRU hidden"
    )
    gru_gating_conv_widths: Optional[List[int]] = LIST_FIELD(
        arrList=None,
        default_type=None,
        description="SelectiveConvGRU conv0/conv1 outputs. None -> [input_dim, input_dim+hidden].",
        display_name="GRU gating conv widths"
    )
    disp_head_input_dim: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        description="DispHead input width. None -> hidden_dim.",
        display_name="DispHead input"
    )
    disp_head_intermediate: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        description="DispHead intermediate width. None -> input_dim.",
        display_name="DispHead intermediate"
    )
    disp_head_pwconv1_widths: Optional[List[int]] = LIST_FIELD(
        arrList=None,
        default_type=None,
        description="DispHead EdgeNext per-encoder pwconv1 widths. None -> [4*intermediate]*2.",
        display_name="DispHead pwconv1 widths"
    )
    mask_widths: List[int] = LIST_FIELD(
        arrList=[64, 32],
        description="BasicSelectiveMultiUpdateBlock mask layer 0/2 widths.",
        display_name="Mask widths"
    )
    stem_2_widths: List[int] = LIST_FIELD(
        arrList=[32, 32],
        description="stem_2 two-layer widths.",
        display_name="stem_2 widths"
    )
    spx_2_gru_widths: List[int] = LIST_FIELD(
        arrList=[32, 32, 32, 32],
        description=(
            "Four-element [in, mid, rem, out] for spx_2_gru Conv2xDownScale. "
            "in = mask_feat_4 channels, mid = conv1 output, rem = stem_2x "
            "channels, out = conv2 output. Default reproduces TAO's "
            "Conv2x(32, 32) + out_channels*2 formula."
        ),
        display_name="spx_2_gru widths"
    )
    spx_gru_out: int = INT_FIELD(
        value=9,
        default_value=9,
        description="spx_gru ConvTranspose output channels.",
        display_name="spx_gru out"
    )
    classifier_mid: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        description="Classifier mid-layer width. None -> volume_dim//2.",
        display_name="Classifier mid"
    )
    cnet_conv04_widths: Optional[List[int]] = LIST_FIELD(
        arrList=None,
        default_type=None,
        description=(
            "Per-head output widths for ContextNetSharedBackbone.conv04 "
            "(net_init head, inp_init head). None -> [hidden_dims[0], "
            "hidden_dims[0]] (TAO default)."
        ),
        display_name="cnet conv04 widths"
    )
    cam_mid_channels: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        description=(
            "Hidden dimension for ChannelAttentionEnhancement.fc. None -> "
            "in_planes // 16 (TAO default). FFS commercial ckpt uses 8."
        ),
        display_name="CAM mid channels"
    )
    cost_agg_conv_patch_padding: Optional[List[int]] = LIST_FIELD(
        arrList=None,
        default_type=None,
        description=(
            "Padding for HourGlass.conv_patch Conv3d. None -> [1, 1, 0] "
            "(TAO default). Pass [0, 0, 0] to match FFS training repo."
        ),
        display_name="cost_agg conv_patch padding"
    )
    concat_channel: int = INT_FIELD(
        value=24,
        default_value=24,
        description=(
            "Concat-volume channel count. bp2 ckpt invariant — changing "
            "breaks ckpt key-shape match."
        ),
        display_name="concat channel"
    )
    encoder: str = STR_FIELD(
        value="vitl",
        default_value="vitl",
        description="DepthAnythingV2 Encoder options",
        valid_options=",".join([
            "vits", "vitb", "vitl", "vitg"
        ])
    )
    max_disparity: int = INT_FIELD(
        value=416,
        display_name="max disparity",
        description="""
        The maximum disparity of the model used in the training of a stereo model
        """
    )
