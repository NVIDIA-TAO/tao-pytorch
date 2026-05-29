# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backbone EdgeNeXt model definition."""
import torch
import torch.utils.checkpoint as checkpoint

from nvidia_tao_pytorch.cv.backbone_v2.edgenext import EdgeNeXt, EdgeNeXtBNHS
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.registry import RTDETR_BACKBONE_REGISTRY


class EdgeNeXtFPN(EdgeNeXt):
    """EdgeNeXt Feature Pyramid Network (FPN) backbone for RT-DETR.

    This class extends the EdgeNeXt backbone to provide multi-scale feature extraction
    suitable for object detection tasks. It implements a Feature Pyramid Network structure
    that extracts features at different scales and returns them based on specified indices.

    The backbone processes input images through a series of downsample layers and stages,
    with optional positional embeddings and activation checkpointing for memory efficiency.

    Args:
        return_idx (list, optional): List of stage indices to return as features.
            Default is [0, 1, 2, 3] corresponding to all 4 stages.
        **kwargs: Additional keyword arguments passed to the EdgeNeXt parent class.

    Attributes:
        return_idx (list): Indices of stages to return as output features.
        out_channels (list): Number of output channels for each returned stage.
        out_strides (list): Output stride (downsampling factor) for each returned stage.

    Note:
        The output strides are [4, 8, 16, 32] for stages [0, 1, 2, 3] respectively.
        The actual output channels depend on the EdgeNeXt variant configuration.
    """

    def __init__(self, return_idx=[0, 1, 2, 3], **kwargs):
        """Init"""
        super().__init__(**kwargs)

        self.return_idx = return_idx
        _out_strides = [4, 8, 16, 32]
        _out_channels = self.dims
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

    def forward_feature_pyramid(self, x):
        """Forward"""
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        if self.pos_embd:
            B, _, H, W = x.shape
            x = x + self.pos_embd(B, H, W)
        out = []
        for idx in range(1, 4):
            x = self.downsample_layers[idx](x)
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = self.stages[idx](x)
            else:
                x = checkpoint.checkpoint(self.stages[idx], x)
            out.append(x)
        return out


class EdgeNeXtBNHSFPN(EdgeNeXtBNHS):
    """EdgeNeXtBNHS Feature Pyramid Network (FPN) backbone for RT-DETR.

    This class extends the EdgeNeXt backbone to provide multi-scale feature extraction
    suitable for object detection tasks. It implements a Feature Pyramid Network structure
    that extracts features at different scales and returns them based on specified indices.

    The backbone processes input images through a series of downsample layers and stages,
    with optional positional embeddings and activation checkpointing for memory efficiency.

    Args:
        return_idx (list, optional): List of stage indices to return as features.
            Default is [0, 1, 2, 3] corresponding to all 4 stages.
        **kwargs: Additional keyword arguments passed to the EdgeNeXt parent class.

    Attributes:
        return_idx (list): Indices of stages to return as output features.
        out_channels (list): Number of output channels for each returned stage.
        out_strides (list): Output stride (downsampling factor) for each returned stage.

    Note:
        The output strides are [4, 8, 16, 32] for stages [0, 1, 2, 3] respectively.
        The actual output channels depend on the EdgeNeXt variant configuration.
    """

    def __init__(self, return_idx=[0, 1, 2, 3], **kwargs):
        """Init"""
        super().__init__(**kwargs)

        self.return_idx = return_idx
        _out_strides = [4, 8, 16, 32]
        _out_channels = self.dims
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

    def forward_feature_pyramid(self, x):
        """Forward"""
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        if self.pos_embd:
            B, _, H, W = x.shape
            x = x + self.pos_embd(B, H, W)
        out = []
        for idx in range(1, 4):
            x = self.downsample_layers[idx](x)
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = self.stages[idx](x)
            else:
                x = checkpoint.checkpoint(self.stages[idx], x)
            out.append(x)
        return out


@RTDETR_BACKBONE_REGISTRY.register()
def edgenext_x_small(out_indices=[1, 2, 3], **kwargs):
    """EdgeNeXt-xx-small model.
    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EdgeNeXtFPN(
        depths=[3, 3, 9, 3], dims=[32, 64, 100, 192], expan_ratio=4,
        global_block=[0, 1, 1, 1],
        global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
        use_pos_embd_xca=[False, True, False, False],
        kernel_sizes=[3, 5, 7, 9],
        heads=[4, 4, 4, 4],
        d2_scales=[2, 2, 3, 4],
        return_idx=out_indices,
        num_classes=0,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def edgenext_small(out_indices=[1, 2, 3], **kwargs):
    """EdgeNeXt-small model.
    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EdgeNeXtFPN(
        depths=[3, 3, 9, 3], dims=[48, 96, 160, 304], expan_ratio=4,
        global_block=[0, 1, 1, 1],
        global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
        use_pos_embd_xca=[False, True, False, False],
        kernel_sizes=[3, 5, 7, 9],
        d2_scales=[2, 2, 3, 4],
        return_idx=out_indices,
        num_classes=0,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def edgenext_base(out_indices=[1, 2, 3], **kwargs):
    """EdgeNeXt-base model.
    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EdgeNeXtFPN(
        depths=[3, 3, 9, 3], dims=[80, 160, 288, 584], expan_ratio=4,
        global_block=[0, 1, 1, 1],
        global_block_type=['None', 'SDTA', 'SDTA', 'SDTA'],
        use_pos_embd_xca=[False, True, False, False],
        kernel_sizes=[3, 5, 7, 9],
        d2_scales=[2, 2, 3, 4],
        num_classes=0,
        return_idx=out_indices,
        **kwargs
    )


@RTDETR_BACKBONE_REGISTRY.register()
def edgenext_xx_small_bn_hs(out_indices=[1, 2, 3], **kwargs):
    """EdgeNeXt-xx-small-bn-hs model.
    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EdgeNeXtBNHSFPN(
        depths=[2, 2, 6, 2], dims=[24, 48, 88, 168], expan_ratio=4,
        global_block=[0, 1, 1, 1],
        global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS', 'SDTA_BN_HS'],
        use_pos_embd_xca=[False, True, False, False],
        kernel_sizes=[3, 5, 7, 9],
        heads=[4, 4, 4, 4],
        d2_scales=[2, 2, 3, 4],
        return_idx=out_indices,
        num_classes=0,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def edgenext_x_small_bn_hs(out_indices=[1, 2, 3], **kwargs):
    """EdgeNeXt-x-small-bn-hs model.
    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EdgeNeXtBNHSFPN(
        depths=[3, 3, 9, 3], dims=[32, 64, 100, 192], expan_ratio=4,
        global_block=[0, 1, 1, 1],
        global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS', 'SDTA_BN_HS'],
        use_pos_embd_xca=[False, True, False, False],
        kernel_sizes=[3, 5, 7, 9],
        heads=[4, 4, 4, 4],
        d2_scales=[2, 2, 3, 4],
        return_idx=out_indices,
        num_classes=0,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def edgenext_small_bn_hs(out_indices=[1, 2, 3], **kwargs):
    """EdgeNeXt-small-bn-hs model.
    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EdgeNeXtBNHSFPN(
        depths=[3, 3, 9, 3], dims=[48, 96, 160, 304], expan_ratio=4,
        global_block=[0, 1, 1, 1],
        global_block_type=['None', 'SDTA_BN_HS', 'SDTA_BN_HS', 'SDTA_BN_HS'],
        use_pos_embd_xca=[False, True, False, False],
        kernel_sizes=[3, 5, 7, 9],
        d2_scales=[2, 2, 3, 4],
        return_idx=out_indices,
        num_classes=0,
        **kwargs,
    )
