# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ConvNext backbone for RT-DETR."""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from nvidia_tao_pytorch.cv.backbone_v2.convnext_v2 import ConvNeXtV2 as ConvNeXt
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.registry import RTDETR_BACKBONE_REGISTRY


class ConvNeXtFPN(ConvNeXt):
    """ConvNeXtFPN."""

    def __init__(self, return_idx=[1, 2, 3], out_channels=[512, 1024, 2048], **kwargs):
        """Initialize ConvNeXtFPN.

        Args:
            in_chans (int): Number of input image channels. Default: `3`.
            num_classes (int): Number of classes for classification head. Default: `1000`.
            depths (tuple(int)): Number of blocks at each stage. Default: `[3, 3, 9, 3]`.
            dims (tuple(int)): Feature dimension at each stage. Default: `[96, 192, 384, 768]`.
            use_grn (bool): Whether to use `GlobalResponseNorm`. Default: `True`.
            layer_scale_init_value (float): Init value for Layer Scale. Default: `0.0`.
            drop_path_rate (float): Stochastic depth rate. Default: `0`.
            head_init_scale (float): Init scaling value for classifier weights and biases. Default: `1`.
            export_pre_logits (bool): Whether to export the pre_logits features of the model. Default: `False`.
            activation_checkpoint (bool): Whether to use activation checkpointing. Default: `False`.
            return_idx (list): List of block indices to return as feature. Default: `[1, 2, 3]`.
            out_channels (list): List of output channels. Default: `[512, 1024, 2048]`.
        """
        super().__init__(**kwargs)
        self.return_idx = return_idx
        self.out_channels = out_channels
        assert len(self.return_idx) == 3, f"ConvNext only supports num_feature_levels == 3, Got {len(self.return_idx)}"

        self.conv_512 = nn.Conv2d(
            self.dims[self.return_idx[0]], self.out_channels[0], kernel_size=3, stride=1, padding=1
        )
        self.conv_1024 = nn.Conv2d(
            self.dims[self.return_idx[1]], self.out_channels[1], kernel_size=3, stride=1, padding=1
        )
        self.conv_2048 = nn.Conv2d(
            self.dims[self.return_idx[2]], self.out_channels[2], kernel_size=3, stride=1, padding=1
        )
        self.conv_upsample = [self.conv_512, self.conv_1024, self.conv_2048]

        self.apply(self._init_weights)

    def forward_feature_pyramid(self, x):
        """Forward pass through the backbone to extract intermediate feature maps."""
        outs = []
        conv_upsample_idx = 0
        for idx in range(self.num_stages):
            x = self.downsample_layers[idx](x)
            # Disable activation checkpointing during ONNX export
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = self.stages[idx](x)
            else:
                x = checkpoint.checkpoint(self.stages[idx], x)
            if idx in self.return_idx:
                feature_pyramid = self.conv_upsample[conv_upsample_idx](x)
                outs.append(feature_pyramid)
                conv_upsample_idx += 1
        return outs


@RTDETR_BACKBONE_REGISTRY.register()
def convnext_tiny(out_indices=[1, 2, 3], **kwargs):
    """ConvNext-Tiny model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ConvNeXtFPN(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        use_grn=False,
        layer_scale_init_value=1e-6,
        return_idx=out_indices,
        num_classes=0,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def convnext_small(out_indices=[1, 2, 3], **kwargs):
    """ConvNext-Small model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ConvNeXtFPN(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        use_grn=False,
        layer_scale_init_value=1e-6,
        return_idx=out_indices,
        num_classes=0,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def convnext_base(out_indices=[1, 2, 3], **kwargs):
    """ConvNext-Base model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ConvNeXtFPN(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        use_grn=False,
        layer_scale_init_value=1e-6,
        return_idx=out_indices,
        num_classes=0,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def convnext_large(out_indices=[1, 2, 3], **kwargs):
    """ConvNext-Large model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ConvNeXtFPN(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        use_grn=False,
        layer_scale_init_value=1e-6,
        return_idx=out_indices,
        num_classes=0,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def convnext_xlarge(out_indices=[1, 2, 3], **kwargs):
    """ConvNext-XLarge model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ConvNeXtFPN(
        depths=[3, 3, 27, 3],
        dims=[256, 512, 1024, 2048],
        use_grn=False,
        layer_scale_init_value=1e-6,
        return_idx=out_indices,
        num_classes=0,
        **kwargs,
    )
