# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backbone ResNet model definition."""

from timm.models.resnet import BasicBlock, Bottleneck

from nvidia_tao_pytorch.cv.backbone_v2.resnet import ResNet
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.registry import RTDETR_BACKBONE_REGISTRY


class ResNetFPN(ResNet):
    """ResNet FPN module."""

    def __init__(self, out_channels, return_idx=[0, 1, 2, 3], **kwargs):
        """Init"""
        super().__init__(**kwargs)

        self.return_idx = return_idx
        _out_strides = [4, 8, 16, 32]
        self.out_channels = [out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

    def forward_feature_pyramid(self, x):
        """Forward"""
        x = super().forward_intermediates(x, indices=4, intermediates_only=True)
        out = []
        for i in self.return_idx:
            out.append(x[i])
        return out


@RTDETR_BACKBONE_REGISTRY.register()
def resnet_18(out_indices=[1, 2, 3], **kwargs):
    """Resnet-18 model.
    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ResNetFPN(
        num_classes=0,
        out_channels=[64, 128, 256, 512],
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        return_idx=out_indices,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def resnet_34(out_indices=[1, 2, 3], **kwargs):
    """Resnet-34 model.
    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ResNetFPN(
        num_classes=0,
        out_channels=[64, 128, 256, 512],
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        return_idx=out_indices,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def resnet_50(out_indices=[1, 2, 3], **kwargs):
    """ResNet-50 model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ResNetFPN(
        num_classes=0,
        out_channels=[256, 512, 1024, 2048],
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        return_idx=out_indices,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def resnet_101(out_indices=[1, 2, 3], **kwargs):
    """ResNet-101 model.
    Args:
        out_indices (list): List of block indices to return as feature
    """
    return ResNetFPN(
        num_classes=0,
        out_channels=[256, 512, 1024, 2048],
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        return_idx=out_indices,
        **kwargs,
    )
