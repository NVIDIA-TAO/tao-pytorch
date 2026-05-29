# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EfficientViT backbone for RT-DETR."""

import torch
import torch.utils.checkpoint as checkpoint
from torch import nn

from nvidia_tao_pytorch.cv.backbone_v2.efficientvit import EfficientViT, EfficientViTLarge
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.registry import RTDETR_BACKBONE_REGISTRY


class EfficientViTFPN(EfficientViT):
    """EfficientViT FPN model."""

    def __init__(self, return_idx=[1, 2, 3], **kwargs):
        """Initialize an EfficientViT backbone for RT-DETR."""
        width_list = kwargs.get("width_list", None)
        assert width_list is not None, "width_list should be provided"
        super().__init__(**kwargs)
        self.return_idx = return_idx
        self.out_channels = width_list[2:]

        # add a norm layer for each output
        for idx in return_idx:
            layer = nn.LayerNorm(width_list[1:][idx])
            layer_name = f"norm{idx}"
            self.add_module(layer_name, layer)

    def forward_feature_pyramid(self, x: torch.Tensor):
        """Forward function."""
        outs = []
        x = self.input_stem(x)
        for idx, stage in enumerate(self.stages):
            # Disable activation checkpointing during ONNX export
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = stage(x)
            else:
                x = checkpoint.checkpoint(stage, x)
            if idx in self.return_idx:
                norm_layer = getattr(self, f"norm{idx}")
                out = norm_layer(x.permute(0, 2, 3, 1).contiguous())
                outs.append(out.permute(0, 3, 1, 2).contiguous())
        return outs


class EfficientViTLargeFPN(EfficientViTLarge):
    """EfficientViT Large FPN model."""

    def __init__(self, return_idx=[1, 2, 3], **kwargs):
        """Initialize an EfficientViT Large backbone for RT-DETR."""
        width_list = kwargs.get("width_list", None)
        super().__init__(**kwargs)
        self.return_idx = return_idx
        self.out_channels = width_list[2:]

        # add a norm layer for each output
        for idx in return_idx:
            layer = nn.LayerNorm(width_list[1:][idx])
            layer_name = f"norm{idx}"
            self.add_module(layer_name, layer)

    def forward_feature_pyramid(self, x: torch.Tensor):
        """Forward function."""
        outs = []
        x = self.stages[0](x)
        for idx, stage in enumerate(self.stages[1:]):
            # Disable activation checkpointing during ONNX export
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = stage(x)
            else:
                x = checkpoint.checkpoint(stage, x)
            if idx in self.return_idx:
                norm_layer = getattr(self, f"norm{idx}")
                out = norm_layer(x.permute(0, 2, 3, 1).contiguous())
                outs.append(out.permute(0, 3, 1, 2).contiguous())
        return outs


@RTDETR_BACKBONE_REGISTRY.register()
def efficientvit_b0(out_indices=[1, 2, 3], **kwargs):
    """EfficientViT-B0 model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EfficientViTFPN(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        return_idx=out_indices,
        num_classes=0,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def efficientvit_b1(out_indices=[1, 2, 3], **kwargs):
    """EfficientViT-B1 model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EfficientViTFPN(
        width_list=[16, 32, 64, 128, 256],
        depth_list=[1, 2, 3, 3, 4],
        dim=16,
        return_idx=out_indices,
        num_classes=0,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def efficientvit_b2(out_indices=[1, 2, 3], **kwargs):
    """EfficientViT-B2 model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EfficientViTFPN(
        width_list=[24, 48, 96, 192, 384],
        depth_list=[1, 3, 4, 4, 6],
        dim=32,
        return_idx=out_indices,
        num_classes=0,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def efficientvit_b3(out_indices=[1, 2, 3], **kwargs):
    """EfficientViT-B3 model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EfficientViTFPN(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        dim=32,
        return_idx=out_indices,
        num_classes=0,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def efficientvit_l0(out_indices=[1, 2, 3], **kwargs):
    """EfficientViT-L0 model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EfficientViTLargeFPN(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 4, 4],
        return_idx=out_indices,
        num_classes=0,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def efficientvit_l1(out_indices=[1, 2, 3], **kwargs):
    """EfficientViT-L1 model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EfficientViTLargeFPN(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 6, 6],
        return_idx=out_indices,
        num_classes=0,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def efficientvit_l2(out_indices=[1, 2, 3], **kwargs):
    """EfficientViT-L2 model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EfficientViTLargeFPN(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 2, 2, 8, 8],
        return_idx=out_indices,
        num_classes=0,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def efficientvit_l3(out_indices=[1, 2, 3], **kwargs):
    """EfficientViT-L3 model.

    Args:
        out_indices (list): List of block indices to return as feature
    """
    return EfficientViTLargeFPN(
        width_list=[64, 128, 256, 512, 1024],
        depth_list=[1, 2, 2, 8, 8],
        return_idx=out_indices,
        num_classes=0,
        **kwargs,
    )
