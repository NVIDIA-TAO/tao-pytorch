# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blocks for NVPanoptix3D model."""

import torch.nn as nn
import torch.nn.functional as F


class ProjectionBlock(nn.Module):
    """Project 2D features to a target spatial resolution.

    The block first downsamples the input with a stride-2 3x3 convolution, then
    upsamples to a requested size via bilinear interpolation, and finally applies
    a 1x1 convolution to produce the output channels. This aligns feature maps
    across scales while keeping a lightweight computation footprint.
    """

    def __init__(self, in_feature, out_feature):
        """Initialize the projection block.

        Args:
            in_feature: Number of input channels.
            out_feature: Number of output channels.
        """
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
        self.conv_block2 = nn.Conv2d(
            out_feature, out_feature,
            kernel_size=1, stride=1,
            padding=0
        )

    def forward(self, x, target_size):
        """Project a feature map to the requested spatial size.

        Args:
            x: Input feature tensor of shape (N, C_in, H, W).
            target_size: Output spatial size for F.interpolate (e.g. (H_out, W_out)).

        Returns:
            A tensor of shape (N, C_out, H_out, W_out).
        """
        x = self.conv_block1(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = self.conv_block2(x)
        return x


class ConvBlock(nn.Module):
    """Single-stage downsampling block for depth features.

    Applies a 3x3 stride-2 convolution followed by batch norm and ReLU, reducing
    spatial resolution while preserving the channel dimension.
    """

    def __init__(self, in_feature, out_feature):
        """Initialize the conv block.

        Args:
            in_feature: Number of input channels.
            out_feature: Number of output channels.
        """
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )

    def forward(self, x):
        """Forward pass."""
        return self.conv_block(x)


class DepthProjector(nn.Module):
    """Project multi-scale depth features to match target image scales.

    Each input level is passed through a small convolutional projection and then
    resized to the requested spatial shape. The module returns the final depth
    feature (highest level) and a list of projected features aligned with the
    provided size_list.
    """

    def __init__(
            self,
            in_channels: int = 256,
            out_channels: int = 256,
            num_proj_convs: int = 4,
            **kwargs
    ):
        """Initialize the depth projector.

        Args:
            in_channels: Number of channels in the input depth feature maps.
            out_channels: Number of channels produced per projected feature map.
            num_proj_convs: Number of projection stages (expected feature levels).
            **kwargs: Unused extra keyword arguments (kept for API compatibility).
        """
        super(DepthProjector, self).__init__()
        self.proj_convs1 = nn.ModuleList([
            ConvBlock(in_channels, in_channels) for _ in range(num_proj_convs)
        ])
        self.proj_convs2 = nn.ModuleList([
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, stride=1,
                padding=0
            ) for _ in range(num_proj_convs)
        ])

    def forward(self, depth_features, depth_feature_shape, size_list):
        """Project multi-level depth features to requested spatial sizes.

        Args:
            depth_features: Sequence of depth feature tensors, one per scale.
            depth_feature_shape: Spatial shape (H, W) of the final depth feature.
            size_list: List of target spatial shapes. Iterated in reverse order.

        Returns:
            A tuple (final_depth_feature, projected_features) where:
            - final_depth_feature is depth_features[-1].
            - projected_features is a list of projected tensors aligned to
              size_list order.
        """
        output_list = []
        size_list.append(depth_feature_shape)
        for i, (_, feat_shape) in enumerate(zip(
            self.proj_convs1,
            size_list[::-1]
        )):
            feat = depth_features[i]
            output = self.proj_convs1[i](feat)
            output = F.interpolate(output, feat_shape, mode="bilinear", align_corners=False)
            output = self.proj_convs2[i](output)
            output_list.append(output)

        return depth_features[-1], output_list[1:][::-1]
