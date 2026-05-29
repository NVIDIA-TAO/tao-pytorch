# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Depth AnythingV2 Block module."""

import torch.nn as nn


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    """
    This function creates a set of convolutional layers that serve as scratch
    layers for the DPT head. These layers process multi-scale feature maps
    from the backbone network and prepare them for depth prediction.

    Args:
        in_shape (list): List of input channel dimensions for each scale level.
            Length indicates the number of scales (typically 3 or 4).
        out_shape (int): Base number of output channels for each layer.
        groups (int, optional): Number of groups for grouped convolutions.
            Defaults to 1 (standard convolution).
        expand (bool, optional): Whether to use expanding channel dimensions
            across scales. If True, later scales get more channels.
            Defaults to False.

    Returns:
        nn.Module: A module containing scratch layers for each scale level.
            The module has attributes like layer1_rn, layer2_rn, etc.

    Note:
        - If expand=True, channel dimensions increase by factors of 2, 4, 8
        - Each layer uses 3x3 convolutions with padding=1 and stride=1
        - All convolutions use bias=False for better training stability
    """
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch


class ResidualConvUnit(nn.Module):
    """
    This module implements a residual convolution unit that consists of two
    consecutive 3x3 convolutions with optional batch normalization and
    activation functions. It adds the input to the output through a skip
    connection, following the residual learning paradigm.

    Attributes:
        conv1 (nn.Conv2d): First 3x3 convolution layer.
        conv2 (nn.Conv2d): Second 3x3 convolution layer.
        bn1 (nn.BatchNorm2d, optional): First batch normalization layer.
        bn2 (nn.BatchNorm2d, optional): Second batch normalization layer.
        activation (callable): Activation function applied after each convolution.
        skip_add (nn.quantized.FloatFunctional): Quantized addition operation.
    """

    def __init__(self, features, activation, bn):
        """Initialize the ResidualConvUnit.

        Args:
            features (int): Number of input and output channels for the convolutions.
            activation (callable): Activation function to apply after convolutions.
            bn (bool): Whether to use batch normalization after convolutions.
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn is True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass through the residual convolution unit.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of the same shape as input, with
                residual connection added.

        Note:
            The forward pass follows the pattern: activation -> conv1 -> bn1 ->
            activation -> conv2 -> bn2 -> add_residual
        """
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn is True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn is True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """
    This module fuses features from multiple scales using residual convolution
    units and optional upsampling. It's designed to integrate information
    from different resolution levels in a hierarchical manner.

    Attributes:
        out_conv (nn.Conv2d): Final 1x1 convolution for channel adjustment.
        resConfUnit1 (ResidualConvUnit): First residual convolution unit.
        resConfUnit2 (ResidualConvUnit): Second residual convolution unit.
        skip_add (nn.quantized.FloatFunctional): Quantized addition operation.
        deconv (bool): Whether to use deconvolution for upsampling.
        align_corners (bool): Alignment mode for interpolation.
        expand (bool): Whether to expand output channels.
        size (tuple, optional): Target size for upsampling.
    """

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None
    ):
        """Initialize the FeatureFusionBlock.

        Args:
            features (int): Number of input and output features.
            activation (callable): Activation function for residual units.
            deconv (bool, optional): Whether to use deconvolution for upsampling.
                Defaults to False.
            bn (bool, optional): Whether to use batch normalization in residual units.
                Defaults to False.
            expand (bool, optional): Whether to expand output channels by factor of 2.
                Defaults to False.
            align_corners (bool, optional): Alignment mode for bilinear interpolation.
                Defaults to True.
            size (tuple, optional): Target size for upsampling (H, W).
                Defaults to None.
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand is True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass through the feature fusion block.

        Args:
            *xs: Variable number of input tensors from different scales.
                The first tensor is the primary input, subsequent tensors
                are fused with it.
            size (tuple, optional): Override size for upsampling. If provided,
                overrides the size set during initialization.

        Returns:
            torch.Tensor: Fused and upsampled feature tensor.

        Note:
            - If only one input is provided, only residual processing is applied
            - If two inputs are provided, the second is processed and added to the first
            - Upsampling is performed using bilinear interpolation
            - Final 1x1 convolution adjusts the output channels
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)

        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_bn, size=None):
    """
    This function creates a FeatureFusionBlock with commonly used default
    parameters for depth estimation tasks.

    Args:
        features (int): Number of features for the fusion block.
        use_bn (bool): Whether to use batch normalization in residual units.
        size (tuple, optional): Target size for upsampling (H, W).
            Defaults to None.

    Returns:
        FeatureFusionBlock: Configured feature fusion block with:
            - ReLU activation
            - No deconvolution (uses bilinear interpolation)
            - Optional batch normalization
            - No channel expansion
            - Aligned corners for interpolation
    """
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )
