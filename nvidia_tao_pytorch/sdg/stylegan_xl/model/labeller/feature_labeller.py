# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" BigDatasetGAN Feature Labeller. """

import torch
from torch import nn


def normalization(channels):
    """
    Make a standard normalization layer.

    Args:
        channels (int): Number of input channels.

    Returns:
        nn.Module: A normalization layer.
    """
    return nn.GroupNorm(32, channels)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution Layer."""

    def __init__(self, nin, nout, kernel_size=3, padding=1, bias=False):
        """Initializes the DepthwiseSeparableConv layer.

        Args:
            nin (int): Number of input channels.
            nout (int): Number of output channels.
            kernel_size (int, optional): Size of the convolution kernel. Default is 3.
            padding (int, optional): Padding added to all four sides of the input. Default is 1.
            bias (bool, optional): If True, adds a learnable bias to the output. Default is False.
        """
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying depthwise and pointwise convolutions.
        """
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class FeatureFuseLayer(nn.Module):
    """Feature Fusion Layer."""

    def __init__(self, feat_dim, out_dim, use_fuse=False, use_upsample=True):
        """Initializes the FeatureFuseLayer.

        Args:
            feat_dim (int): Dimension of the input features.
            out_dim (int): Dimension of the output features.
            use_fuse (bool, optional): If True, use feature fusion. Default is False.
            use_upsample (bool, optional): If True, use upsampling. Default is True.
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.use_fuse = use_fuse
        self.use_upsample = use_upsample

        self.conv1x1 = nn.Conv2d(feat_dim, out_dim, kernel_size=1, bias=False)
        if use_fuse:
            self.conv_fuse = nn.Sequential(
                normalization(out_dim * 2),
                nn.SiLU(),
                DepthwiseSeparableConv(out_dim * 2, out_dim, kernel_size=3, padding=1),
                # nn.Conv2d(out_dim*2, out_dim, kernel_size=3, padding=1)
            )
        if use_upsample:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, cur_feat, prev_feat=None):
        """Forward function.

        Args:
            cur_feat (torch.Tensor): Current feature tensor.
            prev_feat (Tensor, optional): Previous feature tensor. Default is None.

        Returns:
            torch.Tensor: Fused feature tensor.
        """
        cur_feat = cur_feat.to(torch.float)
        cur_feat = self.conv1x1(cur_feat)
        if prev_feat is not None:
            prev_feat = prev_feat.to(torch.float)
            concat_feat = torch.cat([cur_feat, prev_feat], dim=1)
            fuse_feat = self.conv_fuse(concat_feat)
        else:
            fuse_feat = cur_feat

        if self.use_upsample:
            fuse_feat = self.upsample(fuse_feat)

        return fuse_feat


class FeatureLabellerSGXL(nn.Module):
    """Feature Labeller on top of StyleGAN-XL."""

    def __init__(self, n_class, in_channels, dropout_ratio=0.1, feat_fuse_dim=256):
        """Initializes the FeatureLabellerSGXL.

        Args:
            n_class (int): Number of output classes.
            in_channels (List[int]): List of input channel dimensions.
            dropout_ratio (float, optional): Dropout ratio. Default is 0.1.
            feat_fuse_dim (int, optional): Dimension of the fused features. Default is 256.
        """
        super().__init__()
        self.n_class = n_class
        self.in_channels = in_channels
        self.dropout_ratio = dropout_ratio

        self.feat_fuse_modules = nn.ModuleList()

        for i in range(len(in_channels)):
            if i == 0:
                self.feat_fuse_modules.append(
                    FeatureFuseLayer(in_channels[i], feat_fuse_dim, use_fuse=False, use_upsample=True)
                )
            elif i == len(in_channels) - 1:
                self.feat_fuse_modules.append(
                    FeatureFuseLayer(in_channels[i], feat_fuse_dim, use_fuse=True, use_upsample=False)
                )
            else:
                self.feat_fuse_modules.append(
                    FeatureFuseLayer(in_channels[i], feat_fuse_dim, use_fuse=True, use_upsample=True)
                )

        self.conv_seg = nn.Sequential(
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(feat_fuse_dim, n_class, kernel_size=1)
        )
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, feat_list):
        """Forward function.

        Args:
            feat_list (List[Tensor]): List of feature tensors.

        Returns:
            torch.Tensor: Segmentation logits.
        """
        prev_feat = None
        for i, feat in enumerate(feat_list):
            feat_fuse = self.feat_fuse_modules[i](feat, prev_feat)
            prev_feat = feat_fuse

        seg_logits = self.conv_seg(feat_fuse)

        seg_logits = self.upsample(seg_logits)

        return seg_logits
