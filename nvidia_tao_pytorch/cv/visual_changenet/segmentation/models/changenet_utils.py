# Copyright (c) 2023 Chaminda Bandara

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# Original source taken from https://github.com/wgcban/ChangeFormer
#
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Visual ChangeNet Model utils"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple, trunc_normal_


class UpsampleConvLayer(torch.nn.Module):
    """Upsample Conv Layer"""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        """Initialize UpsampleConvLayer class"""
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        """Forward functions"""
        out = self.conv2d(x)
        return out


class ConvLayer(nn.Module):
    """ConvLayer"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        """Initialize ConvLayer class"""
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        """Forward functions"""
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    """Residual Block"""

    def __init__(self, channels):
        """Initialize ResidualBlock class"""
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward functions"""
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        """Initialize OverlapPatchEmbed class"""
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """Forward functions"""
        # pdb.set_trace()
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


def resize(input,  # pylint: disable=W0622
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    """Resize the input tensor using the specified method."""
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:  # pylint: disable=R0916
                if ((output_h > 1 and output_w > 1 and input_h > 1 and  # pylint: disable=R0916
                     input_w > 1) and (output_h - 1) % (input_h - 1) and (output_w - 1) % (input_w - 1)):  # pylint: disable=R0916
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


# Transformer Decoder
class MLP(nn.Module):
    """Linear Embedding"""

    def __init__(self, input_dim=2048, embed_dim=768):
        """Initialize MLP class"""
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.input_dim = input_dim
        self.embed_dim = embed_dim

    def forward(self, x):
        """Forward functions"""
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


# Difference module
def conv_diff(in_channels, out_channels):
    """Learnable difference module"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )


def make_prediction(in_channels, out_channels):
    """Intermediate prediction module"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )
