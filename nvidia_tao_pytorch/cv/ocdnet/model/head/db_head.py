#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DBHead module."""
import torch
from torch import nn


class DBHead(nn.Module):
    """DBHead class."""

    def __init__(self, in_channels, enlarge_feature_map_size=False, k=50, **kwargs):
        """Initialize."""
        super().__init__()
        self.k = k
        self.enlarge_feature_map_size = enlarge_feature_map_size

        binarize = []
        binarize += [
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        ]
        if not self.enlarge_feature_map_size:
            binarize += [
                nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            ]
        binarize += [
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
            nn.Sigmoid()
        ]
        self.binarize = nn.Sequential(*binarize)
        self.binarize.apply(self.weights_init)

        self.thresh = self._init_thresh(in_channels)
        self.thresh.apply(self.weights_init)

    def forward(self, x):
        """Forward."""
        shrink_logits = self.binarize[:-1](x)
        # probability map
        shrink_maps = self.binarize[-1](shrink_logits)

        if not self.training:
            return shrink_maps
        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = torch.cat((shrink_logits, threshold_maps, binary_maps), dim=1)
        return y

    def weights_init(self, m):
        """Weights init."""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels, serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1

        thresh = []
        thresh += [
            nn.Conv2d(in_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
        ]
        if not self.enlarge_feature_map_size:
            thresh += [
                self._init_upsample(inner_channels // 4, inner_channels // 4, smooth=smooth, bias=bias),
                nn.BatchNorm2d(inner_channels // 4),
                nn.ReLU(inplace=True),
            ]
        thresh += [
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid()
        ]
        return nn.Sequential(*thresh)

    def _init_upsample(self, in_channels, out_channels, smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=True))
            return nn.Sequential(module_list)
        return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def step_function(self, x, y):
        """Differentiable binarization function."""
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
