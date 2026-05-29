# Copyright (c) OpenMMLab. All rights reserved.
#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" ConvModule """

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        norm (str): Normalization layer. Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]] = 0,
                 norm: Optional[str] = None
                 ):
        """Initialize ConvModule"""
        super().__init__()

        # https://mmcv.readthedocs.io/en/2.x/api/generated/mmcv.cnn.ConvModule.html
        # Bugfix: MAL vit model was trained using mmcv where there are bias terms in the conv layer
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=(norm is None))

        if norm:
            # @seanf note: this was refactored from an mmcv implementation
            # SyncBN is used by the FAN neck in OCDNet, so it's included here
            # This module can be extended to support more if need be
            if norm == "SyncBN":
                self.norm = torch.nn.SyncBatchNorm(out_channels)
                for param in self.norm.parameters():
                    param.requires_grad = True
            else:
                raise NotImplementedError(f"{norm} is not a supported normalization layer.")
        else:
            self.norm = None

        self.activate = nn.ReLU(inplace=True)

        self.init_weights()

    def init_weights(self):
        """Init model weights"""
        if not hasattr(self.conv, 'init_weights'):
            kaiming_init(self.conv, a=0, nonlinearity='relu')
        if self.norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        x = self.activate(x)
        return x


def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    """Kaiming initialization of weights"""
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    """Constant initialization of weights"""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
