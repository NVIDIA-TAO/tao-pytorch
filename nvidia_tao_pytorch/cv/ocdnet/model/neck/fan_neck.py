# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FAN Neck module."""
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.core.modules.conv_module import ConvModule


def resize(input,   # pylint: disable=W0622
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    """ Resize Function."""
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:   # pylint: disable=R0916
                if ((output_h > 1 and output_w > 1 and input_h > 1 and  # pylint: disable=R0916
                     input_w > 1) and (output_h - 1) % (input_h - 1) and (output_w - 1) % (input_w - 1)):  # pylint: disable=R0916
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)

    return F.interpolate(input, size, scale_factor, mode, align_corners)


class MLP(nn.Module):
    """Linear Embedding."""

    def __init__(self, input_dim=2048, embed_dim=768, export=False):
        """Init."""
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.export = export

    def forward(self, x):
        """Forward."""
        if self.export:
            _, C, H, W = x.shape
            x = x.view(-1, C, H * W).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class FANNeck(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, in_channels, feature_strides=[4, 8, 16, 32], dropout_ratio=0.1, embedding_dim=768, export=False, **kwargs):
        """Init Module."""
        super(FANNeck, self).__init__()
        self.in_channels = in_channels
        assert len(feature_strides) == len(self.in_channels), "The number of feature strides:{} should be equal to number of channels: {}".format(feature_strides, len(self.in_channels))
        assert min(feature_strides) == feature_strides[0], "Minimum of feature strides is not supported."
        self.feature_strides = feature_strides
        self.export = export
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim, export=self.export)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim, export=self.export)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim, export=self.export)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim, export=self.export)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm='SyncBN'
        )
        self.out_channels = embedding_dim

    def forward(self, x):
        """Forward."""
        c1, c2, c3, c4 = x
        neck_out_size = c1.size()[2:]
        # MLP decoder on C1-C4 #
        n, _, _, _ = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=neck_out_size, mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=neck_out_size, mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # disable autocast to address "nan" error when AMP training
        with torch.amp.autocast(enabled=False, device_type="cuda"):
            _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
            _c3 = resize(_c3, size=neck_out_size, mode='bilinear', align_corners=False)
            _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        return x
