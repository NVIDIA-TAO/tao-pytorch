# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLP Segformer Head."""

from torch import nn
import torch
from nvidia_tao_pytorch.cv.segformer.model.segformer_utils import resize


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


class TAOSegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=False,
                 in_channels=[64, 128, 320, 512], embedding_dim=768, output_nc=2,
                 feature_strides=[4, 8, 16, 32], model_name='FANHybrid'):
        """Init Module."""
        super().__init__()
        assert len(feature_strides) == len(in_channels), "The number of feature strides:{} should be equal to number of channels: {}".format(feature_strides, len(in_channels))
        assert min(feature_strides) == feature_strides[0], "Minimum of feature strides is not supported."

        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        self.model_name = model_name
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * 4, out_channels=self.embedding_dim, kernel_size=1, bias=True),
            nn.BatchNorm2d(self.embedding_dim),
            nn.ReLU(inplace=True),
        )

        # dropout defined in mmseg basedecoder
        self.dropout = nn.Dropout2d(p=0.1)

        self.linear_pred = nn.Conv2d(self.embedding_dim, self.output_nc, kernel_size=1)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):
        """Forward."""
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        n, _, _, _ = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
