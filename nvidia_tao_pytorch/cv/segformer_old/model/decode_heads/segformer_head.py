# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLP Segformer Head."""

from torch import nn
import torch

from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from mmseg.models.utils.wrappers import resize
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


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


@MODELS.register_module()
class TAOSegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, feature_strides, decoder_params, img_shape, phase, **kwargs):
        """Init Module."""
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels), "The number of feature strides:{} should be equal to number of channels: {}".format(feature_strides, len(self.in_channels))
        assert min(feature_strides) == feature_strides[0], "Minimum of feature strides is not supported."
        self.phase = phase
        self.img_shape = img_shape
        self.feature_strides = feature_strides
        self.export = False

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # @seanf: the decoder_params that set us apart from mmseg aren't used in our BaseDecodeHead, so we can use mmseg's
        # decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim, export=self.export)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim, export=self.export)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim, export=self.export)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim, export=self.export)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=self.norm_cfg
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # @seanf note for future
        # conv_seg exists in the decode head but is useless as linear_pred takes its place
        # However, it's not deleted, so mmengine will throw an error about it being unused,
        # hence requiring find_unused_parameters which will be much slower than normal training
        # On the other hand, we still need the variable lienar_pred to exist because checkpoints load its state
        # So, both need to exist, but linear_pred should be used, and conv_seg should be ignored

    def forward(self, inputs):
        """Forward."""
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        # MLP decoder on C1-C4 #
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

        if self.phase == 'export':
            x = resize(input=x, size=self.img_shape, mode='bilinear', align_corners=False)
            x = self._postprocess_result(x)
        return x

    def _postprocess_result(self, seg_logits):

        _, C, _, _ = seg_logits.shape

        if C > 1:
            seg_logits = seg_logits.argmax(dim=1, keepdim=True)
        else:
            seg_logits = seg_logits.sigmoid()
            seg_logits = (seg_logits >
                          self.decode_head.threshold).to(seg_logits)

        return seg_logits
