# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# -------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2021 Facebook, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Modified by Daquan Zhou
# -------------------------------------------------------------------------
""" ConvNeXt

Paper: `A ConvNet for the 2020s` - https://arxiv.org/pdf/2201.03545.pdf

Original code and weights from https://github.com/facebookresearch/ConvNeXt, original copyright below

Modifications and additions for timm hacked together by / Copyright 2022, Ross Wightman
"""

from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import named_apply, build_model_with_cfg
from timm.layers import ClassifierHead, SelectAdaptivePool2d, DropPath
from nvidia_tao_pytorch.cv.backbone.convnext_utils import checkpoint_filter_fn, _init_weights, LayerNorm2d, ConvMlp


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = dict(
    convnext_tiny=_cfg(url="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth"),
    convnext_small=_cfg(url="https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth"),
    convnext_base=_cfg(url="https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth"),
    convnext_large=_cfg(url="https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth"),

    convnext_tiny_hnf=_cfg(url=''),

    convnext_base_in22k=_cfg(
        # url="https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth", num_classes=21841),
        url="pretrained/convnext_base_22k_224.pth", num_classes=21841),
    convnext_large_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth", num_classes=21841),
    convnext_xlarge_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth", num_classes=21841),
)


class Mlp(nn.Module):
    """MLP Module."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """Init Function."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward Function."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvNeXtBlock(nn.Module):
    """ ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.
    """

    def __init__(self, dim, drop_path=0., ls_init_value=1e-6, conv_mlp=True, mlp_ratio=4, norm_layer=None):
        """Initialize ConvNext Block.

        Args:
            dim (int): Number of input channels.
            drop_path (float): Stochastic depth rate. Default: 0.0
            ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
        """
        super().__init__()
        if not norm_layer:
            norm_layer = partial(LayerNorm2d, eps=1e-6) if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
        mlp_layer = ConvMlp if conv_mlp else Mlp
        self.use_conv_mlp = conv_mlp
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=nn.GELU)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """Forward function."""
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class ConvNeXtStage(nn.Module):
    """ConvNeXt Stage."""

    def __init__(
            self, in_chs, out_chs, stride=2, depth=2, dp_rates=None, ls_init_value=1.0, conv_mlp=True,
            norm_layer=None, cl_norm_layer=None, cross_stage=False):
        """Initialize ConvNext Stage.

        Args:
            in_chs (int): Number of input channels.
            out_chs (int): Number of output channels.
        """
        super().__init__()

        if in_chs != out_chs or stride > 1:
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                nn.Conv2d(in_chs, out_chs, kernel_size=stride, stride=stride),
            )
        else:
            self.downsample = nn.Identity()

        dp_rates = dp_rates or [0.] * depth
        self.blocks = nn.Sequential(*[ConvNeXtBlock(
            dim=out_chs, drop_path=dp_rates[j], ls_init_value=ls_init_value, conv_mlp=conv_mlp,
            norm_layer=norm_layer if conv_mlp else cl_norm_layer)
            for j in range(depth)]
        )

    def forward(self, x):
        """Forward function."""
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class ConvNeXt(nn.Module):
    """ ConvNeXt
    A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf
    """

    def __init__(
            self, in_chans=3, num_classes=1000, global_pool='avg', output_stride=32, patch_size=4,
            depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),  ls_init_value=1e-6, conv_mlp=True, use_head=True,
            head_init_scale=1., head_norm_first=False, norm_layer=None, drop_rate=0., drop_path_rate=0.,
    ):
        """ Initialize the ConvNext Class
        Args:
            in_chans (int): Number of input image channels. Default: 3
            num_classes (int): Number of classes for classification head. Default: 1000
            depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
            dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
            drop_rate (float): Head dropout rate
            drop_path_rate (float): Stochastic depth rate. Default: 0.
            ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
            head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        """
        super().__init__()
        assert output_stride == 32
        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)
            cl_norm_layer = norm_layer if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
        else:
            assert conv_mlp, \
                'If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input'
            cl_norm_layer = norm_layer

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []

        # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size),
            norm_layer(dims[0])
        )

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        curr_stride = patch_size
        prev_chs = dims[0]
        stages = []
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(len(depths)):
            stride = 2 if i > 0 else 1
            # FIXME support dilation / output_stride
            curr_stride *= stride
            out_chs = dims[i]
            stages.append(ConvNeXtStage(
                prev_chs, out_chs, stride=stride,
                depth=depths[i], dp_rates=dp_rates[i], ls_init_value=ls_init_value, conv_mlp=conv_mlp,
                norm_layer=norm_layer, cl_norm_layer=cl_norm_layer)
            )
            prev_chs = out_chs
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)

        self.num_features = prev_chs

        if head_norm_first:
            # norm -> global pool -> fc ordering, like most other nets (not compat with FB weights)
            self.norm_pre = norm_layer(self.num_features)  # final norm layer, before pooling
            if use_head:
                self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)
        else:
            # pool -> norm -> fc, the default ConvNeXt ordering (pretrained FB weights)
            self.norm_pre = nn.Identity()
            if use_head:
                self.head = nn.Sequential(OrderedDict([
                    ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
                    ('norm', norm_layer(self.num_features)),
                    ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
                    ('drop', nn.Dropout(self.drop_rate)),
                    ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
                ]))

        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

    def get_classifier(self):
        """Returns classifier of ConvNeXt."""
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool='avg'):
        """Redefine the classification head"""
        if isinstance(self.head, ClassifierHead):
            # norm -> global pool -> fc
            self.head = ClassifierHead(
                self.num_features, num_classes, pool_type=global_pool, drop_rate=self.drop_rate)
        else:
            # pool -> norm -> fc
            self.head = nn.Sequential(OrderedDict([
                ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
                ('norm', self.head.norm),
                ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
                ('drop', nn.Dropout(self.drop_rate)),
                ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
            ]))

    def forward_features(self, x, return_feat=False):
        """Forward Features."""
        x = self.stem(x)
        out_list = []
        # import pdb; pdb.set_trace()
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            out_list.append(x)
        x = self.norm_pre(x)

        return x, out_list if return_feat else x

    def forward(self, x):
        """Forward Function."""
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _create_hybrid_backbone(variant='convnext_base_in22k', pretrained=False, **kwargs):
    """Helper function to create the hybrid model."""
    model = build_model_with_cfg(
        ConvNeXt, variant, pretrained,
        pretrained_cfg=default_cfgs[variant],
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs)
    return model
