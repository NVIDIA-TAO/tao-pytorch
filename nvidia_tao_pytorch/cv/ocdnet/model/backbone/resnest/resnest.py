#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ResNeSt models"""

from nvidia_tao_pytorch.cv.ocdnet.model.backbone.resnest.resnet import ResNet, Bottleneck

__all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269']


def resnest50(pretrained=False, root='~/.encoding/models', **kwargs):
    """Resnest50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    return model


def resnest101(pretrained=False, root='~/.encoding/models', **kwargs):
    """Resnest101 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    return model


def resnest200(pretrained=False, root='~/.encoding/models', **kwargs):
    """Resnest200 model."""
    model = ResNet(Bottleneck, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    return model


def resnest269(pretrained=False, root='~/.encoding/models', **kwargs):
    """Resnest269 model."""
    model = ResNet(Bottleneck, [3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    return model
