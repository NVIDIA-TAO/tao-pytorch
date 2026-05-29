#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""model init."""
import copy
from torch import nn

# pylint: disable=W0401,W0611,W0614
# flake8: noqa: F401, F403
from nvidia_tao_pytorch.cv.ocdnet.model.head.conv_head import ConvHead
from nvidia_tao_pytorch.cv.ocdnet.model.head.db_head import DBHead
from nvidia_tao_pytorch.cv.ocdnet.model.losses.DB_loss import DBLoss
from nvidia_tao_pytorch.cv.ocdnet.model.neck.FPN import FPN
from nvidia_tao_pytorch.cv.ocdnet.model.neck.fan_neck import FANNeck
from nvidia_tao_pytorch.cv.ocdnet.model.backbone.resnet import *
from nvidia_tao_pytorch.cv.ocdnet.model.backbone.fan import *
from nvidia_tao_pytorch.cv.ocdnet.model.backbone.resnest import *
from nvidia_tao_pytorch.cv.ocdnet.model.backbone.shufflenetv2 import *
from nvidia_tao_pytorch.cv.ocdnet.model.backbone.mobilenet_v3 import MobileNetV3


__all__ = ['build_head', 'build_loss', 'build_neck', 'build_backbone']
support_head = ['ConvHead', 'DBHead']
support_loss = ['DBLoss']
support_neck = ['FPN','FANNeck']
support_backbone = ['resnet18', 'deformable_resnet18', 'deformable_resnet50',
                    'resnet50', 'resnet34', 'resnet101', 'resnet152',
                    'resnest50', 'resnest101', 'resnest200', 'resnest269',
                    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
                    'MobileNetV3',
                    'fan_tiny_8_p4_hybrid', 'fan_small_12_p4_hybrid','fan_large_16_p4_hybrid'
                    ]


def build_head(head_name, **kwargs):
    """Build head."""
    assert head_name in support_head, f'all support head is {support_head}'
    head = globals()[head_name](**kwargs)
    return head


def build_loss(config):
    """Build loss."""
    copy_config = copy.deepcopy(config)
    loss_type = copy_config.pop('type')
    assert loss_type in support_loss, f'all support loss is {support_loss}'
    criterion = globals()[loss_type](**copy_config)
    return criterion


def build_neck(neck_name, **kwargs):
    """Build neck."""
    assert neck_name in support_neck, f'all support neck is {support_neck}'
    neck = globals()[neck_name](**kwargs)
    return neck


def build_backbone(backbone_name, **kwargs):
    """Build backbone."""
    assert backbone_name in support_backbone, f'all support backbone is {support_backbone}'
    backbone = globals()[backbone_name](**kwargs)
    return backbone


class Model(nn.Module):
    """Model class."""

    def __init__(self, model_config: dict):
        """Construct Model."""
        super().__init__()
        backbone_type = model_config["backbone"]
        if 'fan' in backbone_type:
            model_config['neck'] = 'FANNeck'
        elif 'resnet' in backbone_type:
            model_config['neck'] = 'FPN'
        neck_type = model_config['neck']
        head_type = model_config['head']

        enlarge_feature_map_size = model_config['enlarge_feature_map_size']
        if 'fan' not in backbone_type:
            print("Only FAN backbone support enlarge feature map, overriding enlarge_feature_map_size to false")
            enlarge_feature_map_size = False

        dict_backbone = {
            "pretrained": model_config['pretrained'],
            "in_channels": model_config['in_channels'],
            "enlarge_feature_map_size": enlarge_feature_map_size,
            "quant": model_config['quant'],
            "activation_checkpoint": model_config['activation_checkpoint'],
            'fuse_qkv_proj': model_config['fuse_qkv_proj']
            }
        dict_neck = {"inner_channels": model_config['inner_channels']}
        dict_head = {"out_channels": model_config['out_channels'], "k": model_config['k'], "enlarge_feature_map_size": enlarge_feature_map_size}

        self.backbone = build_backbone(backbone_type, **dict_backbone)
        self.neck = build_neck(neck_type, in_channels=self.backbone.out_channels, **dict_neck)
        self.head = build_head(head_type, in_channels=self.neck.out_channels, **dict_head)
        self.name = f'{backbone_type}_{neck_type}_{head_type}'

    def forward(self, x):
        """Forward."""
        backbone_out = self.backbone(x)
        neck_out = self.neck(backbone_out)
        with torch.amp.autocast(enabled=False, device_type="cuda"):
            y = self.head(neck_out)

        return y
