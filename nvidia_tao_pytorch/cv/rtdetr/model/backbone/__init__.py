# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backbones for RT-DETR."""

from nvidia_tao_pytorch.cv.rtdetr.model.backbone.registry import RTDETR_BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.resnet import (
    resnet_18,
    resnet_34,
    resnet_50,
    resnet_101,
)
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.convnext import (
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
    convnext_xlarge
)
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.convnext_v2 import (
    convnextv2_nano,
    convnextv2_tiny,
    convnextv2_base,
    convnextv2_large,
    convnextv2_huge,
)
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.efficientvit import (
    efficientvit_b0,
    efficientvit_b1,
    efficientvit_b2,
    efficientvit_b3,
    efficientvit_l0,
    efficientvit_l1,
    efficientvit_l2,
    efficientvit_l3,
)
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.fan import (
    fan_tiny_8_p4_hybrid,
    fan_small_12_p4_hybrid,
    fan_base_12_p4_hybrid,
    fan_large_12_p4_hybrid,
)
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.edgenext import (
    edgenext_x_small,
    edgenext_small,
    edgenext_base,
)

__all__ = [
    "RTDETR_BACKBONE_REGISTRY",
    "resnet_18",
    "resnet_34",
    "resnet_50",
    "resnet_101",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "convnext_xlarge",
    "convnextv2_nano",
    "convnextv2_tiny",
    "convnextv2_base",
    "convnextv2_large",
    "convnextv2_huge",
    "efficientvit_b0",
    "efficientvit_b1",
    "efficientvit_b2",
    "efficientvit_b3",
    "efficientvit_l0",
    "efficientvit_l1",
    "efficientvit_l2",
    "efficientvit_l3",
    "fan_tiny_8_p4_hybrid",
    "fan_small_12_p4_hybrid",
    "fan_base_12_p4_hybrid",
    "fan_large_12_p4_hybrid",
    "edgenext_x_small",
    "edgenext_small",
    "edgenext_base",
]
