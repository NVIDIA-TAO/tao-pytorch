# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RT-DETR backbone registry."""
# from nvidia_tao_pytorch.cv.backbone.backbone_base import BackboneBase
from fvcore.common.registry import Registry  # for backward compatibility.


RTDETR_BACKBONE_REGISTRY = Registry("RTDETR_BACKBONE")
RTDETR_BACKBONE_REGISTRY.__doc__ = """
Registry for RT-DETR backbones, which extract feature maps from images
Registered object must return instance of :class:`BackboneBase`.
"""
