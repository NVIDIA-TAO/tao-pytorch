# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse4Dbackbone registry."""
from fvcore.common.registry import Registry  # for backward compatibility.


SPARSE4D_BACKBONE_REGISTRY = Registry("SPARSE4D_BACKBONE")
SPARSE4D_BACKBONE_REGISTRY.__doc__ = """
Registry for Sparse4D backbones, which extract feature maps from images
Registered object must return instance of :class:`BackboneBase`.
"""
