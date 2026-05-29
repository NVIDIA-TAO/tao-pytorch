# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core model components for TAO Toolkit models."""
from typing import List, Optional

import torch
import torch.nn as nn

import timm


class Backbone(nn.Module):
    """Base backbone module"""

    def out_strides(self) -> List[int]:
        """Returns the output strides of the backbone"""
        raise NotImplementedError

    def out_channels(self) -> List[int]:
        """Returns the output channels of the backbone"""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass of the backbone"""
        raise NotImplementedError


class TimmBackbone(Backbone):
    """TimmBackbone class to use timm for creating backbone

    Uses model_name to create backbone and downloads pretrained weights if specified
    """

    def __init__(self, model_name: str,
                 pretrained: bool = True,
                 out_indices: Optional[List[int]] = None,
                 pretrained_path: Optional[str] = None
                 ):
        """Initializes the TimmBackbone"""
        super().__init__()

        if pretrained_path:
            self.model = timm.create_model(
                model_name=model_name,
                pretrained=False,
                features_only=True,
                out_indices=out_indices,
                pretrained_cfg_overlay=dict(file=pretrained_path)
            )
        else:
            self.model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices
            )

    def out_strides(self) -> List[int]:
        """Returns the output strides of the backbone"""
        return self.model.feature_info.reduction()

    def out_channels(self) -> List[int]:
        """Returns the output channels of the backbone"""
        return self.model.feature_info.channels()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass of the backbone"""
        return self.model(x)
