# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py

"""MLP module for the VGGT model."""

from typing import Callable, Optional
from torch import Tensor, nn


class Mlp(nn.Module):
    """
    Mlp module

    Args:
        in_features (int): Input features dimension.
        hidden_features (Optional[int]): Hidden features dimension. Default is None.
        out_features (Optional[int]): Output features dimension. Default is None.
        act_layer (Callable[..., nn.Module]): Activation layer. Default is nn.GELU.
        drop (float): Dropout rate. Default is 0.0.
        bias (bool): Whether to use bias. Default is True.

    Returns:
        Tensor: Output features of shape (B, N, C).
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Mlp constructor"""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
