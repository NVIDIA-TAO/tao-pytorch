# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer scale module for the VGGT model."""

from typing import Union
import torch
from torch import Tensor
from torch import nn


class LayerScale(nn.Module):
    """
    LayerScale module

    Args:
        dim (int): Dimension of the input features.
        init_values (Union[float, Tensor]): Initial value for the LayerScale. Default is 1e-5.
        inplace (bool): Whether to use inplace operation. Default is False.

    Returns:
        Tensor: Output features of shape (B, N, C).
    """

    def __init__(
        self, dim: int, init_values: Union[float, Tensor] = 1e-5, inplace: bool = False
    ) -> None:
        """LayerScale constructor"""
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass"""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
