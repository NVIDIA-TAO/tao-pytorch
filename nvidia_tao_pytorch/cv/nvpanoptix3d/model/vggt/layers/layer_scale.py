# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions of this code are based on the VGGT project by Facebook Research (Meta):
# https://github.com/facebookresearch/vggt

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
