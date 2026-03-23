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

"""SwiGLU FFN module for the VGGT model."""

from typing import Callable, Optional
from torch import Tensor, nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """
    SwiGLUFFN module

    Args:
        in_features (int): Input features dimension.
        hidden_features (Optional[int]): Hidden features dimension. Default is None.
        out_features (Optional[int]): Output features dimension. Default is None.
        act_layer (Callable[..., nn.Module]): Activation layer. Default is None.
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
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        """SwiGLUFFN constructor"""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass"""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


SwiGLU = SwiGLUFFN


class SwiGLUFFNFused(SwiGLU):
    """
    SwiGLUFFNFused module

    Args:
        in_features (int): Input features dimension.
        hidden_features (Optional[int]): Hidden features dimension. Default is None.
        out_features (Optional[int]): Output features dimension. Default is None.
        act_layer (Callable[..., nn.Module]): Activation layer. Default is None.
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
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        """SwiGLUFFNFused constructor"""
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features, hidden_features=hidden_features, out_features=out_features, bias=bias
        )
