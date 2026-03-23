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

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

"""Attention module for the VGGT model."""

from torch import Tensor
from torch import nn
import torch.nn.functional as F

try:
    from xformers.ops import unbind, memory_efficient_attention
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    unbind = None
    memory_efficient_attention = None


class Attention(nn.Module):
    """
    Attention module

    Args:
        dim (int): Dimension of the input features.
        num_heads (int): Number of attention heads. Default is 8.
        qkv_bias (bool): Whether to use bias in the QKV projection. Default is True.
        proj_bias (bool): Whether to use bias in the projection. Default is True.
        attn_drop (float): Dropout rate for the attention. Default is 0.0.
        proj_drop (float): Dropout rate for the projection. Default is 0.0.
        norm_layer (nn.Module): Normalization layer. Default is nn.LayerNorm.
        qk_norm (bool): Whether to use QK normalization. Default is False.
        fused_attn (bool): Whether to use fused attention. Default is True.
        rope (nn.Module): Rotary position embedding module. Default is None.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        """Attention constructor"""
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None) -> Tensor:
        """
        Forward pass

        Args:
            x (Tensor): Input features of shape (B, N, C).
            pos (Tensor): Positional embedding of shape (B, N, C). Default is None.

        Returns:
            Tensor: Output features of shape (B, N, C).
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    """
    Memory efficient attention module
    Inherits from Attention class.
    """

    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        """
        Forward pass

        Args:
            x (Tensor): Input features of shape (B, N, C).
            attn_bias (Tensor): Attention bias of shape (B, N, N). Default is None.
            pos (Tensor): Positional embedding of shape (B, N, C). Default is None.

        Returns:
            Tensor: Output features of shape (B, N, C).
        """
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
