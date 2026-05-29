# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention"""

from torch import Tensor
from timm.models.vision_transformer import Attention

try:
    from xformers.ops import memory_efficient_attention, unbind

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False


class MemEffAttention(Attention):
    """Memory Efficient Attention"""

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        """Apply memory_efficient_attention in xformers

        Args:
            x (torch.Tensor): Input tensor
            attn_bias (torch.Tensor, optional): Bias to apply to the attention matrix. Defaults to None.
        Returns:
            torch.Tensor: Output tensor after memory_efficient_attention
        """
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
