# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention"""

import torch
from timm.models.vision_transformer import Attention
from xformers.ops import memory_efficient_attention


class MemoryEfficientAttention(Attention):
    """Memory Efficient Attention"""

    def forward(self, x, attn_bias=None, use_custom_attention=True):
        """Apply memory_efficient_attention in xformers

        Args:
            x (torch.Tensor): Input tensor
            attn_bias (torch.Tensor, optional): Bias to apply to the attention matrix. Defaults to None.
            use_custom_attention (bool): Whether to use memory_efficient_attention.
        Returns:
            torch.Tensor: Output tensor after memory_efficient_attention
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        q, k, v = qkv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)

        if use_custom_attention:
            with torch.autocast("cuda", enabled=False):
                x = memory_efficient_attention(
                    q.half(),
                    k.half(),
                    v.half(),
                    attn_bias=attn_bias,
                    p=self.attn_drop.p,
                )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
            x = x.transpose(1, 2)

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
