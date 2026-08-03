# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention"""

import torch
from torch.nn import functional as F
from timm.models.vision_transformer import Attention
from xformers.ops import memory_efficient_attention


class MemoryEfficientAttention(Attention):
    """Memory Efficient Attention"""

    def _fallback_attention(self, q, k, v):
        """Non-xformers fallback attention via PyTorch SDPA, computed in fp32.

        ``q``, ``k``, ``v`` are ``[B, N, num_heads, head_dim]`` (xformers layout, post QK-norm);
        the head axis is moved forward to ``[B, num_heads, N, head_dim]`` for
        ``scaled_dot_product_attention``. Run in fp32 (autocast off) so the fused, memory-efficient
        SDPA kernel replaces a hand-rolled ``[B, num_heads, N, N]`` score materialization while
        staying numerically safe: with ``qk_norm=False`` the raw QK scores are unbounded and, under
        16-mixed autocast, an fp16 score matmul saturates past fp16's 65504 ceiling to +/-inf so
        ``softmax(inf)=NaN`` poisons the loss. fp32 avoids the overflow on any SDPA backend/GPU (the
        target Blackwell path can't be assumed to have a Hopper-style fused fp16 kernel), and SDPA
        is ~2x faster than the explicit fp32 score matmul. This path is exercised e.g. on Blackwell
        GPUs, where ``use_custom_attention`` is force-disabled for both SSL families. See bug 6460915
        and the MR !652 SDPA review.

        Returns:
            torch.Tensor: ``[B, N, num_heads, head_dim]`` (same layout as the input).
        """
        out_dtype = q.dtype
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        with torch.autocast("cuda", enabled=False):
            x = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(),
                dropout_p=self.attn_drop.p if self.training else 0.0,
                scale=self.scale,
            )
        return x.to(out_dtype).transpose(1, 2)

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
            x = self._fallback_attention(q, k, v)

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
