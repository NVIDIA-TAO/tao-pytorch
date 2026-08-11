# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention"""

import torch
from torch.nn import functional as F
from timm.models.vision_transformer import Attention
from xformers.ops import memory_efficient_attention


class MemoryEfficientAttention(Attention):
    """Memory Efficient Attention"""

    def _fallback_attention(self, q, k, v, attn_bias=None):
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

        ``attn_bias`` MUST be honoured. The SSL blocks concatenate their crop list into a
        single sequence via ``get_attn_bias_and_cat`` -- so one "batch" row holds every image's
        tokens end to end -- and the block-diagonal mask is the only thing preventing image i's
        tokens from attending to image j's. Dropping it silently blends the whole batch
        together: features stop depending on their own image, and the damage grows with batch
        size. Measured on H100 against timm's reference DINOv3, CLS cosine was 1.000 at batch 1,
        0.658 at 2, 0.368 at 4 and 0.193 at 8, and ImageNet k-NN fell from 82.4% to 4.0%.

        Args:
            q, k, v: ``[B, N, num_heads, head_dim]`` (xformers layout, post QK-norm).
            attn_bias: xformers block-diagonal mask, or ``None``. Materialized into an
                additive float mask for SDPA.

        Returns:
            torch.Tensor: ``[B, N, num_heads, head_dim]`` (same layout as the input).
        """
        out_dtype = q.dtype
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_mask = None
        if attn_bias is not None:
            # Materialize to [B, num_heads, N, N]; -inf off-block keeps crops separated.
            attn_mask = attn_bias.materialize(
                (q.shape[0], q.shape[1], q.shape[2], k.shape[2]),
                dtype=torch.float32, device=q.device,
            )
        with torch.autocast("cuda", enabled=False):
            x = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(),
                attn_mask=attn_mask,
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
            x = self._fallback_attention(q, k, v, attn_bias)

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
