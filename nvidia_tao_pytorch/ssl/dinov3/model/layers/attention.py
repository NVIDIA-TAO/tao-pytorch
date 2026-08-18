# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoPE-aware Memory Efficient Attention for DINOv3."""

import torch
from xformers.ops import memory_efficient_attention

from nvidia_tao_pytorch.ssl.nvdinov2.model.layers.attention import MemoryEfficientAttention
from nvidia_tao_pytorch.ssl.dinov3.model.layers.rope import apply_rope


class RoPEMemoryEfficientAttention(MemoryEfficientAttention):
    """Memory-efficient attention with 2D axial RoPE applied to Q/K.

    Identical to ``nvdinov2``'s :class:`MemoryEfficientAttention` except that, when a
    ``rope`` table is supplied, the rotation is applied to the query and key (in fp32, for
    parity with the timm reference) **after** QK-norm and **before** the xformers
    memory-efficient attention call. The rotation table already excludes ``[CLS]`` and
    register tokens (identity rows), so no masking is needed here.
    """

    def forward(self, x, attn_bias=None, use_custom_attention=True, rope=None):
        """Apply (optionally RoPE-rotated) memory-efficient attention.

        Args:
            x (torch.Tensor): Input tensor shaped ``[B, N, C]``.
            attn_bias (xformers mask, optional): Block-diagonal attention bias.
            use_custom_attention (bool): Whether to use xformers' memory_efficient_attention.
            rope (Tuple[torch.Tensor, torch.Tensor], optional): ``(sin, cos)`` tables each
                shaped ``[N, head_dim]`` aligned with the (concatenated) token sequence.

        Returns:
            torch.Tensor: Output tensor shaped ``[B, N, C]``.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        q, k, v = qkv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            sin, cos = rope
            q = apply_rope(q.float(), sin, cos).to(q.dtype)
            k = apply_rope(k.float(), sin, cos).to(k.dtype)

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
            # Non-xformers path (e.g. Blackwell, where use_custom_attention is forced off). Uses
            # the shared SDPA-based fallback (base MemoryEfficientAttention._fallback_attention) so
            # both SSL families get the same numerically-safe, fused compute; RoPE has already been
            # applied to q/k above. See bug 6460915.
            #
            # attn_bias is forwarded: the block concatenates its crop list into one sequence,
            # so without the block-diagonal mask every image attends to every other image in
            # the batch, which silently corrupts features wherever this fallback is the only
            # available path.
            x = self._fallback_attention(q, k, v, attn_bias)

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
