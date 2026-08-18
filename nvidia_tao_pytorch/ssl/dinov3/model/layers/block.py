# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoPE-aware NestedTensorBlock for DINOv3.

Mirrors ``nvdinov2``'s :class:`NestedTensorBlock` (same nested-tensor batching, stochastic
depth, xformers ``BlockDiagonalMask`` path) but threads a 2D axial RoPE table into the
attention call. The rotation is applied to Q/K **before** the memory-efficient attention,
and the per-token tables are concatenated to match the block-diagonal token layout so RoPE
composes with the variable-length nested batching.
"""

from typing import List, Union

import torch
from torch import Tensor
from timm.models.vision_transformer import DropPath, LayerScale

from nvidia_tao_pytorch.ssl.nvdinov2.model.layers.block import (
    Block,
    NestedTensorBlock,
    get_attn_bias_and_cat,
    drop_add_residual_stochastic_depth_list,
)
from nvidia_tao_pytorch.ssl.dinov3.model.layers.attention import RoPEMemoryEfficientAttention


def _build_rope_cat(rope_list, attn_bias):
    """Concatenate per-crop RoPE tables to match a block-diagonal token sequence.

    ``get_attn_bias_and_cat`` flattens ``x_list`` into a single sequence of per-image
    blocks (crop 0's images, then crop 1's images, ...). Every image of a given crop shares
    the same RoPE table, so the concatenated table is each crop's table tiled by its
    (possibly stochastic-depth-subsampled) image count, read from ``attn_bias._batch_sizes``.

    Args:
        rope_list (Optional[List[Tuple[Tensor, Tensor]]]): Per-crop ``(sin, cos)`` tables,
            each shaped ``[N_i, head_dim]``, parallel to the block's input list. ``None``
            disables RoPE.
        attn_bias: Block-diagonal mask carrying ``_batch_sizes`` (images per crop).

    Returns:
        Optional[Tuple[Tensor, Tensor]]: Concatenated ``(sin, cos)`` shaped
        ``[sum_i(b_i * N_i), head_dim]``, or ``None`` if ``rope_list`` is ``None``.
    """
    if rope_list is None:
        return None
    batch_sizes = attn_bias._batch_sizes
    sins, coss = [], []
    for (sin, cos), b in zip(rope_list, batch_sizes):
        sins.append(sin.repeat(b, 1))
        coss.append(cos.repeat(b, 1))
    return torch.cat(sins, 0), torch.cat(coss, 0)


class RoPENestedTensorBlock(NestedTensorBlock):
    """NestedTensorBlock that applies 2D axial RoPE inside attention."""

    def __init__(self, *args, use_custom_attention=True, **kwargs):
        """Initialize with the RoPE-aware attention.

        Bypasses :class:`NestedTensorBlock`'s attention-class guard (which only permits the
        non-RoPE ``MemoryEfficientAttention``) by initializing the grandparent
        :class:`Block` directly with :class:`RoPEMemoryEfficientAttention`.

        Args:
            *args: Forwarded to :class:`Block`.
            use_custom_attention (bool): Whether to use xformers' memory_efficient_attention.
            **kwargs: Forwarded to :class:`Block`.
        """
        # Intentional grandparent init (see docstring): bypasses NestedTensorBlock's attn-class
        # guard to install the RoPE attention.
        Block.__init__(self, *args, attn_class=RoPEMemoryEfficientAttention, **kwargs)  # pylint: disable=non-parent-init-called
        self.use_custom_attention = use_custom_attention

    def forward(
        self,
        x_or_x_list: Union[Tensor, List[Tensor]],
        rope=None,
    ) -> Union[Tensor, List[Tensor]]:
        """Forward pass with RoPE threaded into attention.

        Args:
            x_or_x_list (Union[Tensor, List[Tensor]]): A single token tensor or a list of
                per-crop token tensors.
            rope: For a single tensor, a ``(sin, cos)`` table; for a list, a list of
                ``(sin, cos)`` tables parallel to ``x_or_x_list``. ``None`` disables RoPE.

        Returns:
            Union[Tensor, List[Tensor]]: Processed tensor or list of tensors.
        """
        if isinstance(x_or_x_list, Tensor):
            wrapped = None if rope is None else [rope]
            return self.forward([x_or_x_list], rope=wrapped)[0]

        x_list = x_or_x_list
        rope_list = rope
        drop_ratio = (
            self.drop_path1.drop_prob if isinstance(self.drop_path1, DropPath) else 0.0
        )

        if self.training and drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                """Compute the RoPE attention residual."""
                return self.attn(
                    self.norm1(x),
                    attn_bias=attn_bias,
                    use_custom_attention=self.use_custom_attention,
                    rope=_build_rope_cat(rope_list, attn_bias),
                )

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                """Compute the feedforward residual."""
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                drop_ratio=drop_ratio,
                scaling_vector=self.ls1.gamma
                if isinstance(self.ls1, LayerScale)
                else None,
            )

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                drop_ratio=drop_ratio,
                scaling_vector=self.ls2.gamma
                if isinstance(self.ls2, LayerScale)
                else None,
            )
            return x_list

        def attn_residual_func_ls(x: Tensor, attn_bias=None) -> Tensor:
            """Compute the layer-scaled RoPE attention residual."""
            return self.ls1(
                self.attn(
                    self.norm1(x),
                    attn_bias=attn_bias,
                    use_custom_attention=self.use_custom_attention,
                    rope=_build_rope_cat(rope_list, attn_bias),
                )
            )

        def ffn_residual_func_ls(x: Tensor, attn_bias=None) -> Tensor:
            """Compute the layer-scaled feedforward residual."""
            return self.ls2(self.mlp(self.norm2(x)))

        attn_bias, x = get_attn_bias_and_cat(x_list)
        x = x + attn_residual_func_ls(x, attn_bias=attn_bias)
        x = x + ffn_residual_func_ls(x)

        return attn_bias.split(x)
