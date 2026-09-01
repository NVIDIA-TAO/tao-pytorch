# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

"""Patch embedding module for the VGGT model."""

from typing import Callable, Optional, Tuple, Union

from torch import Tensor
import torch.nn as nn


def make_2tuple(x):
    """Broadcast an int to a 2-tuple, or pass a 2-tuple through unchanged."""
    if isinstance(x, tuple):
        if len(x) != 2:
            raise ValueError(f"Expected a 2-tuple, got {x!r}")
        return x

    if not isinstance(x, int):
        raise TypeError(f"Expected an int or 2-tuple, got {type(x).__name__}")
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        patch_HW = make_2tuple(patch_size)

        self.patch_size = patch_HW
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        if H % patch_H != 0:
            raise ValueError(
                f"Input height {H} is not divisible by patch height {patch_H}"
            )
        if W % patch_W != 0:
            raise ValueError(
                f"Input width {W} is not divisible by patch width {patch_W}"
            )

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x
