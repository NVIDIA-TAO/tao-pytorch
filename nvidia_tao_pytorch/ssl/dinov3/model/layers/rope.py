# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""2D axial Rotary Position Embedding (RoPE) for DINOv3.

DINOv3 replaces DINOv2's absolute positional embedding with 2D axial RoPE applied to
the query/key of *patch* tokens inside attention. The ``[CLS]`` token and register
tokens are non-spatial and receive an identity rotation (sin=0, cos=1).

Token layout (inherited from ``nvdinov2``): ``[CLS, patch_0 .. patch_{HW-1}, reg_0 .. reg_{R-1}]``
i.e. ``[CLS]`` at index 0 and register tokens appended at the **end** of the sequence.
Meta/timm DINOv3 instead place registers right after ``[CLS]`` (``[CLS, regs, patches]``),
but full bidirectional attention is permutation-equivariant and registers carry identity
rotation, so the CLS/patch outputs are invariant to that placement; the step-4 checkpoint
remapper therefore only renames the register parameter, it does not reorder tokens.

The math here is numerically aligned to timm's ``RotaryEmbeddingDinoV3`` (the
``vit_base_patch16_dinov3.lvd1689m`` reference), verified by the rope-parity unit test and
the feature-parity smoke test:

* ``periods = theta ** (2 * arange(head_dim//4) / (head_dim//2))``; angle uses ``2*pi/period``,
* coords are 0.5-centered, normalized per-axis (``normalize_coords="separate"``) and mapped
  to ``[-1, 1]`` (``make_coords_dinov3`` with ``grid_indexing="ij"``),
* the two axes' angles are concatenated then **tiled** (``rotate_half`` layout:
  ``[h_freqs, w_freqs, h_freqs, w_freqs]``), matching ``apply_rot_embed_cat(half=True)``.
"""

import math

import torch
from torch import nn


def rotate_half(x):
    """Rotate the last dimension by splitting it in half: ``[a, b] -> [-b, a]``.

    Matches timm's ``rope_rotate_half`` (the ``rotate_half=True`` / cat convention).

    Args:
        x (torch.Tensor): Tensor with an even-sized last dimension.

    Returns:
        torch.Tensor: The half-rotated tensor, same shape as ``x``.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, sin, cos):
    """Apply rotary position embedding to a query/key tensor.

    Args:
        x (torch.Tensor): Tensor shaped ``[B, N, num_heads, head_dim]`` (xformers layout).
        sin (torch.Tensor): Sine table shaped ``[N, head_dim]``.
        cos (torch.Tensor): Cosine table shaped ``[N, head_dim]``.

    Returns:
        torch.Tensor: ``x`` with RoPE applied, same shape and dtype as ``x``.
    """
    # [N, head_dim] -> [1, N, 1, head_dim] to broadcast over batch and heads.
    sin = sin.unsqueeze(0).unsqueeze(2).to(x.dtype)
    cos = cos.unsqueeze(0).unsqueeze(2).to(x.dtype)
    return x * cos + rotate_half(x) * sin


class RoPE2D(nn.Module):
    """2D axial rotary position embedding generator (timm DINOv3-compatible).

    Produces per-token ``(sin, cos)`` tables for a patch grid, with identity rotation for
    the prefix ``[CLS]`` token and the trailing register tokens. The period schedule is
    registered as a (non-persistent) buffer so FSDP does not attempt to shard it.
    """

    def __init__(self, head_dim: int, theta: float = 100.0,
                 num_prefix_tokens: int = 1, num_register_tokens: int = 4,
                 normalize_coords: str = "separate", grid_offset: float = 0.0):
        """Initialize the axial RoPE generator.

        Args:
            head_dim (int): Per-head feature dimension. Must be divisible by 4 (two axes,
                each contributing ``head_dim//4`` rotary pairs).
            theta (float): Period base (timm ``temperature``). DINOv3 uses 100.0.
            num_prefix_tokens (int): Number of non-spatial prefix tokens (``[CLS]`` -> 1).
            num_register_tokens (int): Number of trailing register tokens (identity rotation).
            normalize_coords (str): Coordinate normalization, one of ``separate`` (per-axis,
                DINOv3 default), ``max`` or ``min`` (shared denominator).
            grid_offset (float): Constant offset added to the 0.5-centered grid indices.
        """
        super().__init__()
        assert head_dim % 4 == 0, f"head_dim must be divisible by 4 for 2D axial RoPE, got {head_dim}"
        self.head_dim = head_dim
        self.theta = float(theta)
        self.num_prefix_tokens = num_prefix_tokens
        self.num_register_tokens = num_register_tokens
        self.normalize_coords = normalize_coords
        self.grid_offset = grid_offset

        n_freq = head_dim // 4  # rotary pairs per spatial axis
        # periods = theta ** (2 * i / (head_dim//2)); freq = 2*pi / period (see timm
        # RotaryEmbeddingDinoV3._compute_periods). bands == 1 / periods.
        exponents = 2.0 * torch.arange(n_freq, dtype=torch.float32) / (head_dim // 2)
        periods = self.theta ** exponents
        self.register_buffer("periods", periods, persistent=False)

        # Small cache so repeated forwards at the same grid size avoid recompute.
        self._cache = {}

    def _make_coords(self, H: int, W: int, device, dtype):
        """Build 0.5-centered, per-axis-normalized coords in ``[-1, 1]`` (timm-aligned).

        Args:
            H (int): Grid height.
            W (int): Grid width.
            device: Target device.
            dtype: Output dtype.

        Returns:
            torch.Tensor: Coordinates shaped ``[H*W, 2]`` (row-major, ``ij`` indexing).
        """
        coords_h = torch.arange(0.5, H, device=device, dtype=torch.float32) + self.grid_offset
        coords_w = torch.arange(0.5, W, device=device, dtype=torch.float32) + self.grid_offset

        if self.normalize_coords == "max":
            h_denom = w_denom = float(max(H, W))
        elif self.normalize_coords == "min":
            h_denom = w_denom = float(min(H, W))
        elif self.normalize_coords == "separate":
            h_denom, w_denom = float(H), float(W)
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        coords_h = coords_h / h_denom
        coords_w = coords_w / w_denom

        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1
        ).flatten(0, 1)  # [H*W, 2]
        coords = 2.0 * coords - 1.0
        return coords.to(dtype)

    def forward(self, H: int, W: int, device, dtype=torch.float32):
        """Build per-token sin/cos tables for one crop's token sequence.

        Args:
            H (int): Patch-grid height.
            W (int): Patch-grid width.
            device: Target device.
            dtype: Compute/return dtype for the tables (default fp32).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ``(sin, cos)`` each shaped
            ``[num_prefix + H*W + num_register, head_dim]``. Prefix and register rows are
            identity (sin=0, cos=1).
        """
        key = (H, W, device, dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        n_patches = H * W
        seq_len = self.num_prefix_tokens + n_patches + self.num_register_tokens

        coords = self._make_coords(H, W, device, dtype)          # [HW, 2]
        periods = self.periods.to(device=device, dtype=dtype)     # [head_dim//4]
        # angle = 2*pi * coord / period, per (coord-axis, period) -> [HW, 2, n_freq]
        angles = 2.0 * math.pi * coords[:, :, None] / periods[None, None, :]
        angles = angles.flatten(1)                               # [HW, head_dim//2]
        angles = angles.tile(2)                                  # [HW, head_dim]
        sin_p, cos_p = angles.sin(), angles.cos()

        sin = torch.zeros(seq_len, self.head_dim, device=device, dtype=dtype)
        cos = torch.ones(seq_len, self.head_dim, device=device, dtype=dtype)
        start = self.num_prefix_tokens
        sin[start:start + n_patches] = sin_p
        cos[start:start + n_patches] = cos_p

        self._cache[key] = (sin, cos)
        return sin, cos
