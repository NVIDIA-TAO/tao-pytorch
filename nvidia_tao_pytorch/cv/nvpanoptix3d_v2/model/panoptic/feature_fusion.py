# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVPanoptix3Dv2 visual feature fusion.

NVPanoptix3Dv2 concatenates its frozen visual streams, including DINO and VGGT.
Views are processed independently, so the mixer has no ordinal frame encoding
or view-count-dependent operation. Geometry is deliberately not injected.
"""

import torch
from torch import nn

from nvidia_tao_pytorch.cv.nvpanoptix3d.model.vggt.layers.block import Block
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.vggt.layers.rope import (
    RotaryPositionEmbedding2D,
)


class ConcatenatedFeatureFusion(nn.Module):
    """Concatenate/project DINO+VGGT, then spatially mix each view."""

    def __init__(
        self,
        dino_dim: int = 1024,
        vggt_dim: int = 2048,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 3,
        ff_dim_mult: float = 4.0,
    ):
        super().__init__()
        self.dino_dim = int(dino_dim)
        self.vggt_dim = int(vggt_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        if self.num_layers < 0:
            raise ValueError(f"num_layers must be non-negative, got {num_layers}")
        if self.num_layers > 0:
            if self.num_heads <= 0 or self.hidden_dim % self.num_heads != 0:
                raise ValueError(
                    f"hidden_dim={hidden_dim} must be divisible by a positive "
                    f"num_heads={num_heads}"
                )
            if (self.hidden_dim // self.num_heads) % 4 != 0:
                raise ValueError(
                    "2D RoPE requires attention head_dim to be divisible by "
                    f"4, got {self.hidden_dim // self.num_heads}"
                )

        self.proj = nn.Linear(
            self.dino_dim + self.vggt_dim, self.hidden_dim,
        )
        rope = RotaryPositionEmbedding2D(frequency=100.0)
        self.mixer_blk = nn.ModuleList([
            Block(
                dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=float(ff_dim_mult),
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.0,
                qk_norm=False,
                fused_attn=True,
                rope=rope,
            )
            for _ in range(self.num_layers)
        ])
        self.out_norm = nn.LayerNorm(self.hidden_dim)

    @staticmethod
    def spatial_positions(
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return row-major ``(y, x)`` positions for the native patch grid."""
        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        positions = torch.stack((y, x), dim=-1).reshape(1, height * width, 2)
        return positions.expand(batch_size, -1, -1)

    def forward(
        self,
        f_dino: torch.Tensor,
        f_vggt: torch.Tensor,
        spatial_shape=None,
    ) -> torch.Tensor:
        """Project the concatenated DINOv2/VGGT tokens and refine them per view."""
        if f_dino.shape[:-1] != f_vggt.shape[:-1]:
            raise ValueError(
                "DINO and VGGT token shapes must match before channels: "
                f"{tuple(f_dino.shape)} vs {tuple(f_vggt.shape)}"
            )

        x = self.proj(torch.cat([f_dino, f_vggt], dim=-1))
        if self.mixer_blk:
            if spatial_shape is None:
                raise ValueError(
                    "spatial_shape=(patch_h, patch_w) is required when the "
                    "spatial mixer is enabled"
                )
            patch_h, patch_w = int(spatial_shape[0]), int(spatial_shape[1])
            if patch_h <= 0 or patch_w <= 0:
                raise ValueError(
                    f"spatial_shape must be positive, got {(patch_h, patch_w)}"
                )
            if x.shape[1] != patch_h * patch_w:
                raise ValueError(
                    "Token count does not match the native patch grid: "
                    f"P={x.shape[1]}, grid={patch_h}*{patch_w}"
                )
            positions = self.spatial_positions(
                x.shape[0], patch_h, patch_w, x.device,
            )
            for block in self.mixer_blk:
                x = block(x, pos=positions)

        return self.out_norm(x)
