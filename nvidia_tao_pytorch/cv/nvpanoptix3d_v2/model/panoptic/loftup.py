# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LoftUp Upscaler for NVPanoptix3Dv2.

Self-contained implementation adapted from the LoftUp project
(https://github.com/andrehuang/loftup, MIT License).

Generates high-resolution mask features by cross-attending Fourier-encoded
image features with low-resolution patch features. No InputMixer — the
concatenated DINO/VGGT features are used directly. Spatial tensors always
remain in the input image's actual ``(H, W)`` orientation.

All building blocks (CrossAttention, Mlp, DropPath, CrossonlyDecoderBlock)
are inlined to avoid external dependencies on croco.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.cv.nvpanoptix3d.model.vggt.layers.mlp import Mlp


# Inlined building blocks (originally from croco.models.blocks)

class DropPath(nn.Module):
    """Stochastic depth (drop entire residual branch during training)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype) >= self.drop_prob
        return x / (1.0 - self.drop_prob) * keep


class CrossAttention(nn.Module):
    """Multi-head cross-attention (no RoPE — simplified for LoftUp use)."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value):
        """Forward pass."""
        B, Nq, C = query.shape
        Nk = key.shape[1]

        q = self.q_proj(query).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # F.scaled_dot_product_attention dispatches to Flash Attention 2 / a
        # memory-efficient kernel and never materializes the (Nq, Nk) matrix.
        # The original eager form (attn = (q @ k.T) * scale; softmax; @ v)
        # allocates ~Nq*Nk*B*heads*dtype_bytes which OOMs at multi-view
        # inference: e.g. 50 views * (518/14)^2 = 68k tokens -> 34 GiB.
        # Functionally identical for inference (no dropout in eval mode).
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossonlyDecoderBlock(nn.Module):
    """Cross-attention decoder block: cross-attn + FFN with pre-norm."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0,
                 qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True):
        super().__init__()
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y):
        """Forward pass."""
        y_ = self.norm_y(y)
        x = x + self.drop_path(self.cross_attn(self.norm2(x), y_, y_))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


# LoftUp components

class MinMaxScaler(nn.Module):
    """Rescale each channel of a feature map to the unit range."""

    def forward(self, x):
        """Forward pass."""
        c = x.shape[1]
        flat_x = x.permute(1, 0, 2, 3).reshape(c, -1)
        flat_x_min = flat_x.min(dim=-1).values.reshape(1, c, 1, 1)
        flat_x_scale = flat_x.max(dim=-1).values.reshape(1, c, 1, 1) - flat_x_min
        return ((x - flat_x_min) / flat_x_scale.clamp_min(0.0001)) - 0.5


class ImplicitFeaturizer(nn.Module):
    """Fourier positional / color features for high-res guidance tokens."""

    def __init__(self, color_feats=True, n_freqs=10):
        super().__init__()
        self.color_feats = color_feats
        self.n_freqs = n_freqs

        self.dim_multiplier = 2
        if self.color_feats:
            self.dim_multiplier += 3
        self.biases = nn.Parameter(
            torch.randn(2, self.dim_multiplier, n_freqs, dtype=torch.float32)
        )

    def forward(self, original_image):
        """Lift an RGB image to Fourier-encoded features conditioned on colour."""
        b, _, h, w = original_image.shape
        grid_h = torch.linspace(-1, 1, h, device=original_image.device)
        grid_w = torch.linspace(-1, 1, w, device=original_image.device)
        feats = torch.cat(
            [t.unsqueeze(0) for t in torch.meshgrid(grid_h, grid_w, indexing="ij")]
        ).unsqueeze(0)
        feats = feats.expand(b, -1, -1, -1)

        if self.color_feats:
            feat_list = [feats, original_image]
        else:
            feat_list = [feats]

        feats = torch.cat(feat_list, dim=1).unsqueeze(1)
        freqs = torch.exp(
            torch.linspace(-2, 10, self.n_freqs, device=original_image.device)
        ).reshape(1, self.n_freqs, 1, 1, 1)
        feats = feats * freqs

        sin_feats = feats + self.biases[0].reshape(
            1, self.n_freqs, self.dim_multiplier, 1, 1,
        )
        cos_feats = feats + self.biases[1].reshape(
            1, self.n_freqs, self.dim_multiplier, 1, 1,
        )

        sin_feats = sin_feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w)
        cos_feats = cos_feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w)

        if self.color_feats:
            all_feats = [torch.sin(sin_feats), torch.cos(cos_feats), original_image]
        else:
            all_feats = [torch.sin(sin_feats), torch.cos(cos_feats)]

        return torch.cat(all_feats, dim=1)


class LoftUpUpscaler(nn.Module):
    """
    LoftUp-based upscaler that produces high-resolution mask features.

    Takes (lr_feats, images) and returns (fpn, mask_feats):
      - fpn:        None (the decoder uses one native-resolution memory level)
      - mask_feats: high-res features at output_stride via Fourier + cross-attention

    No InputMixer — receives fused visual features directly. Both patch
    tokens and RGB guidance stay in the input image's actual orientation;
    LoftUp never rotates portrait inputs internally.
    """

    def __init__(
        self,
        input_dim: int,
        dim: int,
        output_stride: int = 2,
        patch_size: int = 14,
        color_feats: bool = True,
        n_freqs: int = 20,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self.output_stride = output_stride
        self.patch_size = patch_size

        if color_feats:
            start_dim = 5 * n_freqs * 2 + 3  # (2 coord + 3 rgb) * n_freqs * 2(sin/cos) + 3(raw rgb)
        else:
            start_dim = 2 * n_freqs * 2

        self.lr_pe = ImplicitFeaturizer(color_feats=False, n_freqs=5)
        concat_dim = input_dim + 2 * 5 * 2

        self.lr_input_proj = nn.Sequential(
            nn.Linear(concat_dim, dim),
            nn.LayerNorm(dim),
        )

        self.fourier_feat = nn.Sequential(
            MinMaxScaler(),
            ImplicitFeaturizer(color_feats, n_freqs=n_freqs),
        )

        self.first_conv = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=start_dim),
            nn.Conv2d(start_dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=dim),
            nn.ReLU(inplace=True),
        )

        self.ca_transformer_blocks = nn.ModuleList([
            CrossonlyDecoderBlock(dim=dim, num_heads=num_heads, mlp_ratio=1)
            for _ in range(num_layers)
        ])
        self.ca_transformer_norm = nn.LayerNorm(dim)

    def forward(self, inputs, img_shape):
        """
        Args:
            inputs: (lr_feats, img)
                lr_feats: [B, P, C] — fused patch tokens (no InputMixer)
                img:      [B, 3, H, W] — guidance images
            img_shape: (H, W), which must match ``img.shape[-2:]``

        Returns:
            (None, mask_feats) where mask_feats: [B, dim, H_out, W_out] high-res features
        """
        lr_feats, img = inputs
        H, W = (int(img_shape[0]), int(img_shape[1]))

        if tuple(img.shape[-2:]) != (H, W):
            raise ValueError(
                "LoftUp img_shape must match the actual RGB tensor: "
                f"img_shape={(H, W)}, tensor_shape={tuple(img.shape[-2:])}"
            )
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Image shape {(H, W)} must be divisible by patch_size="
                f"{self.patch_size}"
            )

        ph, pw = H // self.patch_size, W // self.patch_size
        expected_patches = ph * pw
        if lr_feats.shape[1] != expected_patches:
            raise ValueError(
                "LoftUp patch-token count does not match the image grid: "
                f"got P={lr_feats.shape[1]}, expected {ph}*{pw}="
                f"{expected_patches} for image shape {(H, W)}"
            )
        lr_feats_2d = lr_feats.transpose(-1, -2).view(
            lr_feats.shape[0], -1, ph, pw
        )

        if self.output_stride <= 0:
            raise ValueError(
                f"output_stride must be positive, got {self.output_stride}"
            )
        if H % self.output_stride != 0 or W % self.output_stride != 0:
            raise ValueError(
                f"Image shape {(H, W)} must be divisible by output_stride="
                f"{self.output_stride}"
            )
        if self.output_stride != 1:
            img = F.interpolate(
                img, size=(H // self.output_stride, W // self.output_stride),
                mode="bilinear", align_corners=False,
            )

        x = self.fourier_feat(img)
        x = self.first_conv(x)
        B, Ch, Hout, Wout = x.shape
        x = x.flatten(2).transpose(-1, -2)  # [B, Hout*Wout, Ch]

        lr_pe = self.lr_pe(lr_feats_2d)
        lr_feats_pe = torch.cat([lr_feats_2d, lr_pe], dim=1)
        lr_feats_pe = lr_feats_pe.flatten(2).permute(0, 2, 1)

        lr_feats_pe = self.lr_input_proj(lr_feats_pe)  # [B, ph*pw, dim]

        for blk in self.ca_transformer_blocks:
            x = blk(x, lr_feats_pe)
        x = self.ca_transformer_norm(x)

        x = x.transpose(-1, -2).reshape(B, Ch, Hout, Wout)

        return None, x
