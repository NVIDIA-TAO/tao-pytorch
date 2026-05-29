# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ViTDet backbone with Simple Feature Pyramid (SFP) for CoDETR.

This implements the ViTDet-style Vision Transformer used in Co-DETR, which
differs from standard ViT by using:
- Hybrid window + global attention blocks
- SwiGLU feed-forward network
- 2D Rotary Position Embeddings (RoPE)
- Simple Feature Pyramid neck that converts single-scale output to multi-scale

Reference: Co-DETR/projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE) for 2D vision
# ---------------------------------------------------------------------------

def _rotate_half(x):
    """Rotate pairs of elements: [a, b, c, d, ...] -> [-b, a, -d, c, ...]."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


class VisionRotaryEmbeddingFast(nn.Module):
    """Pre-computed 2D rotary position embedding for fixed spatial size."""

    def __init__(self, dim, pt_seq_len=16, ft_seq_len=None, theta=10000):
        super().__init__()
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        t = torch.arange(ft_seq_len).float() / ft_seq_len * pt_seq_len
        freqs = torch.einsum("i,j->ij", t, freqs)           # (S, dim//4)
        freqs = freqs.repeat_interleave(2, dim=-1)           # (S, dim//2)

        freqs_2d = torch.cat([
            freqs[:, None, :].expand(ft_seq_len, ft_seq_len, -1),
            freqs[None, :, :].expand(ft_seq_len, ft_seq_len, -1),
        ], dim=-1)                                            # (S, S, dim)
        freqs_2d = freqs_2d.reshape(-1, freqs_2d.shape[-1])  # (S*S, dim)

        self.register_buffer("freqs_cos", freqs_2d.cos())
        self.register_buffer("freqs_sin", freqs_2d.sin())

    def forward(self, t):
        """Forward pass for the VisionRotaryEmbeddingFast."""
        return t * self.freqs_cos + _rotate_half(t) * self.freqs_sin


def _get_rope_dynamic(t, H, W, dim=32, pt_seq_len=16):
    """Compute RoPE on-the-fly for variable spatial sizes (global attention)."""
    theta = 10000
    device = t.device

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: dim // 2].float() / dim))
    tH = torch.arange(H, device=device).float() / H * pt_seq_len
    tW = torch.arange(W, device=device).float() / W * pt_seq_len

    freqsH = torch.einsum("i,j->ij", tH, freqs).repeat_interleave(2, dim=-1)
    freqsW = torch.einsum("i,j->ij", tW, freqs).repeat_interleave(2, dim=-1)

    freqs_2d = torch.cat([
        freqsH[:, None, :].expand(H, W, -1),
        freqsW[None, :, :].expand(H, W, -1),
    ], dim=-1).reshape(-1, freqsH.shape[-1] * 2)

    return t * freqs_2d.cos() + _rotate_half(t) * freqs_2d.sin()


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image to patch embedding via convolution."""

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """Forward pass for the PatchEmbed."""
        return self.proj(x).permute(0, 2, 3, 1)   # B C H W -> B H W C


# ---------------------------------------------------------------------------
# Window partition / un-partition
# ---------------------------------------------------------------------------

def _window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    return (x.permute(0, 1, 3, 2, 4, 5).contiguous()
             .view(-1, window_size, window_size, C)), (Hp, Wp)


def _window_unpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = (
        windows.view(
            B, Hp // window_size, Wp // window_size,
            window_size, window_size, -1).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1))
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


# ---------------------------------------------------------------------------
# Absolute positional embedding interpolation
# ---------------------------------------------------------------------------

def _get_abs_pos(abs_pos, has_cls_token, hw):
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num
    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w), mode="bicubic", align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    return abs_pos.reshape(1, h, w, -1)


# ---------------------------------------------------------------------------
# SwiGLU feed-forward
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (gated linear unit with SiLU)."""

    def __init__(self, in_features, hidden_features, out_features=None,
                 norm_layer=nn.LayerNorm, subln=False, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward pass for the SwiGLU."""
        hidden = self.act(self.w1(x)) * self.w2(x)
        return self.drop(self.w3(self.ffn_ln(hidden)))


# ---------------------------------------------------------------------------
# Attention with RoPE
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head attention with 2D rotary position embedding."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, rope=None, rope_half_dim=32):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.rope = rope
        self.rope_half_dim = rope_half_dim
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """Forward pass for the Attention."""
        B, H, W, C = x.shape
        N = H * W
        x_flat = x.view(B, N, C)

        q = F.linear(x_flat, self.q_proj.weight, self.q_bias)
        k = F.linear(x_flat, self.k_proj.weight, None)
        v = F.linear(x_flat, self.v_proj.weight, self.v_bias)

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        if self.rope is not None:
            q = self.rope(q).type_as(v)
            k = self.rope(k).type_as(v)
        else:
            q = _get_rope_dynamic(q, H, W, dim=self.rope_half_dim).type_as(v)
            k = _get_rope_dynamic(k, H, W, dim=self.rope_half_dim).type_as(v)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        return x.view(B, H, W, C)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Transformer block with optional window attention."""

    def __init__(self, dim, num_heads, mlp_ratio=4 * 2 / 3, qkv_bias=True,
                 drop_path=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 window_size=0, rope=None, rope_half_dim=32,
                 use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              rope=rope, rope_half_dim=rope_half_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = SwiGLU(in_features=dim,
                          hidden_features=int(dim * mlp_ratio),
                          subln=True, norm_layer=norm_layer)
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint

    def _forward_impl(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = _window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = _window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x):
        """Forward pass for the Block."""
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


# ---------------------------------------------------------------------------
# LayerNorm for channels-first tensors (used by SFP)
# ---------------------------------------------------------------------------

class _LayerNormCF(nn.Module):
    """LayerNorm for (B, C, H, W) tensors."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


# ---------------------------------------------------------------------------
# Simple Feature Pyramid (SFP) neck
# ---------------------------------------------------------------------------

class SimpleFeaturePyramid(nn.Module):
    """Converts single-scale ViT output to 5-level feature pyramid."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        LN = _LayerNormCF

        self.p2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            LN(in_channels // 2), nn.GELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1, bias=False),
            LN(in_channels // 4), nn.GELU(),
            nn.Conv2d(in_channels // 4, out_channels, 1, bias=False),
            LN(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            LN(out_channels),
        )
        self.p3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            LN(in_channels // 2), nn.GELU(),
            nn.Conv2d(in_channels // 2, out_channels, 1, bias=False),
            LN(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            LN(out_channels),
        )
        self.p4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            LN(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            LN(out_channels),
        )
        self.p5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            LN(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            LN(out_channels),
        )
        self.p6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, bias=False),
            LN(in_channels), nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            LN(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            LN(out_channels),
        )

    def forward(self, x):
        """Forward pass for the SimpleFeaturePyramid."""
        return [self.p2(x), self.p3(x), self.p4(x), self.p5(x), self.p6(x)]


# ---------------------------------------------------------------------------
# ViTDet backbone
# ---------------------------------------------------------------------------

class ViTDetBackbone(nn.Module):
    """ViTDet backbone with SFP neck for CoDETR.

    Produces a 5-level feature pyramid via ``forward_feature_pyramid()``.
    """

    def __init__(self, img_size=1536, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4 * 2 / 3, qkv_bias=True,
                 drop_path_rate=0.4, window_size=24,
                 window_block_indexes=(), pretrain_img_size=512,
                 pretrain_use_cls_token=True,
                 pt_hw_seq_len=16, intp_freq=True,
                 sfp_out_channels=256,
                 use_act_checkpoint=False):
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dim)

        # Absolute positional embedding (sized for pretrain resolution)
        num_patches = (pretrain_img_size // patch_size) ** 2
        num_positions = num_patches + 1 if pretrain_use_cls_token else num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))

        # RoPE for window attention (pre-computed for fixed window size)
        half_head_dim = embed_dim // num_heads // 2
        self.rope_win = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=pt_hw_seq_len,
            ft_seq_len=window_size if intp_freq else None,
        )
        # Global attention uses dynamic RoPE (rope=None in those blocks)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop_path=dpr[i],
                window_size=window_size if i in window_block_indexes else 0,
                rope=self.rope_win if i in window_block_indexes else None,
                rope_half_dim=half_head_dim,
                use_checkpoint=use_act_checkpoint,
            ))

        self.out_norm = nn.LayerNorm(embed_dim)

        # Simple Feature Pyramid neck
        self.sfp = SimpleFeaturePyramid(embed_dim, sfp_out_channels)
        self.num_features = [sfp_out_channels] * 5

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_feature_pyramid(self, x):
        """Return 5-level feature pyramid as dict {layer0..layer4}."""
        x = self.patch_embed(x)
        x = x + _get_abs_pos(
            self.pos_embed, self.pretrain_use_cls_token,
            (x.shape[1], x.shape[2]))
        for blk in self.blocks:
            x = blk(x)

        x = self.out_norm(x).permute(0, 3, 1, 2).contiguous()  # B H W C -> B C H W
        feats = self.sfp(x)
        return {f'layer{i}': feat for i, feat in enumerate(feats)}

    def forward(self, x):
        """Forward pass for the ViTDetBackbone."""
        return self.forward_feature_pyramid(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_VITDET_WINDOW_BLOCK_INDEXES = (
    list(range(0, 3)) + list(range(4, 7)) + list(range(8, 11)) +
    list(range(12, 15)) + list(range(16, 19)) + list(range(20, 23))
)


@BACKBONE_REGISTRY.register()
def vit_large_codetr(**kwargs):
    """ViT-Large backbone for CoDETR (ViTDet + SFP, 5-scale)."""
    activation_checkpoint = kwargs.pop("activation_checkpoint", False)
    # Consume and ignore backbone_v2 kwargs that don't apply here
    kwargs.pop("freeze_at", None)
    kwargs.pop("freeze_norm", None)
    kwargs.pop("num_classes", None)
    kwargs.pop("export", None)
    kwargs.pop("out_indices", None)
    return ViTDetBackbone(
        img_size=1536,
        pretrain_img_size=512,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        drop_path_rate=0.4,
        window_size=24,
        window_block_indexes=_VITDET_WINDOW_BLOCK_INDEXES,
        qkv_bias=True,
        sfp_out_channels=256,
        use_act_checkpoint=activation_checkpoint,
        **kwargs,
    )
