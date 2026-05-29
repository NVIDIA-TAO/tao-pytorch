# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dilated Neighborhood Attention Transformer (DiNAT) backbone module.

This module provides DiNAT implementations for the TAO PyTorch framework.
DiNAT extends the Neighborhood Attention Transformer (NAT) with dilated
attention patterns, enabling larger receptive fields without increasing
computational cost.

The DiNAT architecture was introduced in "Dilated Neighborhood Attention
Transformer" by Hassani and Shi. This implementation provides a hierarchical
vision transformer with dilated neighborhood attention that balances local
and global context efficiently.

Key Features:
- Dilated neighborhood attention for efficient long-range modeling
- Hierarchical multi-scale feature extraction
- PyTorch fallback when NATTEN backend is unavailable
- Compatible with the TAO backbone registry

Factory Functions:
- dinat_large_kernel7: DiNAT-Large with 7x7 kernel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

from nvidia_tao_pytorch.cv.backbone_v2.registry import BACKBONE_REGISTRY

try:
    from natten import NeighborhoodAttention2D as NeighborhoodAttention
    _NATTEN_AVAILABLE = True
except ImportError:
    _NATTEN_AVAILABLE = False
    NeighborhoodAttention = None


def _to_2d_dilation(dilation):
    """Convert dilation to ``(d_h, d_w)`` for 2D.

    Args:
        dilation: ``None`` (use (1, 1)), int (square), length-1 sequence (square),
            or length-2 sequence ``(d_h, d_w)``.

    Returns:
        tuple: ``(d_h, d_w)`` integers.
    """
    if dilation is None:
        return (1, 1)
    if isinstance(dilation, int):
        return (dilation, dilation)
    d = list(dilation)
    if len(d) == 1:
        return (d[0], d[0])
    return (d[0], d[1])


class NeighborhoodAttentionPyTorchFallback(nn.Module):
    """PyTorch fallback for 2D neighborhood attention when NATTEN backend is unavailable.
    Matches NATTEN NeighborhoodAttention2D interface; input/output shape (B, H, W, C).
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        dilation=None,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        proj_drop=0.0,
    ):
        """Initialize fallback neighborhood attention.

        Args:
            dim (int): Channel dimension (must be divisible by ``num_heads``).
            kernel_size (int): Neighborhood window size. Default: 7.
            dilation: Per-axis dilation for the neighborhood; passed through
                :func:`_to_2d_dilation`. Default: None (1, 1).
            num_heads (int): Number of attention heads. Default: 8.
            qkv_bias (bool): If True, add bias to the QKV linear. Default: True.
            qk_scale (float, optional): Manual softmax scale; default ``head_dim ** -0.5``.
            proj_drop (float): Dropout on the output projection. Default: 0.0.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or (self.head_dim ** -0.5)
        self.kernel_size = kernel_size
        self.dilation = _to_2d_dilation(dilation)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """Compute neighborhood attention."""
        B, H, W, C = x.shape
        N, D = self.num_heads, self.head_dim
        k = self.kernel_size
        d_h, d_w = self.dilation
        pad_h = d_h * (k - 1) // 2
        pad_w = d_w * (k - 1) // 2

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, H, W, 3, N, D).permute(3, 0, 4, 5, 1, 2)
        q, k_in, v = qkv[0], qkv[1], qkv[2]

        q_flat = q.permute(0, 2, 3, 1, 4).reshape(B, H * W, N, D)

        k_pad = F.pad(
            k_in.reshape(B, N * D, H, W),
            (pad_w, pad_w, pad_h, pad_h),
            mode="constant",
            value=0,
        )
        v_pad = F.pad(
            v.reshape(B, N * D, H, W),
            (pad_w, pad_w, pad_h, pad_h),
            mode="constant",
            value=0,
        )

        k_unf = F.unfold(k_pad, kernel_size=(k, k), dilation=(d_h, d_w), padding=0)
        v_unf = F.unfold(v_pad, kernel_size=(k, k), dilation=(d_h, d_w), padding=0)

        k_neigh = k_unf.reshape(B, N, D, k * k, H * W).permute(0, 4, 3, 1, 2)
        v_neigh = v_unf.reshape(B, N, D, k * k, H * W).permute(0, 4, 3, 1, 2)

        attn = torch.einsum("blnd,blknd->blnk", q_flat, k_neigh) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("blnk,blknd->blnd", attn, v_neigh)

        out = out.reshape(B, H, W, C)
        return self.proj_drop(self.proj(out))


class ConvTokenizer(nn.Module):
    """Convolutional tokenizer for input images."""

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        """Initialize two-stride conv patch embedding.

        Args:
            in_chans (int): Number of input image channels. Default: 3.
            embed_dim (int): Output channel dimension per spatial location. Default: 96.
            norm_layer (callable, optional): Constructor ``norm_layer(embed_dim)`` applied
                after convs; if None, no normalization.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        """Tokenize input image into patch embeddings."""
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvDownsampler(nn.Module):
    """Convolutional downsampler between stages."""

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        """Initialize 2x spatial downsampling with doubled channels.

        Args:
            dim (int): Input channel dimension (tokens as (B, H, W, dim)).
            norm_layer (callable): Constructor for ``norm_layer(2 * dim)`` after reduction.
        """
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """Downsample spatial resolution by 2x."""
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    """Multi-layer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        """Initialize two-layer MLP.

        Args:
            in_features (int): Input feature size.
            hidden_features (int, optional): Hidden size; defaults to ``in_features``.
            out_features (int, optional): Output size; defaults to ``in_features``.
            act_layer (callable): Activation module class. Default: ``nn.GELU``.
            drop (float): Dropout probability after each linear. Default: 0.0.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Apply two-layer MLP with activation and dropout."""
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class NATLayer(nn.Module):
    """Neighborhood Attention Transformer layer."""

    def __init__(self, dim, num_heads, kernel_size=7, dilation=None,
                 mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        """Initialize one NAT transformer layer (attention + MLP).

        Args:
            dim (int): Channel dimension.
            num_heads (int): Attention heads (``dim`` must be divisible by this).
            kernel_size (int): Neighborhood kernel size. Default: 7.
            dilation: Attention dilation; see neighborhood attention modules.
            mlp_ratio (float): Hidden MLP dimension is ``int(dim * mlp_ratio)``. Default: 4.0.
            qkv_bias (bool): Bias on QKV projection. Default: True.
            qk_scale (float, optional): Attention scale override.
            drop (float): Dropout on projection and MLP. Default: 0.0.
            attn_drop (float): Reserved for attention dropout (NATTEN path). Default: 0.0.
            drop_path (float): Stochastic depth rate for residual branches. Default: 0.0.
            act_layer (callable): MLP activation. Default: ``nn.GELU``.
            norm_layer (callable): Pre-norm constructor, e.g. ``nn.LayerNorm``.
            layer_scale (float or int, optional): If set, enables LayerScale with this init.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self._attn_kwargs = {
            "dim": dim, "kernel_size": kernel_size, "dilation": dilation,
            "num_heads": num_heads, "qkv_bias": qkv_bias, "qk_scale": qk_scale,
            "proj_drop": drop,
        }

        self.norm1 = norm_layer(dim)
        if _NATTEN_AVAILABLE and NeighborhoodAttention is not None:
            self.attn = NeighborhoodAttention(**self._attn_kwargs)
        else:
            self.attn = NeighborhoodAttentionPyTorchFallback(**self._attn_kwargs)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def _replace_with_fallback(self, device, dtype):
        """Replace NATTEN attention with PyTorch fallback when backend is unavailable."""
        fallback = NeighborhoodAttentionPyTorchFallback(**self._attn_kwargs)
        fallback = fallback.to(device=device, dtype=dtype)
        state = self.attn.state_dict()
        fallback_state = fallback.state_dict()
        for k, v in state.items():
            if k in fallback_state and fallback_state[k].shape == v.shape:
                fallback_state[k].copy_(v)
        fallback.load_state_dict(fallback_state, strict=False)
        self.attn = fallback

    def _forward_attn(self, x):
        try:
            return self.attn(x)
        except NotImplementedError:
            if isinstance(self.attn, NeighborhoodAttentionPyTorchFallback):
                raise
            self._replace_with_fallback(x.device, x.dtype)
            return self.attn(x)

    def forward(self, x):
        """Apply neighborhood attention with residual and MLP."""
        if not self.layer_scale:
            shortcut = x
            x = self._forward_attn(self.norm1(x))
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self._forward_attn(self.norm1(x))
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class NATBlock(nn.Module):
    """Neighborhood Attention Transformer block (one stage)."""

    def __init__(self, dim, depth, num_heads, kernel_size, dilations=None,
                 downsample=True, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm, layer_scale=None):
        """Initialize a stage of stacked ``NATLayer`` blocks.

        Args:
            dim (int): Channel dimension for this stage.
            depth (int): Number of ``NATLayer`` instances.
            num_heads (int): Heads per layer.
            kernel_size (int): Neighborhood kernel size.
            dilations (list[int], optional): Length-``depth`` dilation per layer; if None, 1.
            downsample (bool): If True, apply ``ConvDownsampler`` after the stage.
            mlp_ratio (float): MLP expansion ratio passed to each layer.
            qkv_bias (bool): QKV bias flag for each layer.
            qk_scale (float, optional): Attention scale for each layer.
            drop (float): Dropout rate for each layer.
            attn_drop (float): Attention dropout for each layer.
            drop_path: Scalar or list of length ``depth`` for per-layer drop path.
            norm_layer (callable): Norm constructor for layers and downsampler.
            layer_scale (float or int, optional): LayerScale init for each ``NATLayer``.
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList([
            NATLayer(
                dim=dim, num_heads=num_heads, kernel_size=kernel_size,
                dilation=None if dilations is None else dilations[i],
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, layer_scale=layer_scale,
            )
            for i in range(depth)
        ])
        self.downsample = None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        """Process input through all layers and optional downsampling."""
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is None:
            return x, x
        return self.downsample(x), x


class DiNAT(nn.Module):
    """Dilated Neighborhood Attention Transformer.

    A hierarchical vision transformer backbone that uses dilated neighborhood
    attention for efficient multi-scale feature extraction. See ``__init__`` for
    full constructor arguments.
    """

    def __init__(
        self,
        embed_dim=192,
        mlp_ratio=2.0,
        depths=(3, 4, 18, 5),
        num_heads=(6, 12, 24, 48),
        drop_path_rate=0.2,
        in_chans=3,
        kernel_size=7,
        dilations=None,
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        frozen_stages=-1,
        layer_scale=None,
        **kwargs,
    ):
        """Construct DiNAT.

        Args:
            embed_dim (int): Embedding dimension at the first stage. Default: 192.
            mlp_ratio (float): MLP expansion ratio in each ``NATLayer``. Default: 2.0.
            depths (tuple[int, ...]): Layer count per stage. Default: (3, 4, 18, 5).
            num_heads (tuple[int, ...]): Heads per stage; length must match ``depths``.
                Default: (6, 12, 24, 48).
            drop_path_rate (float): Maximum stochastic depth rate (linear schedule).
                Default: 0.2.
            in_chans (int): Input image channels. Default: 3.
            kernel_size (int): Neighborhood attention kernel size. Default: 7.
            dilations (list[list[int]], optional): Per-layer dilations per stage; if None,
                all dilations are 1.
            out_indices (tuple[int, ...]): Stage indices (0-based) whose pre-downsample
                outputs are returned as ``res{idx+2}``. Default: (0, 1, 2, 3).
            qkv_bias (bool): Enable bias on QKV projections. Default: True.
            qk_scale (float, optional): Manual attention scale.
            drop_rate (float): Dropout after patch embed and in blocks. Default: 0.0.
            attn_drop_rate (float): Attention dropout rate in blocks. Default: 0.0.
            norm_layer (callable): Normalization constructor, e.g. ``nn.LayerNorm``.
            frozen_stages (int): If ``>= 0``, freeze ``patch_embed``; if ``>= 2``, also
                freeze levels ``0 .. frozen_stages-2``. Default: -1 (no freezing).
            layer_scale (float or int, optional): LayerScale initial value when set.
            **kwargs: Reserved for forward compatibility (ignored here).
        """
        super().__init__()
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_levels)]
        self.mlp_ratio = mlp_ratio
        self.out_indices = out_indices

        self.patch_embed = ConvTokenizer(in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(
                dim=int(embed_dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size,
                dilations=None if dilations is None else dilations[i],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
            )
            self.levels.append(level)

        for i_layer in self.out_indices:
            layer = norm_layer(self.num_features[i_layer])
            self.add_module(f"norm{i_layer}", layer)

        self.frozen_stages = frozen_stages

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 2:
            for i in range(0, self.frozen_stages - 1):
                m = self.levels[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        """Set training mode and re-freeze stages if configured."""
        super().train(mode)
        self._freeze_stages()

    def forward(self, x):
        """Forward pass. Returns dict of multi-scale features keyed as 'res2'..'res5'."""
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        outs = {}
        for idx, level in enumerate(self.levels):
            x, xo = level(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(xo)
                outs[f"res{idx + 2}"] = x_out.permute(0, 3, 1, 2).contiguous()
        return outs


# ---------------------------------------------------------------------------
# Factory functions for backbone registry
# ---------------------------------------------------------------------------

@BACKBONE_REGISTRY.register()
def dinat_large_kernel7(**kwargs):
    """Create a DiNAT-Large model with 7x7 kernel.

    Configuration:
    - Embedding dimension: 192
    - MLP ratio: 2.0
    - Depths: (3, 4, 18, 5)
    - Number of heads: (6, 12, 24, 48)
    - Kernel size: 7
    - Default dilations for large receptive field

    Args:
        **kwargs: Additional arguments passed to DiNAT constructor.

    Returns:
        DiNAT: Configured DiNAT-Large model.
    """
    default_dilations = [
        [1, 20, 1],
        [1, 5, 1, 10],
        [1, 2, 1, 3, 1, 4, 1, 5, 1, 2, 1, 3, 1, 4, 1, 5, 1, 5],
        [1, 2, 1, 2, 1],
    ]
    kwargs.setdefault("dilations", default_dilations)
    return DiNAT(
        embed_dim=192,
        mlp_ratio=2.0,
        depths=(3, 4, 18, 5),
        num_heads=(6, 12, 24, 48),
        kernel_size=7,
        drop_path_rate=0.3,
        **kwargs,
    )
