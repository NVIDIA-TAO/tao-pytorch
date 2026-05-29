# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GCViT backbone module.

This module provides GCViT (Global Context Vision Transformer) implementations
for the TAO PyTorch framework. GCViT is a hierarchical vision transformer that
combines local window attention with global context modeling.

The GCViT architecture introduces a novel attention mechanism that captures both
local and global information efficiently. It uses window-based attention for local
feature extraction and global context modeling for capturing long-range dependencies.

Key Features:
- Hierarchical architecture with multiple stages
- Window-based local attention for efficiency
- Global context modeling for long-range dependencies
- Support for multiple model sizes (XXTiny, XTiny, Tiny, Small, Base, Large)
- Integration with TAO backbone framework
- Support for activation checkpointing and layer freezing
- Efficient design with good accuracy/speed balance

Classes:
    Mlp: Multi-Layer Perceptron block
    SqueezeExcitation: Squeeze and excitation block
    ReduceSize: Feature reduction layer
    PatchEmbed: Patch embedding layer
    FeatExtract: Feature extraction layer
    WindowAttention: Window-based attention mechanism
    WindowAttentionGlobal: Global window attention
    GCViTBlock: GCViT transformer block
    GlobalQueryGen: Global query generator
    GCViTLayer: GCViT layer with multiple blocks
    GCViT: Main GCViT model

Functions:
    window_partition: Partition input into windows
    window_reverse: Reverse window partitioning
    gc_vit_xxtiny: GCViT XXTiny model
    gc_vit_xtiny: GCViT XTiny model
    gc_vit_tiny: GCViT Tiny model
    gc_vit_small: GCViT Small model
    gc_vit_base: GCViT Base model
    gc_vit_large: GCViT Large model
    gc_vit_base_384: GCViT Base model for 384x384 images
    gc_vit_large_384: GCViT Large model for 384x384 images

Example:
    >>> from nvidia_tao_pytorch.cv.backbone_v2 import gc_vit_tiny
    >>> model = gc_vit_tiny(num_classes=1000)
    >>> x = torch.randn(1, 3, 224, 224)
    >>> output = model(x)

References:
    - [GCViT: Global Context Vision Transformer](https://arxiv.org/abs/2206.09959)
    - [https://github.com/NVlabs/GCViT](https://github.com/NVlabs/GCViT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase


def window_partition(x, window_size, h_w, w_w):
    """Partition input tensor into windows for local attention computation.

    This function divides the input tensor into non-overlapping windows to enable
    efficient local attention computation. Each window is processed independently
    to reduce computational complexity.

    Args:
        x (torch.Tensor): Input tensor of shape (B, H, W, C) where B is batch size,
            H and W are height and width, and C is the number of channels.
        window_size (int): Size of each window (assumed to be square).
        h_w (int): Number of windows in height dimension.
        w_w (int): Number of windows in width dimension.

    Returns:
        torch.Tensor: Windowed features of shape (num_windows*B, window_size, window_size, C)
            where num_windows = h_w * w_w.

    Example:
        >>> x = torch.randn(2, 56, 56, 96)
        >>> windows = window_partition(x, 7, 8, 8)
        >>> print(windows.shape)  # torch.Size([128, 7, 7, 96])
    """
    B, _, _, C = x.shape
    x = x.view(B, h_w, window_size, w_w, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W, h_w, w_w):
    """Reverse window partitioning to reconstruct the original tensor.

    This function reconstructs the original tensor from windowed features by
    reversing the window partitioning operation.

    Args:
        windows (torch.Tensor): Windowed features of shape (num_windows*B, window_size, window_size, C).
        window_size (int): Size of each window (assumed to be square).
        H (int): Height of the original image.
        W (int): Width of the original image.
        h_w (int): Number of windows in height dimension.
        w_w (int): Number of windows in width dimension.

    Returns:
        torch.Tensor: Reconstructed tensor of shape (B, H, W, C).

    Example:
        >>> windows = torch.randn(128, 7, 7, 96)
        >>> x = window_reverse(windows, 7, 56, 56, 8, 8)
        >>> print(x.shape)  # torch.Size([2, 56, 56, 96])

    Note:
        The batch size B is calculated as windows.shape[0] // (H * W // window_size // window_size).
    """
    # Casting to int leads to error
    B = windows.shape[0] // (H * W // window_size // window_size)
    x = windows.view(B, h_w, w_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    """Multi-Layer Perceptron (MLP) block.

    This class implements a standard MLP block with two linear layers, an activation
    function, and dropout layers. It is commonly used in transformer architectures
    as the feed-forward network component.

    Args:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of hidden features. If None, defaults
            to in_features. Default: `None`.
        out_features (int, optional): Number of output features. If None, defaults
            to in_features. Default: `None`.
        act_layer (nn.Module, optional): Activation function. Default: `nn.GELU`.
        drop (float, optional): Dropout rate. Default: `0.0`.

    Attributes:
        fc1 (nn.Linear): First linear layer.
        act (nn.Module): Activation function.
        fc2 (nn.Linear): Second linear layer.
        drop (nn.Dropout): Dropout layer.

    Example:
        >>> mlp = Mlp(in_features=256, hidden_features=512, out_features=256)
        >>> x = torch.randn(1, 256)
        >>> output = mlp(x)  # Shape: (1, 256)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        """Initialize the MLP block.

        Args:
            in_features (int): Number of input features.
            hidden_features (int, optional): Number of hidden features. If None, defaults
                to in_features. Default: `None`.
            out_features (int, optional): Number of output features. If None, defaults
                to in_features. Default: `None`.
            act_layer (nn.Module, optional): Activation function. Default: `nn.GELU`.
            drop (float, optional): Dropout rate. Default: `0.0`.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward pass through the MLP block.

        This method applies the MLP transformation: Linear -> Activation -> Dropout -> Linear -> Dropout.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C) where B is batch size,
                N is sequence length, and C is feature dimension.

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SqueezeExcitation(nn.Module):
    """
    Squeeze and excitation block
    """

    def __init__(self, inp, oup, expansion=0.25):
        """
        Args:
            inp: input features dimension.
            oup: output features dimension.
            expansion: expansion ratio.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward function."""
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ReduceSize(nn.Module):
    """
    Down-sampling block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, keep_dim=False):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.GELU(),
            SqueezeExcitation(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False)
        self.norm2 = norm_layer(dim_out)
        self.norm1 = norm_layer(dim)

    def forward(self, x):
        """Forward function."""
        x = x.contiguous()
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)  # To channel first.
        x = x + self.conv(x)
        x = self.reduction(x)
        x = x.permute(0, 2, 3, 1)  # To channel last.
        x = self.norm2(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self, in_chans=3, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        super().__init__()
        self.proj = nn.Conv2d(in_chans, dim, 3, 2, 1)
        self.conv_down = ReduceSize(dim=dim, keep_dim=True)

    def forward(self, x):
        """Forward function."""
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # To channel last.
        x = self.conv_down(x)
        return x


class FeatExtract(nn.Module):
    """
    Feature extraction block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self, dim, keep_dim=False):
        """
        Args:
            dim: feature size dimension.
            keep_dim: bool argument for maintaining the resolution.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.GELU(),
            SqueezeExcitation(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        if not keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.keep_dim = keep_dim

    def forward(self, x):
        """Forward function."""
        x = x.contiguous()
        x = x + self.conv(x)
        if not self.keep_dim:
            x = self.pool(x)
        return x


class WindowAttention(nn.Module):
    """Local window attention based on: "Liu et al.,

    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_rel_pos_bias=True,
    ):
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            window_size: window size.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
            use_rel_pos_bias: set bias for relative positional embedding
        """
        super().__init__()
        window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads, rounding_mode="floor")
        self.fast_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")  # pylint:disable=I1101
        self.scale = qk_scale or head_dim**-0.5
        self.use_rel_pos_bias = use_rel_pos_bias

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q_global):
        """Forward function."""
        B_, N, C = x.shape
        head_dim = torch.div(C, self.num_heads, rounding_mode="floor")
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if torch.onnx.is_in_onnx_export() or not self.fast_attn or self.use_rel_pos_bias:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if self.use_rel_pos_bias:
                relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
                )  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn = attn + relative_position_bias.unsqueeze(0)
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        else:
            # Since Torch 1.14, scaled_dot_product_attention has been optimized for performance
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p,
            )
            x = x.transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttentionGlobal(nn.Module):
    """Global window attention based on: "Hatamizadeh et al.,

    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_rel_pos_bias=True,
    ):
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            window_size: window size.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
            use_rel_pos_bias: set bias for relative positional embedding
        """
        super().__init__()
        window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads, rounding_mode="floor")
        self.scale = qk_scale or head_dim**-0.5
        self.fast_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")  # pylint:disable=I1101

        self.use_rel_pos_bias = use_rel_pos_bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q_global):
        """Forward function."""
        B_, N, C = x.shape
        B = q_global.shape[0]
        head_dim = torch.div(C, self.num_heads, rounding_mode="floor")
        B_dim = torch.div(B_, B, rounding_mode="floor")
        kv = self.qkv(x).reshape(B_, N, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q_global = q_global.repeat(1, B_dim, 1, 1, 1)
        q = q_global.reshape(B_, self.num_heads, N, head_dim)

        if torch.onnx.is_in_onnx_export() or not self.fast_attn or self.use_rel_pos_bias:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if self.use_rel_pos_bias:
                relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
                )  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn = attn + relative_position_bias.unsqueeze(0)

            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        else:
            # Since Torch 1.14, scaled_dot_product_attention has been optimized for performance
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p,
            )
            x = x.transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GCViTBlock(nn.Module):
    """
    GCViT block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        attention=WindowAttentionGlobal,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        use_rel_pos_bias=True,
    ):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            num_heads: number of attention head.
            window_size: window size.
            mlp_ratio: MLP ratio.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            act_layer: activation function.
            attention: attention block type.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            use_rel_pos_bias: set bias for relative positional embedding
        """
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)

        self.attn = attention(
            dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_rel_pos_bias=use_rel_pos_bias,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0

        inp_w = torch.div(input_resolution, window_size, rounding_mode="floor")
        self.num_windows = int(inp_w * inp_w)

    def forward(self, x, q_global):
        """Forward function."""
        _, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        h_w = torch.div(H, self.window_size, rounding_mode="floor")
        w_w = torch.div(W, self.window_size, rounding_mode="floor")
        x_windows = window_partition(x, self.window_size, h_w, w_w)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, q_global)
        x = window_reverse(attn_windows, self.window_size, H, W, h_w, w_w)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class GlobalQueryGen(nn.Module):
    """Global query generator based on: "Hatamizadeh et al.,

    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self, dim, input_resolution, image_resolution, window_size, num_heads):
        """
        For instance, repeating log(56/7) = 3 blocks, with input window dimension 56 and output window dimension 7 at
        down-sampling ratio 2. Please check Fig.5 of GC ViT paper for details.

        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            window_size: window size.
            num_heads: number of heads.
        """
        super().__init__()
        if input_resolution == image_resolution // 4:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
            )

        elif input_resolution == image_resolution // 8:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
            )

        elif input_resolution == image_resolution // 16:
            if window_size == input_resolution:
                self.to_q_global = nn.Sequential(FeatExtract(dim, keep_dim=True))

            else:
                self.to_q_global = nn.Sequential(FeatExtract(dim, keep_dim=False))

        elif input_resolution == image_resolution // 32:
            self.to_q_global = nn.Sequential(FeatExtract(dim, keep_dim=True))

        self.num_heads = num_heads
        self.N = window_size * window_size
        self.dim_head = torch.div(dim, self.num_heads, rounding_mode="floor")
        self.window_size = window_size

    def forward(self, x):
        """Foward function."""
        x = self.to_q_global(x)
        B, _, H, W = x.shape
        if self.window_size != H or self.window_size != W:
            x = F.interpolate(x, size=(self.window_size, self.window_size), mode="bicubic")
        x = x.permute(0, 2, 3, 1)  # To channel last.
        x = x.reshape(B, 1, self.N, self.num_heads, self.dim_head).permute(0, 1, 3, 2, 4)
        return x


class GCViTLayer(nn.Module):
    """
    GCViT layer based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(
        self,
        dim,
        depth,
        input_resolution,
        image_resolution,
        num_heads,
        window_size,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        use_rel_pos_bias=True,
    ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            input_resolution: input image resolution.
            window_size: window size in each stage.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            use_rel_pos_bias: set bias for relative positional embedding
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                GCViTBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attention=WindowAttention if (i % 2 == 0) else WindowAttentionGlobal,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                    input_resolution=input_resolution,
                    use_rel_pos_bias=use_rel_pos_bias,
                )
                for i in range(depth)
            ]
        )
        self.downsample = None if not downsample else ReduceSize(dim=dim, norm_layer=norm_layer)
        self.q_global_gen = GlobalQueryGen(dim, input_resolution, image_resolution, window_size, num_heads)

    def forward(self, x):
        """Forward function."""
        q_global = self.q_global_gen(x.permute(0, 3, 1, 2))  # To channel first.
        for blk in self.blocks:
            x = blk(x, q_global)
        if self.downsample is None:
            return x
        return self.downsample(x)


class GCViT(BackboneBase):
    """Global context vision transformer (GCViT) model.

    GCViT is a novel architecture that enhances parameter and compute utilization for computer vision. It leverages
    global context self-attention modules, joint with standard local self-attention, to effectively and efficiently
    model both long and short-range spatial interactions, without the need for expensive operations such as computing
    attention masks or shifting local windows.

    References:
    - [Global Context Vision Transformers](https://arxiv.org/abs/2206.09959)
    - [https://github.com/NVlabs/GCVit](https://github.com/NVlabs/GCVit)
    """

    def __init__(
        self,
        dim,
        depths,
        window_size,
        mlp_ratio,
        num_heads,
        resolution=224,
        drop_path_rate=0.2,
        in_chans=3,
        num_classes=1000,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        use_rel_pos_bias=True,
        activation_checkpoint=False,
        freeze_at=None,
        freeze_norm=False,
        **kwargs,
    ):
        """Initialize the GCViT model.

        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            resolution: input image resolution.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            use_rel_pos_bias: set bias for relative positional embedding
            activation_checkpoint (bool): Whether to use activation checkpointing. Default: `False`.
            freeze_at (list): List of keys corresponding to the stages or layers to freeze. If `None`, no specific
                layers are frozen. If `"all"`, the entire model is frozen and set to eval mode. Default: `None`.
            freeze_norm (bool): If `True`, all normalization layers in the backbone will be frozen. Default: `False`.
        """
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
        )

        self.num_features = int(dim * 2 ** (len(depths) - 1))  # TODO(@yuw): to verify!

        self.patch_embed = PatchEmbed(in_chans=in_chans, dim=dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            level = GCViTLayer(
                dim=int(dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]): sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=(i < len(depths) - 1),
                layer_scale=layer_scale,
                input_resolution=int(2 ** (-2 - i) * resolution),
                image_resolution=resolution,
                use_rel_pos_bias=use_rel_pos_bias,
            )
            self.levels.append(level)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Set the gradient (activation) checkpointing for the model."""
        if enable:
            raise NotImplementedError("Activation checkpointing is not implemented for GCViT.")
        self.activation_checkpoint = enable

    def get_stage_dict(self):
        """Get the stage dictionary."""
        stage_dict = {}
        # TODO(@yuw, @hongyuc): No stem. Add patch_embed as stage 0?
        for i, level in enumerate(self.levels, start=1):
            stage_dict[i] = level
        return stage_dict

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """Get the set of parameter keywords to exclude from weight decay."""
        return {"rpb"}

    @torch.jit.ignore
    def get_classifier(self):
        """Get the classifier module."""
        return self.head

    def reset_classifier(self, num_classes=0, **kwargs):
        """Reset the classifier head."""
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_pre_logits(self, x):
        """Forward pass through the backbone, excluding the head."""
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for level in self.levels:
            x = level(x)

        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # To channel first.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward_feature_pyramid(self, x):
        """Forward pass through the backbone to extract intermediate feature maps."""
        outs = []
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for level in self.levels:
            x = level(x)
            x_spatial = x.permute(0, 3, 1, 2)  # To channel first.
            outs.append(x_spatial)
        return outs

    def forward(self, x):
        """Forward."""
        x = self.forward_pre_logits(x)
        x = self.head(x)
        return x


@BACKBONE_REGISTRY.register()
def gc_vit_xxtiny(**kwargs):
    """GCViT-XXTiny model."""
    return GCViT(
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 16],
        window_size=[7, 7, 14, 7],
        dim=64,
        mlp_ratio=3,
        drop_path_rate=0.2,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def gc_vit_xtiny(**kwargs):
    """GCViT-XTiny model."""
    return GCViT(
        depths=[3, 4, 6, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[7, 7, 14, 7],
        dim=64,
        mlp_ratio=3,
        drop_path_rate=0.2,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def gc_vit_tiny(**kwargs):
    """GCViT-Tiny model."""
    return GCViT(
        depths=[3, 4, 19, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[7, 7, 14, 7],
        dim=64,
        mlp_ratio=3,
        drop_path_rate=0.2,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def gc_vit_small(**kwargs):
    """GCViT-Small model."""
    return GCViT(
        depths=[3, 4, 19, 5],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7, 14, 7],
        dim=96,
        mlp_ratio=2,
        drop_path_rate=0.3,
        layer_scale=1e-5,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def gc_vit_base(**kwargs):
    """GCViT-Base model."""
    return GCViT(
        depths=[3, 4, 19, 5],
        num_heads=[4, 8, 16, 32],
        window_size=[7, 7, 14, 7],
        dim=128,
        mlp_ratio=2,
        drop_path_rate=0.5,
        layer_scale=1e-5,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def gc_vit_large(**kwargs):
    """GCViT-Large model."""
    return GCViT(
        depths=[3, 4, 19, 5],
        num_heads=[6, 12, 24, 48],
        window_size=[7, 7, 14, 7],
        dim=192,
        mlp_ratio=2,
        drop_path_rate=0.5,
        layer_scale=1e-5,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def gc_vit_base_384(**kwargs):
    """GCViT-Base model with image resolution of 384."""
    return GCViT(
        depths=[3, 4, 19, 5],
        num_heads=[4, 8, 16, 32],
        window_size=[12, 12, 24, 12],
        dim=128,
        mlp_ratio=2,
        drop_path_rate=0.5,
        layer_scale=1e-5,
        resolution=384,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def gc_vit_large_384(**kwargs):
    """GCViT-Large model with image resolution of 384."""
    return GCViT(
        depths=[3, 4, 19, 5],
        num_heads=[6, 12, 24, 48],
        window_size=[12, 12, 24, 12],
        dim=192,
        mlp_ratio=2,
        drop_path_rate=0.5,
        layer_scale=1e-5,
        resolution=384,
        **kwargs,
    )
