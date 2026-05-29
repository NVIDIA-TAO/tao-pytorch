# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EdgeNeXt model utilities.

This module provides the building blocks for EdgeNeXt architecture, a hybrid
CNN-Transformer model designed for mobile vision applications. EdgeNeXt combines
the efficiency of convolutional layers with the representational power of
transformer blocks through innovative attention mechanisms.

The module includes:
- PositionalEncodingFourier: Fourier-based positional encoding for vision transformers
- SDTAEncoder: Split Depth-wise Transpose Attention encoder block
- SDTAEncoderBNHS: SDTA encoder with BatchNorm and Hard-Swish activation
- XCA: Cross-Covariance Attention module for efficient attention computation
- ConvEncoder: Convolutional encoder with inverted bottleneck structure
- ConvEncoderBNHS: Mobile-optimized convolutional encoder

Key Features:
- Split depth-wise convolutions for efficient local feature extraction
- Cross-covariance attention for global information processing
- Fourier-based positional encodings for spatial awareness
- Mobile-optimized variants with BatchNorm and Hard-Swish
- Layer scaling and stochastic depth for improved training

Classes:
    PositionalEncodingFourier: Fourier-based positional encoding
    SDTAEncoder: Split Depth-wise Transpose Attention encoder
    SDTAEncoderBNHS: SDTA encoder with mobile optimizations
    XCA: Cross-Covariance Attention module
    ConvEncoder: Convolutional encoder block
    ConvEncoderBNHS: Mobile-optimized convolutional encoder

Example:
    ```python
    # Create an SDTA encoder block
    encoder = SDTAEncoder(
        dim=96,
        drop_path=0.1,
        expan_ratio=4,
        num_heads=8
    )

    # Forward pass
    x = torch.randn(1, 96, 56, 56)
    output = encoder(x)
    ```
"""

import math

import torch
from torch import nn

from timm.models.layers import DropPath

from nvidia_tao_pytorch.cv.backbone_v2.nn.norm import LayerNorm2d


class PositionalEncodingFourier(nn.Module):
    """Fourier-based positional encoding for vision transformers.

    This module generates 2D positional encodings using Fourier features, which are
    commonly used in vision transformers to provide spatial position information
    to the attention mechanism.

    The encoding is computed using sine and cosine functions with different frequencies,
    providing a rich representation of spatial positions that can be learned by the
    attention mechanism.

    Args:
        hidden_dim (int, optional): Hidden dimension for Fourier features. Defaults to 32.
        dim (int, optional): Output dimension of the positional encoding. Defaults to 768.
        temperature (float, optional): Temperature parameter for the Fourier features.
            Defaults to 10000.
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        """Forward pass for PositionalEncodingFourier.

        Args:
            B (int): Batch size
            H (int): Height of the feature map
            W (int): Width of the feature map

        Returns:
            torch.Tensor: Positional encoding of shape (B, dim, H, W)
        """
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)

        return pos


class SDTAEncoder(nn.Module):
    """Split Depth-wise Transpose Attention (SDTA) Encoder block.

    This encoder combines split depth-wise convolutions with cross-covariance attention (XCA)
    and an inverted bottleneck structure. It includes optional positional embeddings and
    layer scaling for improved training stability.

    The SDTA mechanism splits the input channels and processes them with different
    depth-wise convolutions, then combines them with cross-covariance attention for
    efficient local and global feature processing.

    Args:
        dim (int): Input/output channel dimension.
        drop_path (float, optional): Drop path rate for stochastic depth. Defaults to 0.
        layer_scale_init_value (float, optional): Initial value for layer scale parameters.
            Set to 0 to disable layer scaling. Defaults to 1e-6.
        expan_ratio (int, optional): Expansion ratio for the inverted bottleneck. Defaults to 4.
        use_pos_emb (bool, optional): Whether to use positional embeddings. Defaults to True.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        qkv_bias (bool, optional): Whether to add bias to QKV projections. Defaults to True.
        attn_drop (float, optional): Attention dropout rate. Defaults to 0.
        drop (float, optional): Dropout rate. Defaults to 0.
        scales (int, optional): Number of split scales for depth-wise convolution. Defaults to 1.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4,
                 use_pos_emb=True, num_heads=8, qkv_bias=True, attn_drop=0., drop=0., scales=1):
        super().__init__()
        width = max(int(math.ceil(dim / scales)), int(math.floor(dim // scales)))
        self.width = width
        if scales == 1:
            self.nums = 1
        else:
            self.nums = scales - 1
        convs = []
        for _ in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width))
        self.convs = nn.ModuleList(convs)

        self.pos_embd = None
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=dim)
        self.norm_xca = LayerNorm2d(dim, eps=1e-6)
        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()  # TODO: MobileViT is using 'swish'
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """Forward pass for SDTAEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        x_in = x

        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        x = torch.cat((out, spx[self.nums]), 1)
        # XCA
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        if self.pos_embd:
            pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding
        x = x + self.drop_path(self.gamma_xca * self.xca(self.norm_xca(x)))
        x = x.reshape(B, H, W, C)

        # Inverted Bottleneck
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = x_in + self.drop_path(x)

        return x


class SDTAEncoderBNHS(nn.Module):
    """Split Depth-wise Transpose Attention (SDTA) Encoder with Batch Normalization and Hard-Swish.

    This is a variant of SDTAEncoder that uses Batch Normalization instead of Layer Normalization
    and Hard-Swish activation instead of GELU. This combination is often used for mobile-optimized
    architectures to improve inference efficiency.

    The key differences from SDTAEncoder are:
    - Uses BatchNorm2d instead of LayerNorm for normalization
    - Uses Hard-Swish activation instead of GELU
    - Optimized for mobile hardware acceleration

    Args:
        dim (int): Input/output channel dimension.
        drop_path (float, optional): Drop path rate for stochastic depth. Defaults to 0.
        layer_scale_init_value (float, optional): Initial value for layer scale parameters.
            Set to 0 to disable layer scaling. Defaults to 1e-6.
        expan_ratio (int, optional): Expansion ratio for the inverted bottleneck. Defaults to 4.
        use_pos_emb (bool, optional): Whether to use positional embeddings. Defaults to True.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        qkv_bias (bool, optional): Whether to add bias to QKV projections. Defaults to True.
        attn_drop (float, optional): Attention dropout rate. Defaults to 0.
        drop (float, optional): Dropout rate. Defaults to 0.
        scales (int, optional): Number of split scales for depth-wise convolution. Defaults to 1.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4,
                 use_pos_emb=True, num_heads=8, qkv_bias=True, attn_drop=0., drop=0., scales=1):
        super().__init__()
        width = max(int(math.ceil(dim / scales)), int(math.floor(dim // scales)))
        self.width = width
        if scales == 1:
            self.nums = 1
        else:
            self.nums = scales - 1
        convs = []
        for _ in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width))
        self.convs = nn.ModuleList(convs)

        self.pos_embd = None
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=dim)
        self.norm_xca = nn.BatchNorm2d(dim)
        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.Hardswish()  # TODO: MobileViT is using 'swish'
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """Forward pass for SDTAEncoderBNHS.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        x_in = x

        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        x = torch.cat((out, spx[self.nums]), 1)
        # XCA
        x = self.norm_xca(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        if self.pos_embd:
            pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding
        x = x + self.drop_path(self.gamma_xca * self.xca(x))
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Inverted Bottleneck
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = x_in + self.drop_path(x)

        return x


class XCA(nn.Module):
    """Cross-Covariance Attention (XCA) module.

    XCA is an efficient attention mechanism that computes attention weights based on
    the cross-covariance between queries and keys. It uses learnable temperature
    parameters and normalizes the query and key vectors before computing attention.

    This attention mechanism is particularly efficient for vision transformers as it
    reduces the computational complexity compared to standard self-attention while
    maintaining the ability to capture global dependencies.

    Args:
        dim (int): Input dimension.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        qkv_bias (bool, optional): Whether to add bias to QKV linear projections.
            Defaults to False.
        attn_drop (float, optional): Attention dropout rate. Defaults to 0.
        proj_drop (float, optional): Output projection dropout rate. Defaults to 0.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """Forward pass for XCA.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C)

        Returns:
            torch.Tensor: Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        """Return the names of parameters that do not require weight decay.

        Returns:
            set: Set of parameter names to exclude from weight decay
        """
        return {'temperature'}


class ConvEncoder(nn.Module):
    """Convolutional Encoder block with inverted bottleneck structure.

    This encoder block consists of a depth-wise convolution followed by an inverted
    bottleneck with layer normalization and GELU activation. It includes optional
    layer scaling and stochastic depth for improved training.

    The architecture follows the design principles of modern convolutional networks
    with depth-wise convolutions for efficient local feature extraction and
    inverted bottlenecks for channel mixing.

    Args:
        dim (int): Input/output channel dimension.
        drop_path (float, optional): Drop path rate for stochastic depth. Defaults to 0.
        layer_scale_init_value (float, optional): Initial value for layer scale parameters.
            Set to 0 to disable layer scaling. Defaults to 1e-6.
        expan_ratio (int, optional): Expansion ratio for the inverted bottleneck. Defaults to 4.
        kernel_size (int, optional): Kernel size for the depth-wise convolution. Defaults to 7.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """Forward pass for ConvEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        x_in = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = x_in + self.drop_path(x)
        return x


class ConvEncoderBNHS(nn.Module):
    """Convolutional Encoder with Batch Normalization and Hard-Swish activation.

    This is a variant of ConvEncoder that uses Batch Normalization instead of Layer
    Normalization and Hard-Swish activation instead of GELU. This combination is
    optimized for mobile deployment with improved inference efficiency.

    The key differences from ConvEncoder are:
    - Uses BatchNorm2d instead of LayerNorm for normalization
    - Uses Hard-Swish activation instead of GELU
    - Optimized for mobile hardware acceleration

    Args:
        dim (int): Input/output channel dimension.
        drop_path (float, optional): Drop path rate for stochastic depth. Defaults to 0.
        layer_scale_init_value (float, optional): Initial value for layer scale parameters.
            Set to 0 to disable layer scaling. Defaults to 1e-6.
        expan_ratio (int, optional): Expansion ratio for the inverted bottleneck. Defaults to 4.
        kernel_size (int, optional): Kernel size for the depth-wise convolution. Defaults to 7.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=4, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.Hardswish()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """Forward pass for ConvEncoderBNHS.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        x_in = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = x_in + self.drop_path(x)
        return x
