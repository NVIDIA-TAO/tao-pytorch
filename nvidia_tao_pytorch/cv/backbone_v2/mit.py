# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mix Transformer (MiT) backbone module.

This module provides Mix Transformer (MiT) implementations for the TAO PyTorch framework.
MiT is a hierarchically structured Transformer encoder that outputs multiscale features
without requiring positional encoding, avoiding interpolation issues when testing resolution
differs from training.

The MiT architecture was introduced in "SegFormer: Simple and Efficient Design for
Semantic Segmentation with Transformers" and is designed for semantic segmentation tasks.
It uses a hierarchical structure with overlapping patch embeddings and efficient attention
mechanisms.

Key Features:
- Hierarchical architecture with multiple stages
- Overlapping patch embeddings for better feature extraction
- No positional encoding required
- Support for multiple model sizes (B0-B5)
- Integration with TAO backbone framework
- Support for activation checkpointing and layer freezing
- Efficient design for semantic segmentation

Classes:
    OverlapPatchEmbed: Overlapping patch embedding layer
    DWConv: Depthwise convolution layer
    Mlp: MLP layer with depthwise convolution
    Attention: Attention layer with spatial reduction
    Block: MiT transformer block
    MixTransformer: Main MiT model

Functions:
    mit_b0: MiT B0 model
    mit_b1: MiT B1 model
    mit_b2: MiT B2 model
    mit_b3: MiT B3 model
    mit_b4: MiT B4 model
    mit_b5: MiT B5 model

Example:
    >>> from nvidia_tao_pytorch.cv.backbone_v2 import mit_b0
    >>> model = mit_b0(num_classes=1000)
    >>> x = torch.randn(1, 3, 224, 224)
    >>> output = model(x)

References:
    - [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](
      https://arxiv.org/abs/2105.15203)
    - [https://github.com/NVlabs/SegFormer](https://github.com/NVlabs/SegFormer)
"""

import math
from functools import partial
from typing import List

import torch
import torch.nn as nn
from timm.layers import DropPath, to_2tuple, trunc_normal_

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase


class OverlapPatchEmbed(nn.Module):
    """Image to patch embedding layer."""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, export=False):
        """Initialize the layer."""
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.export = export

        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        """Forward."""
        x = self.proj(x)
        _, C, H, W = x.shape
        if self.export:
            x = x.reshape(-1, C, H * W).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class DWConv(nn.Module):
    """Depthwise convolution layer."""

    def __init__(self, dim=768, export=False):
        """Initialize the depthwise convolution layer."""
        super().__init__()
        self.export = export
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int):
        """Forward."""
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        if self.export:
            x = x.view(-1, C, H * W).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    """MLP layer."""

    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, export=False
    ):
        """Initialize the MLP layer."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features, export)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int):
        """Forward."""
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Attention layer."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        """Initialize the attention layer."""
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int):
        """Forward."""
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """MiT block."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        export=False,
    ):
        """Initialize the MiT block."""
        super().__init__()
        self.export = export

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, export=export)

    def forward(self, x: torch.Tensor, H: int, W: int):
        """Forward."""
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class MixTransformer(BackboneBase):
    """Mix Transformer (MiT) model.

    MiT is a hierarchically structured Transformer encoder which outputs multiscale features. It does not need
    positional encoding, thereby avoiding the interpolation of positional codes which leads to decreased performance
    when the testing resolution differs from training.

    Reference:
    - [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](
      https://arxiv.org/abs/2105.15203)
    - [https://github.com/NVlabs/SegFormer](https://github.com/NVlabs/SegFormer)
    """

    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        depths=[3, 4, 6, 3],
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        sr_ratios=[8, 4, 2, 1],
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        export=False,
        activation_checkpoint=False,
        freeze_at=None,
        freeze_norm=False,
        **kwargs,
    ):
        """Initialize the Mix Transformer (MiT) model.

        Args:
            img_size (int): Input image size. Default: `224`.
            in_chans (int): Number of input image channels. Default: `3`.
            num_classes (int): Number of classes for classification head. Default: `1000`.
            depths (tuple(int)): Number of blocks at each stage. Default: `[3, 4, 6, 3]`.
            embed_dims (tuple(int)): Feature dimension at each stage. Default: `[64, 128, 320, 512]`.
            num_heads: number of heads in each stage. Default: `[1, 2, 5, 8]`.
            mlp_ratio: MLP ratio in each stage. Default: `[4, 4, 4, 4]`.
            sr_ratios: SR ratio in each stage. Default: `[8, 4, 2, 1]`.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate (float): Stochastic depth rate. Default: `0`.
            norm_layer: normalization layer.
            export (bool): Whether to export the model. Default: `False`.
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
        self.depths = depths
        self.export = export

        self.num_features = embed_dims[3]

        # Initialize the patch embedding layers.
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size, patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0], export=self.export
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
            export=self.export,
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
            export=self.export,
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
            export=self.export,
        )

        # Initialize the encoder.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                    export=self.export,
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                    export=self.export,
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2],
                    export=self.export,
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3],
                    export=self.export,
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = norm_layer(embed_dims[3])

        # Initialize the classification head.
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Init Weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        """Set the gradient (activation) checkpointing for the model."""
        if enable:
            raise NotImplementedError("Activation checkpointing is not implemented for MiT.")
        self.activation_checkpoint = enable

    def get_stage_dict(self):
        """Get the stage dictionary."""
        # TODO(@yuw, @hongyuc): No stem. Add patch_embed* as stage 0?
        stage_dict = {
            1: self.block1,
            2: self.block2,
            3: self.block3,
            4: self.block4,
        }
        return stage_dict

    @torch.jit.ignore
    def no_weight_decay(self):
        """Get the set of parameter names to exclude from weight decay."""
        return {"pos_embed1", "pos_embed2", "pos_embed3", "pos_embed4", "cls_token"}  # has pos_embed may be better

    @torch.jit.ignore
    def get_classifier(self):
        """Get the classifier module."""
        return self.head

    def reset_classifier(self, num_classes):
        """Reset the classifier head."""
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward Features."""
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for _, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for _, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for _, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for _, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return outs

    def forward_pre_logits(self, x: torch.Tensor):
        """Forward pass through the backbone, excluding the head."""
        feats = self.forward_features(x)
        feat = feats[-1]
        return feat.mean([-2, -1])  # Global average pooling, (N, C, H, W) -> (N, C)

    def forward_feature_pyramid(self, x: torch.Tensor):
        """Forward pass through the backbone to extract intermediate feature maps."""
        return self.forward_features(x)

    def forward(self, x: torch.Tensor):
        """Forward."""
        x = self.forward_pre_logits(x)
        x = self.head(x)
        return x


@BACKBONE_REGISTRY.register()
def mit_b0(**kwargs):
    """Create a MiT B0 model.

    This function creates a MiT B0 model with the following specifications:
    - Depths: [2, 2, 2, 2] (number of blocks in each stage)
    - Embed dimensions: [32, 64, 160, 256] (feature dimensions)
    - Number of heads: [1, 2, 5, 8] (attention heads in each stage)
    - MLP ratios: [4, 4, 4, 4] (MLP expansion ratios)
    - SR ratios: [8, 4, 2, 1] (spatial reduction ratios)

    Args:
        **kwargs: Additional arguments passed to MixTransformer constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - img_size (int): Input image size. Default: `224`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        MixTransformer: Configured MiT B0 model.

    Example:
        >>> model = mit_b0(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This is the smallest MiT model variant with approximately 3.7M parameters.
    """
    return MixTransformer(
        depths=[2, 2, 2, 2],
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        sr_ratios=[8, 4, 2, 1],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def mit_b1(**kwargs):
    """Create a MiT B1 model.

    This function creates a MiT B1 model with the following specifications:
    - Depths: [2, 2, 2, 2] (number of blocks in each stage)
    - Embed dimensions: [64, 128, 320, 512] (feature dimensions)
    - Number of heads: [1, 2, 5, 8] (attention heads in each stage)
    - MLP ratios: [4, 4, 4, 4] (MLP expansion ratios)
    - SR ratios: [8, 4, 2, 1] (spatial reduction ratios)

    Args:
        **kwargs: Additional arguments passed to MixTransformer constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - img_size (int): Input image size. Default: `224`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        MixTransformer: Configured MiT B1 model.

    Example:
        >>> model = mit_b1(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 14M parameters and provides a good balance
        between accuracy and computational efficiency.
    """
    return MixTransformer(
        depths=[2, 2, 2, 2],
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        sr_ratios=[8, 4, 2, 1],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def mit_b2(**kwargs):
    """Create a MiT B2 model.

    This function creates a MiT B2 model with the following specifications:
    - Depths: [3, 4, 6, 3] (number of blocks in each stage)
    - Embed dimensions: [64, 128, 320, 512] (feature dimensions)
    - Number of heads: [1, 2, 5, 8] (attention heads in each stage)
    - MLP ratios: [4, 4, 4, 4] (MLP expansion ratios)
    - SR ratios: [8, 4, 2, 1] (spatial reduction ratios)

    Args:
        **kwargs: Additional arguments passed to MixTransformer constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - img_size (int): Input image size. Default: `224`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        MixTransformer: Configured MiT B2 model.

    Example:
        >>> model = mit_b2(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 25M parameters and provides better accuracy
        than B1 with increased computational cost.
    """
    return MixTransformer(
        depths=[3, 4, 6, 3],
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        sr_ratios=[8, 4, 2, 1],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def mit_b3(**kwargs):
    """Create a MiT B3 model.

    This function creates a MiT B3 model with the following specifications:
    - Depths: [3, 4, 18, 3] (number of blocks in each stage)
    - Embed dimensions: [64, 128, 320, 512] (feature dimensions)
    - Number of heads: [1, 2, 5, 8] (attention heads in each stage)
    - MLP ratios: [4, 4, 4, 4] (MLP expansion ratios)
    - SR ratios: [8, 4, 2, 1] (spatial reduction ratios)

    Args:
        **kwargs: Additional arguments passed to MixTransformer constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - img_size (int): Input image size. Default: `224`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        MixTransformer: Configured MiT B3 model.

    Example:
        >>> model = mit_b3(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 45M parameters and provides significantly
        better accuracy than B2 with deeper architecture.
    """
    return MixTransformer(
        depths=[3, 4, 18, 3],
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        sr_ratios=[8, 4, 2, 1],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def mit_b4(**kwargs):
    """Create a MiT B4 model.

    This function creates a MiT B4 model with the following specifications:
    - Depths: [3, 8, 27, 3] (number of blocks in each stage)
    - Embed dimensions: [64, 128, 320, 512] (feature dimensions)
    - Number of heads: [1, 2, 5, 8] (attention heads in each stage)
    - MLP ratios: [4, 4, 4, 4] (MLP expansion ratios)
    - SR ratios: [8, 4, 2, 1] (spatial reduction ratios)

    Args:
        **kwargs: Additional arguments passed to MixTransformer constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - img_size (int): Input image size. Default: `224`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        MixTransformer: Configured MiT B4 model.

    Example:
        >>> model = mit_b4(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This model has approximately 62M parameters and provides excellent
        accuracy for semantic segmentation tasks.
    """
    return MixTransformer(
        depths=[3, 8, 27, 3],
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        sr_ratios=[8, 4, 2, 1],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def mit_b5(**kwargs):
    """Create a MiT B5 model.

    This function creates a MiT B5 model with the following specifications:
    - Depths: [3, 6, 40, 3] (number of blocks in each stage)
    - Embed dimensions: [64, 128, 320, 512] (feature dimensions)
    - Number of heads: [1, 2, 5, 8] (attention heads in each stage)
    - MLP ratios: [4, 4, 4, 4] (MLP expansion ratios)
    - SR ratios: [8, 4, 2, 1] (spatial reduction ratios)

    Args:
        **kwargs: Additional arguments passed to MixTransformer constructor.
            Common arguments include:
            - num_classes (int): Number of output classes. Default: `1000`
            - img_size (int): Input image size. Default: `224`
            - in_chans (int): Number of input channels. Default: `3`
            - activation_checkpoint (bool): Enable activation checkpointing. Default: `False`
            - freeze_at (list): Layers to freeze. Default: `None`
            - freeze_norm (bool): Freeze normalization layers. Default: `False`

    Returns:
        MixTransformer: Configured MiT B5 model.

    Example:
        >>> model = mit_b5(num_classes=1000)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)  # Shape: (1, 1000)

    Note:
        This is the largest MiT model variant with approximately 82M parameters.
        It provides the best accuracy but requires the most computational resources.
    """
    return MixTransformer(
        depths=[3, 6, 40, 3],
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        sr_ratios=[8, 4, 2, 1],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=0.0,
        drop_path_rate=0.1,
        **kwargs,
    )
