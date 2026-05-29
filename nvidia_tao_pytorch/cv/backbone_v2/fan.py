# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FAN (Fully Attentional Network) backbone module.

This module provides FAN implementations for the TAO PyTorch framework.
FAN is a vision transformer architecture that uses fully attentional networks
for image classification and other vision tasks.

The FAN architecture introduces a novel attention mechanism that captures both
local and global information efficiently. It uses class attention for global
context and token mixing for local feature interaction.

Key Features:
- Fully attentional network design
- Class attention for global context modeling
- Token mixing for local feature interaction
- Support for multiple model sizes (Tiny, Small, Base, Large, XLarge)
- Hybrid architectures with Swin backbones
- Integration with TAO backbone framework
- Support for activation checkpointing and layer freezing
- Efficient design with good accuracy/speed balance

Classes:
    ClassAttn: Class attention mechanism
    PositionalEncodingFourier: Fourier positional encoding
    SqueezeExcite: Squeeze and excitation block
    SEMlp: Squeeze-excitation MLP
    Mlp: Standard MLP block
    ConvPatchEmbed: Convolutional patch embedding
    DWConv: Depthwise convolution
    ClassAttentionBlock: Class attention block
    TokenMixing: Token mixing mechanism
    HybridEmbed: Hybrid embedding layer
    ChannelProcessing: Channel processing block
    FANBlock_SE: FAN block with squeeze-excitation
    FANBlock: Standard FAN block
    OverlapPatchEmbed: Overlapping patch embedding
    FAN: Main FAN model

Functions:
    fan_tiny_12_p16_224: FAN Tiny model
    fan_small_12_p16_224_se_attn: FAN Small with SE attention
    fan_small_12_p16_224: FAN Small model
    fan_base_18_p16_224: FAN Base model
    fan_large_24_p16_224: FAN Large model
    fan_tiny_8_p4_hybrid: FAN Tiny hybrid model
    fan_small_12_p4_hybrid: FAN Small hybrid model
    fan_base_16_p4_hybrid: FAN Base hybrid model
    fan_large_16_p4_hybrid: FAN Large hybrid model
    fan_xlarge_16_p4_hybrid: FAN XLarge hybrid model
    fan_swin_tiny_patch4_window7_224: FAN with Swin Tiny backbone
    fan_swin_small_patch4_window7_224: FAN with Swin Small backbone
    fan_swin_base_patch4_window7_224: FAN with Swin Base backbone
    fan_swin_large_patch4_window7_224: FAN with Swin Large backbone

Example:
    >>> from nvidia_tao_pytorch.cv.backbone_v2 import fan_tiny_12_p16_224
    >>> model = fan_tiny_12_p16_224(num_classes=1000)
    >>> x = torch.randn(1, 3, 224, 224)
    >>> output = model(x)

References:
    - [FAN: Focused Attention Networks](https://arxiv.org/abs/2204.12451)
    - [https://github.com/NVlabs/FAN](https://github.com/NVlabs/FAN)
"""

import math
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import Mlp as MlpOri

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase
from nvidia_tao_pytorch.cv.backbone_v2.convnext_utils import ConvNeXtFANBackbone
from nvidia_tao_pytorch.cv.backbone_v2.swin import SwinTransformer


class ClassAttn(nn.Module):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        """Initialize ClassAttn class"""
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fast_attn = hasattr(torch._C._nn, "_scaled_dot_product_attention")  # pylint:disable=I1101

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """Taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        with slight modifications to do CA
        """
        B, N, _ = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if torch.onnx.is_in_onnx_export() or not self.fast_attn:
            q = q * self.scale

            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, self.dim)
        else:
            # Since Torch 1.14, scaled_dot_product_attention has been optimized for performance
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p,
            )
            x_cls = x.transpose(1, 2).reshape(B, 1, self.dim)

        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


class PositionalEncodingFourier(nn.Module):
    """Positional encoding relying on a fourier kernel matching."""

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        """Initialize PositionalEncodingFourier class"""
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.eps = 1e-6

    def forward(self, B: int, H: int, W: int, fp32=True):
        """Forward function"""
        device = self.token_projection.weight.device
        y_embed = (
            torch.arange(1, H + 1, dtype=torch.float32 if fp32 else torch.float16, device=device)
            .unsqueeze(1)
            .repeat(1, 1, W)
        )
        x_embed = torch.arange(1, W + 1, dtype=torch.float32 if fp32 else torch.float16, device=device).repeat(1, H, 1)
        y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32 if fp32 else torch.float16, device=device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.hidden_dim)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos.repeat(B, 1, 1, 1)  # (B, C, H, W)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution + batch norm"""
    return torch.nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
    )


def sigmoid(x, inplace=False):
    """Sigmoid function"""
    return x.sigmoid_() if inplace else x.sigmoid()


def make_divisible(v, divisor=8, min_value=None):
    """Make tensor divisible"""
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    """SqueezeExcite"""

    def __init__(
        self, in_chs, se_ratio=0.25, reduced_base_chs=None, act_layer=nn.ReLU, gate_fn=sigmoid, divisor=1, **_
    ):
        """Initialize SqueezeExcite class"""
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        """Forward function"""
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class SEMlp(nn.Module):
    """SE Mlp Model Module"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        linear=False,
        use_se=True,
    ):
        """Init Module"""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.gamma = nn.Parameter(torch.ones(hidden_features), requires_grad=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.se = SqueezeExcite(out_features, se_ratio=0.25) if use_se else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize Weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

    def forward(self, x, H, W):
        """Forward Function"""
        B, N, C = x.shape
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.drop(self.gamma * self.dwconv(x, H, W)) + x
        x = self.fc2(x)
        x = self.drop(x)
        x = self.se(x.permute(0, 2, 1).reshape(B, C, H, W)).reshape(B, C, N).permute(0, 2, 1)
        return x, H, W


class Mlp(nn.Module):
    """MLP Module"""

    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, linear=False
    ):
        """Init Function"""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.gamma = nn.Parameter(torch.ones(hidden_features), requires_grad=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Init Weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

    def forward(self, x, H, W):
        """Forward Function"""
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.drop(self.gamma * self.dwconv(x, H, W)) + x
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvPatchEmbed(nn.Module):
    """Image to Patch Embedding using multiple convolutional layers"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, act_layer=nn.GELU):
        """Initialize ConvPatchEmbed class"""
        super().__init__()
        img_size = to_2tuple(img_size)
        num_patches = (img_size[1] // patch_size) * (img_size[0] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if patch_size == 16:
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim // 8, 2),
                act_layer(),
                conv3x3(embed_dim // 8, embed_dim // 4, 2),
                act_layer(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                act_layer(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size == 8:
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim // 4, 2),
                act_layer(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                act_layer(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size == 4:
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim // 4, 2),
                act_layer(),
                conv3x3(embed_dim // 4, embed_dim // 1, 2),
            )
        else:
            raise ("For convolutional projection, patch size has to be in [8, 16]")

    def forward(self, x):
        """Forward function"""
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x, (Hp, Wp)


class DWConv(nn.Module):
    """Depth-wise convolution"""

    def __init__(self, in_features, out_features=None, act_layer=nn.GELU, kernel_size=3):
        """Initialize DWConv class"""
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(
            in_features, in_features, kernel_size=kernel_size, padding=padding, groups=in_features
        )
        self.act = act_layer()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv2 = torch.nn.Conv2d(
            in_features, out_features, kernel_size=kernel_size, padding=padding, groups=out_features
        )

    def forward(self, x, H: int, W: int):
        """Forward function"""
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)
        return x


class ClassAttentionBlock(nn.Module):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        eta=1.0,
        tokens_norm=False,
    ):
        """Initialize ClassAttentionBlock class"""
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = ClassAttn(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MlpOri(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if eta is not None:  # LayerScale Initialization (no layerscale when None)
            self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

        # See https://github.com/rwightman/pytorch-image-models/pull/747#issuecomment-877795721
        self.tokens_norm = tokens_norm

    def forward(self, x, return_attention=False):
        """Forward function"""
        x_norm1 = self.norm1(x)
        if return_attention:
            x1, attn = self.attn(x_norm1, use_attn=return_attention)
        else:
            x1 = self.attn(x_norm1)
        x_attn = torch.cat([x1, x_norm1[:, 1:]], dim=1)
        x = x + self.drop_path(self.gamma1 * x_attn)
        if self.tokens_norm:
            x = self.norm2(x)
        else:
            x = torch.cat([self.norm2(x[:, 0:1]), x[:, 1:]], dim=1)
        x_res = x
        cls_token = x[:, 0:1]
        cls_token = self.gamma2 * self.mlp(cls_token)
        x = torch.cat([cls_token, x[:, 1:]], dim=1)
        x = x_res + self.drop_path(x)
        if return_attention:
            return attn
        return x


class TokenMixing(nn.Module):
    """Token Mixing"""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        linear=False,
        share_atten=False,
        drop_path=0.0,
        emlp=False,
        sharpen_attn=False,
        mlp_hidden_dim=None,
        act_layer=nn.GELU,
        drop=None,
        norm_layer=nn.LayerNorm,
    ):
        """Initialize TokenMixing class"""
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.share_atten = share_atten
        self.emlp = emlp

        cha_sr = 1
        self.q = nn.Linear(dim, dim // cha_sr, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2 // cha_sr, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.linear = linear
        self.sr_ratio = sr_ratio
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

    def forward(self, x, H, W, atten=None, return_attention=False):
        """Forward function"""
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = q * self.scale @ k.transpose(-2, -1)  # * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn @ v


class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, patch_size=2, feature_size=None, in_chans=3, embed_dim=384):
        """Init Function"""
        super().__init__()
        assert isinstance(backbone, nn.Module), "backbone must be a nn.Module"
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone.forward_features(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, "feature_info"):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, return_feat=False):
        """Forward Function"""
        out_list = None
        if return_feat:
            x, out_list = self.backbone.forward_features(x, return_feat=return_feat)
        else:
            x = self.backbone.forward_features(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        _, _, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        if return_feat:
            return x, (H // self.patch_size[0], W // self.patch_size[1]), out_list
        return x, (H // self.patch_size[0], W // self.patch_size[1])


class ChannelProcessing(nn.Module):
    """Channel Processing in FAN Module"""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        linear=False,
        drop_path=0.0,
        mlp_hidden_dim=None,
        act_layer=nn.GELU,
        drop=None,
        norm_layer=nn.LayerNorm,
        cha_sr_ratio=1,
        c_head_num=None,
    ):
        """Init Function"""
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        num_heads = c_head_num or num_heads
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.cha_sr_ratio = cha_sr_ratio if num_heads > 1 else 1

        # config of mlp for v processing
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp_v = Mlp(
            in_features=dim // self.cha_sr_ratio,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            linear=linear,
        )
        self.norm_v = norm_layer(dim // self.cha_sr_ratio)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Init Weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

    def _gen_attn(self, q, k):
        """Function to Get Attention"""
        _, _, N, _ = k.shape
        if torch.onnx.is_in_onnx_export():
            # If softmax dim is not the last dimension, then PyTorch decompose the softmax ops into
            # smaller ops like ReduceMax, ReduceSum, Sub, and Div.
            # As a result, ONNX export fails for opset_version >= 12.
            # Here, we rearrange the transpose so that softmax is done over the last dimension.
            q = q.transpose(-1, -2).softmax(-1)
            k = k.transpose(-1, -2).softmax(-1)
            warnings.warn("Replacing default adatpive_avg_pool2d to custom implementation for ONNX export")
            # adaptive_avg_pool2d is not supported for torch to onnx export
            k = adaptive_avg_pool(k.transpose(-1, -2), (N, 1))
        else:
            q = q.softmax(-2).transpose(-1, -2)
            k = torch.nn.functional.adaptive_avg_pool2d(k.softmax(-2), (N, 1))
        attn = torch.nn.functional.sigmoid(q @ k)
        return attn * self.temperature

    def forward(self, x, H, W, atten=None):
        """Forward Function"""
        B, N, C = x.shape

        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = self._gen_attn(q, k)
        attn = self.attn_drop(attn)

        Bv, Hd, Nv, Cv = v.shape
        v = (
            self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd * Cv), H, W))
            .reshape(Bv, Nv, Hd, Cv)
            .transpose(1, 2)
        )

        repeat_time = N // attn.shape[-1]
        attn = attn.repeat_interleave(repeat_time, dim=-1) if attn.shape[-1] > 1 else attn
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
        return x, (attn * v.transpose(-1, -2)).transpose(-1, -2)  # attn

    @torch.jit.ignore
    def no_weight_decay(self):
        """Ignore Weight Decay"""
        return {"temperature"}


def adaptive_avg_pool(x, size):
    """Function to convert to ONNX exportable F.adaptive_avg_pool2d

    Args:
        x: Input tensor
        size: Output Dimension. Can be either an integer or a tuple
    """
    inp_size = x.size()
    kernel_width, kernel_height = inp_size[2], inp_size[3]
    if size is not None:
        if isinstance(size, int):
            kernel_width = math.ceil(inp_size[2] / size)
            kernel_height = math.ceil(inp_size[3] / size)
        elif isinstance(size, (list, tuple)):
            assert len(size) == 2
            kernel_width = math.ceil(inp_size[2] / size[0])
            kernel_height = math.ceil(inp_size[3] / size[1])
    if torch.is_tensor(kernel_width):
        kernel_width = kernel_width.item()
        kernel_height = kernel_height.item()
    return F.avg_pool2d(input=x, ceil_mode=False, kernel_size=(kernel_width, kernel_height))


class FANBlock_SE(nn.Module):
    """FAN Block SE"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        sharpen_attn=False,
        use_se=False,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        eta=1.0,
        sr_ratio=1.0,
        qk_scale=None,
        linear=False,
        downsample=None,
        c_head_num=None,
    ):
        """Init Module"""
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TokenMixing(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            mlp_hidden_dim=int(dim * mlp_ratio),
            sharpen_attn=sharpen_attn,
            attn_drop=attn_drop,
            proj_drop=drop,
            drop=drop,
            drop_path=drop_path,
            sr_ratio=sr_ratio,
            linear=linear,
            emlp=False,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = SEMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)

    def forward(self, x, attn=None):
        """Forward Function"""
        H, W = self.H, self.W
        x_new, _ = self.attn(self.norm1(x), H, W)
        x = x + self.drop_path(self.gamma1 * x_new)
        x_new, H, W = self.mlp(self.norm2(x), H, W)
        x = x + self.drop_path(self.gamma2 * x_new)
        return x


class FANBlock(nn.Module):
    """FAN block from https://arxiv.org/abs/2204.12451"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        sharpen_attn=False,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        eta=1.0,
        sr_ratio=1.0,
        downsample=None,
        c_head_num=None,
    ):
        """Initialize FANBlock class"""
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TokenMixing(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            mlp_hidden_dim=int(dim * mlp_ratio),
            sharpen_attn=sharpen_attn,
            attn_drop=attn_drop,
            proj_drop=drop,
            drop=drop,
            drop_path=drop_path,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = ChannelProcessing(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            drop_path=drop_path,
            drop=drop,
            mlp_hidden_dim=int(dim * mlp_ratio),
            c_head_num=c_head_num,
        )

        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)

        self.downsample = downsample
        self.H = None
        self.W = None

    def forward(self, x, attn=None, return_attention=False):
        """Forward function"""
        H, W = self.H, self.W

        x_new, attn_s = self.attn(self.norm1(x), H, W)
        x = x + self.drop_path(self.gamma1 * x_new)

        x_new, _ = self.mlp(self.norm2(x), H, W, atten=attn)
        x = x + self.drop_path(self.gamma2 * x_new)
        if return_attention:
            return x, attn_s

        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        self.H, self.W = H, W
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        """Overlap PatchEmbed"""
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
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

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

    def forward(self, x, H, W):
        """Forward functions"""
        B, _, C = x.shape
        x = x.transpose(-1, -2).reshape(B, C, H, W)
        x = self.proj(x)
        _, _, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class FAN(BackboneBase):
    """Fully attentional network (FAN) model.

    FAN is a vision transformer architecture that enhances self-attention mechanisms to improve robustness and
    mid-level feature representations. It is highly robust to unseen natural corruptions in various visual recognition
    tasks.

    References:
    - [Understanding The Robustness in Vision Transformers](https://arxiv.org/abs/2204.12451)
    - [https://github.com/NVlabs/FAN](https://github.com/NVlabs/FAN)
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        sharpen_attn=False,
        channel_dims=None,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        sr_ratio=None,
        backbone=None,
        act_layer=None,
        norm_layer=None,
        se_mlp=False,
        cls_attn_layers=2,
        use_pos_embed=True,
        eta=1.0,
        tokens_norm=False,
        c_head_num=None,
        hybrid_patch_size=2,
        head_init_scale=1.0,
        activation_checkpoint=False,
        freeze_at=None,
        freeze_norm=False,
        export=False,
    ):
        """Initialize the FAN model."""
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            export=export,
        )

        img_size = to_2tuple(img_size)
        assert (img_size[0] % patch_size == 0) and (img_size[0] % patch_size == 0), (
            "`patch_size` should divide image dimensions evenly"
        )

        num_heads = [num_heads] * depth if not isinstance(num_heads, list) else num_heads

        channel_dims = [embed_dim] * depth if channel_dims is None else channel_dims
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        if backbone is None:
            self.patch_embed = ConvPatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, act_layer=act_layer
            )
        else:
            self.patch_embed = HybridEmbed(backbone=backbone, patch_size=hybrid_patch_size, embed_dim=embed_dim)
        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.pos_embed = PositionalEncodingFourier(dim=embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        if se_mlp:
            build_block = FANBlock_SE
        else:
            build_block = FANBlock
        self.blocks = nn.ModuleList([])
        for i in range(depth):
            if i < depth - 1 and channel_dims[i] != channel_dims[i + 1]:
                downsample = OverlapPatchEmbed(
                    img_size=img_size, patch_size=3, stride=2, in_chans=channel_dims[i], embed_dim=channel_dims[i + 1]
                )
            else:
                downsample = None
            self.blocks.append(
                build_block(
                    dim=channel_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    sr_ratio=sr_ratio[i],
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    eta=eta,
                    downsample=downsample,
                    c_head_num=c_head_num[i] if c_head_num is not None else None,
                )
            )
        self.num_features = self.embed_dim = channel_dims[i]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, channel_dims[i]))
        self.cls_attn_blocks = nn.ModuleList(
            [
                ClassAttentionBlock(
                    dim=channel_dims[-1],
                    num_heads=num_heads[-1],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    eta=eta,
                    tokens_norm=tokens_norm,
                )
                for _ in range(cls_attn_layers)
            ]
        )

        # Classifier head
        self.norm = norm_layer(channel_dims[i])
        if num_classes > 0:
            self.head = nn.Linear(self.num_features, num_classes)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
        else:
            self.head = nn.Identity()

        # Init weights
        trunc_normal_(self.cls_token, std=0.02)
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

    def get_stage_dict(self):
        """Get the stage dictionary."""
        stage_dict = {}
        # TODO(@yuw, @hongyuc): No stem. Add patch_embed as stage 0?
        for i, block in enumerate(self.blocks, start=1):
            stage_dict[i] = block
        return stage_dict

    @torch.jit.ignore
    def no_weight_decay(self):
        """Get the set of parameter names to exclude from weight decay."""
        return {"pos_embed", "cls_token"}

    @torch.jit.ignore
    def get_classifier(self):
        """Get the classifier module."""
        return self.head

    def reset_classifier(self, num_classes, **kwargs):
        """Reset the classifier head."""
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """Forward pass through the backbone, excluding the head."""
        B = x.shape[0]
        x, (Hp, Wp) = self.patch_embed(x)

        if self.use_pos_embed:
            pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        x = self.pos_drop(x)
        H, W = Hp, Wp
        for blk in self.blocks:
            blk.H, blk.W = H, W
            # Disable activation checkpointing during ONNX export
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = blk(x)
            else:
                x = checkpoint.checkpoint(blk, x, use_reentrant=True)
            H, W = blk.H, blk.W

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.cls_attn_blocks:
            x = blk(x)
        x = self.norm(x)[:, 0]
        return x

    def forward_pre_logits(self, x):
        """Forward pass through the backbone, excluding the head."""
        return self.forward_features(x)

    def forward_feature_pyramid(self, x):
        """Forward pass through the backbone to extract intermediate feature maps."""
        outs = []
        B = x.shape[0]
        x, (Hp, Wp) = self.patch_embed(x)

        if self.use_pos_embed:
            pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        x = self.pos_drop(x)
        H, W = Hp, Wp
        for blk in self.blocks:
            blk.H, blk.W = H, W
            # Disable activation checkpointing during ONNX export
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = blk(x)
            else:
                x = checkpoint.checkpoint(blk, x, use_reentrant=True)
            H, W = blk.H, blk.W
            # [B, L, C] -> [B, C, H, W]
            x_spatial = x.permute(0, 2, 1).view(B, -1, H, W)
            outs.append(x_spatial)
        return outs

    def forward(self, x):
        """Forward."""
        x = self.forward_features(x)
        x = self.head(x)
        return x


# FAN-ViT Models
@BACKBONE_REGISTRY.register()
def fan_tiny_12_p16_224(**kwargs):
    """FAN-ViT Tiny"""
    depth = 12
    sr_ratio = [1] * (depth // 2) + [1] * (depth // 2)
    model_kwargs = dict(
        patch_size=16,
        embed_dim=192,
        depth=depth,
        num_heads=4,
        sr_ratio=sr_ratio,
        eta=1.0,
        tokens_norm=True,
        sharpen_attn=False,
        **kwargs,
    )
    return FAN(**model_kwargs)


@BACKBONE_REGISTRY.register()
def fan_small_12_p16_224_se_attn(**kwargs):
    """FAN-ViT SE Small"""
    depth = 12
    sr_ratio = [1] * (depth // 2) + [1] * (depth // 2)
    model_kwargs = dict(
        patch_size=16,
        embed_dim=384,
        depth=depth,
        num_heads=8,
        sr_ratio=sr_ratio,
        eta=1.0,
        tokens_norm=True,
        sharpen_attn=False,
        se_mlp=True,
        **kwargs,
    )
    return FAN(**model_kwargs)


@BACKBONE_REGISTRY.register()
def fan_small_12_p16_224(**kwargs):
    """FAN-ViT Small"""
    depth = 12
    sr_ratio = [1] * depth
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=depth, num_heads=8, sr_ratio=sr_ratio, eta=1.0, tokens_norm=True, **kwargs
    )
    return FAN(**model_kwargs)


@BACKBONE_REGISTRY.register()
def fan_base_18_p16_224(**kwargs):
    """FAN-ViT Base"""
    depth = 18
    sr_ratio = [1] * (depth // 2) + [1] * (depth // 2)
    model_kwargs = dict(
        patch_size=16,
        embed_dim=448,
        depth=depth,
        num_heads=8,
        sr_ratio=sr_ratio,
        eta=1.0,
        tokens_norm=True,
        sharpen_attn=False,
        **kwargs,
    )
    return FAN(**model_kwargs)


@BACKBONE_REGISTRY.register()
def fan_large_24_p16_224(**kwargs):
    """FAN-ViT Large"""
    depth = 24
    sr_ratio = [1] * (depth // 2) + [1] * (depth // 2)
    model_kwargs = dict(
        patch_size=16,
        embed_dim=480,
        depth=depth,
        num_heads=10,
        sr_ratio=sr_ratio,
        eta=1.0,
        tokens_norm=True,
        sharpen_attn=False,
        **kwargs,
    )
    return FAN(**model_kwargs)


# FAN-Hybrid Models
# CNN backbones are based on ConvNeXt architecture with only first two stages for downsampling purpose
# This has been verified to be beneficial for downstream tasks


@BACKBONE_REGISTRY.register()
def fan_tiny_8_p4_hybrid(**kwargs):
    """FAN Hybrid Tiny"""
    depth = 8
    sr_ratio = [1] * (depth // 2) + [1] * (depth // 2 + 1)
    backbone = ConvNeXtFANBackbone(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False)
    model_kwargs = dict(
        patch_size=16,
        embed_dim=192,
        depth=depth,
        num_heads=8,
        sr_ratio=sr_ratio,
        backbone=backbone,
        eta=1.0,
        tokens_norm=True,
        sharpen_attn=False,
        **kwargs,
    )
    return FAN(**model_kwargs)


@BACKBONE_REGISTRY.register()
def fan_small_12_p4_hybrid(**kwargs):
    """FAN Hybrid Small"""
    depth = 10
    channel_dims = [384] * 10 + [384] * (depth - 10)
    sr_ratio = [1] * (depth // 2) + [1] * (depth // 2)
    backbone = ConvNeXtFANBackbone(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False)
    model_kwargs = dict(
        patch_size=16,
        embed_dim=384,
        depth=depth,
        channel_dims=channel_dims,
        num_heads=8,
        sr_ratio=sr_ratio,
        backbone=backbone,
        eta=1.0,
        tokens_norm=True,
        sharpen_attn=False,
        **kwargs,
    )
    return FAN(**model_kwargs)


@BACKBONE_REGISTRY.register()
def fan_base_16_p4_hybrid(**kwargs):
    """FAN Hybrid Base"""
    depth = 16
    sr_ratio = [1] * (depth // 2) + [1] * (depth // 2)
    backbone = ConvNeXtFANBackbone(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False)
    model_kwargs = dict(
        patch_size=16,
        embed_dim=448,
        depth=depth,
        num_heads=8,
        sr_ratio=sr_ratio,
        backbone=backbone,
        eta=1.0,
        tokens_norm=True,
        sharpen_attn=False,
        **kwargs,
    )
    return FAN(**model_kwargs)


@BACKBONE_REGISTRY.register()
def fan_large_16_p4_hybrid(**kwargs):
    """FAN Hybrid Large"""
    depth = 22
    sr_ratio = [1] * (depth // 2) + [1] * (depth // 2)
    backbone = ConvNeXtFANBackbone(depths=[3, 5], dims=[128, 256, 512, 1024], use_head=False)
    model_kwargs = dict(
        patch_size=16,
        embed_dim=480,
        depth=depth,
        num_heads=10,
        sr_ratio=sr_ratio,
        backbone=backbone,
        eta=1.0,
        tokens_norm=True,
        sharpen_attn=False,
        head_init_scale=0.001,
        **kwargs,
    )
    return FAN(**model_kwargs)


@BACKBONE_REGISTRY.register()
def fan_xlarge_16_p4_hybrid(**kwargs):
    """FAN Hybrid XLarge"""
    depth = 23
    stage_depth = 20
    channel_dims = [528] * stage_depth + [768] * (depth - stage_depth)
    num_heads = [11] * stage_depth + [16] * (depth - stage_depth)
    sr_ratio = [1] * (depth // 2) + [1] * (depth // 2 + 1)
    backbone = ConvNeXtFANBackbone(depths=[3, 7], dims=[128, 256, 512, 1024], use_head=False)
    model_kwargs = dict(
        patch_size=16,
        embed_dim=channel_dims[0],
        depth=depth,
        channel_dims=channel_dims,
        num_heads=num_heads,
        sr_ratio=sr_ratio,
        backbone=backbone,
        eta=1.0,
        tokens_norm=True,
        sharpen_attn=False,
        **kwargs,
    )
    return FAN(**model_kwargs)


# FAN-Swin Models


@BACKBONE_REGISTRY.register()
def fan_swin_tiny_patch4_window7_224(**kwargs):
    """Swin-T @ 224x224, trained ImageNet-1k"""
    mlp_type = ["FAN", "FAN", "FAN", "Mlp"]
    return SwinTransformer(
        img_size=(224, 224),
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        mlp_type=mlp_type,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def fan_swin_small_patch4_window7_224(**kwargs):
    """Swin-S @ 224x224, trained ImageNet-1k"""
    mlp_type = ["FAN", "FAN", "FAN", "Mlp"]
    return SwinTransformer(
        img_size=(224, 224),
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        mlp_type=mlp_type,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def fan_swin_base_patch4_window7_224(**kwargs):
    """Swin-S @ 224x224, trained ImageNet-1k"""
    mlp_type = ["FAN", "FAN", "FAN", "Mlp"]
    return SwinTransformer(
        img_size=(224, 224),
        patch_size=4,
        window_size=7,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        mlp_type=mlp_type,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def fan_swin_large_patch4_window7_224(**kwargs):
    """Swin-S @ 224x224, trained ImageNet-1k"""
    mlp_type = ["FAN", "FAN", "FAN", "Mlp"]
    return SwinTransformer(
        img_size=(224, 224),
        patch_size=4,
        window_size=7,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        mlp_type=mlp_type,
        **kwargs,
    )
