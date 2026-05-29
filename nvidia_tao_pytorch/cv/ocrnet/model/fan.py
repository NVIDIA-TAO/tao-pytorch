# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" FAN Model Module """

import math
from functools import partial
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, trunc_normal_, to_2tuple
from nvidia_tao_pytorch.cv.backbone.convnext_utils import _create_hybrid_backbone
from nvidia_tao_pytorch.cv.backbone.fan import (TokenMixing, SqueezeExcite, OverlapPatchEmbed,
                                                PositionalEncodingFourier, ConvPatchEmbed,
                                                DWConv, adaptive_avg_pool)
import warnings


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj.0.0', 'classifier': 'head',
        **kwargs
    }


class SEMlp(nn.Module):
    """ SE Mlp Model Module """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False, use_se=True):
        """ Init Module """
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
        """ Initialize Weights """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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
        """ Forward Function """
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
    """ MLP Module """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
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
        """ Init Weights """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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
        """ Forward Function """
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.drop(self.gamma * self.dwconv(x, H, W)) + x
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, patch_size=2, feature_size=None, in_chans=3, embed_dim=384):
        """ Init Function """
        super().__init__()
        assert isinstance(backbone, nn.Module)
        if not isinstance(img_size, tuple):
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
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """ Forward Function """
        x = self.backbone.forward_features(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        _, _, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x, (H // self.patch_size[0], W // self.patch_size[1])


class ChannelProcessing(nn.Module):
    """ Channel Processing in FAN Module """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., linear=False, drop_path=0.,
                 mlp_hidden_dim=None, act_layer=nn.GELU, drop=None, norm_layer=nn.LayerNorm, cha_sr_ratio=1, c_head_num=None):
        """ Init Function """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        num_heads = c_head_num or num_heads
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.cha_sr_ratio = cha_sr_ratio if num_heads > 1 else 1

        # config of mlp for v processing
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_v = Mlp(in_features=dim // self.cha_sr_ratio, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)
        self.norm_v = norm_layer(dim // self.cha_sr_ratio)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """ Init Weights """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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
        """ Function to Get Attention """
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
        """ Forward Function """
        B, N, C = x.shape

        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B, N, self.num_heads,  C // self.num_heads).permute(0, 2, 1, 3)

        attn = self._gen_attn(q, k)
        attn = self.attn_drop(attn)

        Bv, Hd, Nv, Cv = v.shape
        v = self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd * Cv), H, W)).reshape(Bv, Nv, Hd, Cv).transpose(1, 2)

        repeat_time = N // attn.shape[-1]
        attn = attn.repeat_interleave(repeat_time, dim=-1) if attn.shape[-1] > 1 else attn
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
        return x,  (attn * v.transpose(-1, -2)).transpose(-1, -2)  # attn

    @torch.jit.ignore
    def no_weight_decay(self):
        """ Ignore Weight Decay """
        return {'temperature'}


class FANBlock_SE(nn.Module):
    """ FAN Block SE """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., sharpen_attn=False, use_se=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, eta=1., sr_ratio=1., qk_scale=None, linear=False, downsample=None, c_head_num=None):
        """ Init Module """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TokenMixing(dim, num_heads=num_heads, qkv_bias=qkv_bias, mlp_hidden_dim=int(dim * mlp_ratio), sharpen_attn=sharpen_attn,
                                attn_drop=attn_drop, proj_drop=drop, drop=drop, drop_path=drop_path, sr_ratio=sr_ratio, linear=linear, emlp=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = SEMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)

    def forward(self, x, H: int, W: int, attn=None):
        """ Forward Function """
        x_new, _ = self.attn(self.norm1(x), H, W)
        x = x + self.drop_path(self.gamma1 * x_new)
        x_new, H, W = self.mlp(self.norm2(x), H, W)
        x = x + self.drop_path(self.gamma2 * x_new)
        return x, H, W


class FANBlock(nn.Module):
    """FAN block from https://arxiv.org/abs/2204.12451"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., sharpen_attn=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, eta=1., sr_ratio=1., downsample=None, c_head_num=None):
        """Initialize FANBlock class"""
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TokenMixing(dim, num_heads=num_heads, qkv_bias=qkv_bias, mlp_hidden_dim=int(dim * mlp_ratio), sharpen_attn=sharpen_attn,
                                attn_drop=attn_drop, proj_drop=drop, drop=drop, drop_path=drop_path, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = ChannelProcessing(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                     drop_path=drop_path, drop=drop, mlp_hidden_dim=int(dim * mlp_ratio), c_head_num=c_head_num)

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


class FAN(nn.Module):
    """Based on timm code bases
    https://github.com/rwightman/pytorch-image-models/tree/master/timm
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, sharpen_attn=False, channel_dims=None,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., sr_ratio=None, backbone=None, use_checkpoint=False,
                 act_layer=None, norm_layer=None, se_mlp=False, cls_attn_layers=2, use_pos_embed=True, eta=1., tokens_norm=False, c_head_num=None, hybrid_patch_size=2, head_init_scale=1.0,
                 **kwargs):
        """ Init Module """
        super().__init__()
        if not isinstance(img_size, tuple):
            img_size = to_2tuple(img_size)
        self.head_init_scale = head_init_scale
        self.use_checkpoint = use_checkpoint
        # assert (img_size[0] % patch_size == 0) and (img_size[0] % patch_size == 0), \
        #     '`patch_size` should divide image dimensions evenly'

        self.num_classes = num_classes
        num_heads = [num_heads] * depth if not isinstance(num_heads, list) else num_heads

        channel_dims = [embed_dim] * depth if channel_dims is None else channel_dims
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        if backbone is None:
            self.patch_embed = ConvPatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, act_layer=act_layer)
        else:
            self.patch_embed = HybridEmbed(img_size=img_size, backbone=backbone, patch_size=hybrid_patch_size, embed_dim=embed_dim, in_chans=in_chans)
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
                downsample = OverlapPatchEmbed(img_size=img_size,
                                               patch_size=3,
                                               stride=2,
                                               in_chans=channel_dims[i],
                                               embed_dim=channel_dims[i + 1])
            else:
                downsample = None
            self.blocks.append(build_block(dim=channel_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, sr_ratio=sr_ratio[i],
                                           attn_drop=attn_drop_rate, drop_path=drop_path_rate, act_layer=act_layer, norm_layer=norm_layer, eta=eta,
                                           downsample=downsample, c_head_num=c_head_num[i] if c_head_num is not None else None))
        self.num_features = self.embed_dim = channel_dims[i]
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, channel_dims[i]))
        # self.cls_attn_blocks = nn.ModuleList([ClassAttentionBlock(dim=channel_dims[-1], num_heads=num_heads[-1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
        #                                                           attn_drop=attn_drop_rate, act_layer=act_layer, norm_layer=norm_layer, eta=eta, tokens_norm=tokens_norm)
        #                                       for _ in range(cls_attn_layers)])

        # # Classifier head
        # self.norm = norm_layer(channel_dims[i])

        # # Init weights
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """layers to ignore for weight decay"""
        return {'pos_embed', 'cls_token'}  # , 'patch_embed'}

    def get_classifier(self):
        """Returns classifier"""
        return self.head

    def forward_features(self, x):
        """Extract features"""
        B = x.shape[0]
        x, (Hp, Wp) = self.patch_embed(x)

        if self.use_pos_embed:
            pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        x = self.pos_drop(x)
        H, W = Hp, Wp
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=True)
            else:
                x = blk(x)
            H, W = blk.H, blk.W

        x = x.permute(0, 2, 1).reshape(B, -1, H, W)

        return x

    def forward(self, x):
        """Base forward function"""
        x = self.forward_features(x)
        return x

    def get_last_selfattention(self, x, use_cls_attn=False, layer_idx=11):
        """ Output of Self-Attention """
        B = x.shape[0]
        x, (Hp, Wp) = self.patch_embed(x)

        if self.use_pos_embed:
            pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        x = self.pos_drop(x)

        return_idx = layer_idx or len(self.blocks) - 1

        for i, blk in enumerate(self.blocks):
            if i == return_idx:
                x, attn = blk(x, Hp, Wp, return_attention=True)
            else:
                x, Hp, Wp = blk(x, Hp, Wp)

        if use_cls_attn:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            for i, blk in enumerate(self.cls_attn_blocks):
                if i < len(self.cls_attn_blocks) - 1:
                    x = blk(x)
                else:
                    attn = blk(x, return_attention=True)
                    return attn
        return attn


# FAN-ViT Models
# @BACKBONES.register_module()
# class fan_tiny_12_p16_224(FAN):
#     """ FAN Tiny ViT """

#     def __init__(self, **kwargs):
#         """ Init Function """
#         depth = 12
#         sr_ratio = [1] * (depth // 2) + [1] * (depth // 2)
#         model_kwargs = dict(
#             patch_size=16, embed_dim=192, depth=depth, num_heads=4, eta=1.0, tokens_norm=True, sharpen_attn=False, sr_ratio=sr_ratio, **kwargs)
#         super(fan_tiny_12_p16_224, self).__init__(**model_kwargs)


# @BACKBONES.register_module()
# class fan_small_12_p16_224_se_attn(FAN):
#     """ FAN Small SE ViT """

#     def __init__(self, **kwargs):
#         """ Init Module """
#         depth = 12
#         sr_ratio = [1] * (depth // 2) + [1] * (depth // 2)
#         model_kwargs = dict(patch_size=16, embed_dim=384, depth=depth, num_heads=8, eta=1.0,
#                             tokens_norm=True, sharpen_attn=False, se_mlp=True, sr_ratio=sr_ratio, **kwargs)
#         super(fan_small_12_p16_224_se_attn, self).__init__(**model_kwargs)


# @BACKBONES.register_module()
# class fan_small_12_p16_224(FAN):
#     """ FAN Small ViT """

#     def __init__(self, **kwargs):
#         """ Init Module """
#         depth = 12
#         sr_ratio = [1] * depth
#         model_kwargs = dict(
#             patch_size=16, embed_dim=384, depth=depth, num_heads=8, eta=1.0, tokens_norm=True, sr_ratio=sr_ratio, **kwargs)
#         super(fan_small_12_p16_224, self).__init__(**model_kwargs)


# @BACKBONES.register_module()
# class fan_base_18_p16_224(FAN):
#     """ FAN Base ViT """

#     def __init__(self, **kwargs):
#         """ Init Module """
#         depth = 18
#         sr_ratio = [1] * (depth // 2) + [1] * (depth // 2)
#         model_kwargs = dict(
#             patch_size=16, embed_dim=448, depth=depth, num_heads=8, eta=1.0, tokens_norm=True, sharpen_attn=False, sr_ratio=sr_ratio, **kwargs)
#         super(fan_base_18_p16_224, self).__init__(**model_kwargs)


# @BACKBONES.register_module()
# class fan_large_24_p16_224(FAN):
#     """ FAN Large ViT """

#     def __init__(self, **kwargs):
#         """ Init Module """
#         depth = 24
#         sr_ratio = [1] * (depth // 2) + [1] * (depth // 2)
#         model_kwargs = dict(
#             patch_size=16, embed_dim=480, depth=depth, num_heads=10, eta=1.0, tokens_norm=True, sharpen_attn=False, sr_ratio=sr_ratio, **kwargs)
#         super(fan_large_24_p16_224, self).__init__(**model_kwargs)


# FAN-Hybrid Models
# CNN backbones are based on ConvNeXt architecture with only first two stages for downsampling purpose
# This has been verified to be beneficial for downstream tasks
class fan_tiny_8_p2_hybrid(FAN):
    """ FAN Tiny Hybrid """

    def __init__(self, **kwargs):
        """ Init Module """
        depth = 8
        sr_ratio = [1] * (depth // 2) + [1] * (depth // 2 + 1)
        model_args = dict(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False, patch_size=2, in_chans=kwargs["in_chans"])
        backbone = _create_hybrid_backbone(pretrained=False, pretrained_strict=False, **model_args)
        model_kwargs = dict(img_size=(kwargs["in_height"], kwargs["in_width"]), patch_size=16, embed_dim=192, depth=depth, num_heads=8, eta=1.0, tokens_norm=True, sharpen_attn=False, sr_ratio=sr_ratio, backbone=backbone, **kwargs)
        super(fan_tiny_8_p2_hybrid, self).__init__(**model_kwargs)


# @BACKBONES.register_module()
# class fan_small_12_p4_hybrid(FAN):
#     """ FAN Small Hybrid """

#     def __init__(self, **kwargs):
#         """Init Module"""
#         depth = 10
#         channel_dims = [384] * 10 + [384] * (depth - 10)
#         sr_ratio = [1] * (depth // 2) + [1] * (depth // 2)
#         model_args = dict(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False)
#         backbone = _create_hybrid_backbone(pretrained=False, pretrained_strict=False, **model_args)
#         model_kwargs = dict(
#             patch_size=16, embed_dim=384, depth=depth, num_heads=8, eta=1.0, channel_dims=channel_dims, tokens_norm=True, sharpen_attn=False, backbone=backbone, sr_ratio=sr_ratio, **kwargs)
#         super(fan_small_12_p4_hybrid, self).__init__(**model_kwargs)


# @BACKBONES.register_module()
# class fan_base_16_p4_hybrid(FAN):
#     """ FAN Base Hybrid """

#     def __init__(self, **kwargs):
#         """ Init Module """
#         depth = 16
#         sr_ratio = [1] * (depth // 2) + [1] * (depth // 2)
#         model_args = dict(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False)
#         backbone = _create_hybrid_backbone(pretrained=False, pretrained_strict=False, **model_args)
#         model_kwargs = dict(
#             patch_size=16, embed_dim=448, depth=depth, num_heads=8, eta=1.0, tokens_norm=True, sharpen_attn=False, sr_ratio=sr_ratio, backbone=backbone, **kwargs)
#         super(fan_base_16_p4_hybrid, self).__init__(**model_kwargs)


# @BACKBONES.register_module()
# class fan_large_16_p4_hybrid(FAN):
#     """ FAN Large Hybrid """

#     def __init__(self, **kwargs):
#         """Init Module"""
#         depth = 22
#         sr_ratio = [1] * (depth // 2) + [1] * (depth // 2)
#         model_args = dict(depths=[3, 5], dims=[128, 256, 512, 1024], use_head=False)
#         backbone = _create_hybrid_backbone(pretrained=False, pretrained_strict=False, **model_args)
#         model_kwargs = dict(
#             patch_size=16, embed_dim=480, depth=depth, num_heads=10, eta=1.0, tokens_norm=True, sharpen_attn=False, head_init_scale=0.001, backbone=backbone, sr_ratio=sr_ratio, **kwargs)
#         super(fan_large_16_p4_hybrid, self).__init__(**model_kwargs)


# @BACKBONES.register_module()
# class fan_Xlarge_16_p4_hybrid(FAN):
#     """FAN XLarge hybrid"""

#     def __init__(self, **kwargs):
#         """Init Module"""
#         depth = 23
#         stage_depth = 20
#         channel_dims = [528] * stage_depth + [768] * (depth - stage_depth)
#         num_heads = [11] * stage_depth + [16] * (depth - stage_depth)
#         sr_ratio = [1] * (depth // 2) + [1] * (depth // 2 + 1)
#         model_args = dict(depths=[3, 7], dims=[128, 256, 512, 1024], use_head=False)
#         backbone = _create_hybrid_backbone(pretrained=False, pretrained_strict=False, **model_args)
#         model_kwargs = dict(
#             patch_size=16, embed_dim=channel_dims[0], depth=depth, num_heads=num_heads, eta=1.0, tokens_norm=True, sharpen_attn=False, sr_ratio=sr_ratio, channel_dims=channel_dims, backbone=backbone, **kwargs)
#         super(fan_Xlarge_16_p4_hybrid, self).__init__(**model_kwargs)
