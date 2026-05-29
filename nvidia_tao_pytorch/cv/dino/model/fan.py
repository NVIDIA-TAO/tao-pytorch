# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FAN Backbone for DINO"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from timm.layers import trunc_normal_, to_2tuple

from nvidia_tao_pytorch.cv.backbone_v2.convnext_utils import ConvNeXtFANBackbone
from nvidia_tao_pytorch.cv.backbone_v2.fan import FAN, HybridEmbed


class FANFPN(FAN):
    """FAN implementation from https://arxiv.org/abs/2204.12451
    Based on timm https://github.com/rwightman/pytorch-image-models/tree/master/timm
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 sr_ratio=None,
                 backbone=None,
                 act_layer=None,
                 norm_layer=None,
                 cls_attn_layers=2,
                 use_pos_embed=True,
                 eta=1.,
                 tokens_norm=False,
                 out_index=-1,
                 out_channels=None,
                 out_indices=[0, 1, 2, 3],
                 activation_checkpoint=True,
                 **kwargs):
        """Initialize FAN class"""
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            sharpen_attn=False,
            channel_dims=None,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            sr_ratio=sr_ratio,
            backbone=backbone,
            act_layer=act_layer,
            norm_layer=norm_layer,
            se_mlp=False,
            cls_attn_layers=cls_attn_layers,
            use_pos_embed=use_pos_embed,
            eta=eta,
            tokens_norm=tokens_norm,
            c_head_num=None,
            hybrid_patch_size=2,
            head_init_scale=1.0,
            activation_checkpoint=activation_checkpoint,
            **kwargs)

        # remove self.norm and self.head
        delattr(self, 'norm')
        delattr(self, 'head')

        img_size = to_2tuple(img_size)

        self.activation_checkpoint = activation_checkpoint
        self.out_index = out_index
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.out_indices = out_indices

        for i_layer in self.out_indices:
            layer = nn.LayerNorm(self.out_channels[i_layer])
            layer_name = f'out_norm{i_layer}'
            self.add_module(layer_name, layer)

        self.learnable_downsample = nn.Conv2d(in_channels=embed_dim,
                                              out_channels=768,
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              dilation=1,
                                              groups=1,
                                              bias=True)

        # Init weights
        trunc_normal_(self.cls_token, std=.02)
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
        return {'pos_embed', 'cls_token'}

    def forward_feature_pyramid(self, x):
        """Extract features
        Args:
            x: tensor

        Returns:
            final_outs: dictionary containing indice name as key and indice feature as value
        """
        outs = []
        B = x.shape[0]
        if isinstance(self.patch_embed, HybridEmbed):
            x, (Hp, Wp), out_list = self.patch_embed(x, return_feat=True)
            outs = outs + out_list
            out_index = [self.out_index]
        else:
            x, (Hp, Wp) = self.patch_embed(x)

        if self.use_pos_embed:
            pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            blk.H, blk.W = Hp, Wp
            # Disable activation checkpointing during ONNX export
            if torch.onnx.is_in_onnx_export() or not self.activation_checkpoint:
                x = blk(x)
            else:
                x = checkpoint.checkpoint(blk, x, use_reentrant=True)

            Hp, Wp = blk.H, blk.W

            if idx in out_index:
                outs.append(x.reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous())
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.cls_attn_blocks:
            x = blk(x)

        x = x[:, 1:, :].reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()
        x = self.learnable_downsample(x)
        outs.append(x)
        final_outs = {}
        for i, out in enumerate(outs):
            if i in self.out_indices:
                out = out.permute(0, 2, 3, 1).contiguous()
                norm_layer = getattr(self, f'out_norm{i}')
                out = norm_layer(out)
                final_outs[f'p{i}'] = out.permute(0, 3, 1, 2).contiguous()
        del outs
        return final_outs


def fan_tiny_8_p4_hybrid(out_indices=[0, 1, 2, 3], activation_checkpoint=True, **kwargs):
    """FAN Hybrid Tiny

    Args:
        out_indices (list): List of block indices to return as feature
    """
    depth = 8
    sr_ratio = [1.0] * depth
    backbone = ConvNeXtFANBackbone(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False)
    model = FANFPN(
        patch_size=16, in_chans=3, num_classes=0, embed_dim=192, depth=depth, backbone=backbone,
        num_heads=8, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.3, sr_ratio=sr_ratio,
        act_layer=None, norm_layer=None, cls_attn_layers=2, use_pos_embed=True, eta=1., tokens_norm=True,
        out_index=7, out_channels=[128, 256, 192, 768], out_indices=out_indices,
        activation_checkpoint=activation_checkpoint, **kwargs)

    return model


def fan_small_12_p4_hybrid(out_indices=[0, 1, 2, 3], activation_checkpoint=True, **kwargs):
    """FAN Hybrid Small

    Args:
        out_indices (list): List of block indices to return as feature
    """
    depth = 10
    sr_ratio = [1.0] * depth
    backbone = ConvNeXtFANBackbone(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False)
    model = FANFPN(
        patch_size=16, in_chans=3, num_classes=0, embed_dim=384, depth=depth, backbone=backbone,
        num_heads=8, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.3, sr_ratio=sr_ratio,
        act_layer=None, norm_layer=None, cls_attn_layers=2, use_pos_embed=True, eta=1., tokens_norm=True,
        out_index=9, out_channels=[128, 256, 384, 768], out_indices=out_indices,
        activation_checkpoint=activation_checkpoint, **kwargs)

    return model


def fan_base_12_p4_hybrid(out_indices=[0, 1, 2, 3], activation_checkpoint=True, **kwargs):
    """FAN Hybrid Base

    Args:
        out_indices (list): List of block indices to return as feature
    """
    depth = 16
    sr_ratio = [1.0] * depth
    backbone = ConvNeXtFANBackbone(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False)
    model = FANFPN(
        patch_size=16, in_chans=3, num_classes=0, embed_dim=448, depth=depth, backbone=backbone,
        num_heads=8, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.3, sr_ratio=sr_ratio,
        act_layer=None, norm_layer=None, cls_attn_layers=2, use_pos_embed=True, eta=1., tokens_norm=True,
        out_index=15, out_channels=[128, 256, 448, 768], out_indices=out_indices,
        activation_checkpoint=activation_checkpoint, **kwargs)

    return model


def fan_large_12_p4_hybrid(out_indices=[0, 1, 2, 3], activation_checkpoint=True, **kwargs):
    """FAN Hybrid Large

    Args:
        out_indices (list): List of block indices to return as feature
    """
    depth = 22
    sr_ratio = [1.0] * depth
    backbone = ConvNeXtFANBackbone(depths=[3, 5], dims=[128, 256, 512, 1024], use_head=False)
    model = FANFPN(
        patch_size=16, in_chans=3, num_classes=0, embed_dim=480, depth=depth, backbone=backbone,
        num_heads=10, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.3, sr_ratio=sr_ratio,
        act_layer=None, norm_layer=None, cls_attn_layers=2, use_pos_embed=True, eta=1., tokens_norm=True,
        out_index=18, out_channels=[128, 256, 480, 768], out_indices=out_indices,
        activation_checkpoint=activation_checkpoint, **kwargs)

    return model


fan_model_dict = {
    'fan_tiny': fan_tiny_8_p4_hybrid,
    'fan_small': fan_small_12_p4_hybrid,
    'fan_base': fan_base_12_p4_hybrid,
    'fan_large': fan_large_12_p4_hybrid
}
