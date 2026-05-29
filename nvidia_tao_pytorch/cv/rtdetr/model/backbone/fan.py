# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FAN backbone for RT-DETR."""

import torch
import torch.utils.checkpoint as checkpoint
from torch import nn

from nvidia_tao_pytorch.cv.backbone_v2.convnext_utils import ConvNeXtFANBackbone
from nvidia_tao_pytorch.cv.backbone_v2.fan import FAN, HybridEmbed
from nvidia_tao_pytorch.cv.rtdetr.model.backbone.registry import RTDETR_BACKBONE_REGISTRY


class FANFPN(FAN):
    """FAN FPN module."""

    def __init__(
        self,
        out_index,
        out_channels=[128, 256, 384, 768],
        return_idx=[0, 1, 2, 3],
        **kwargs,
    ):
        """Init"""
        embed_dim = kwargs.get("embed_dim", None)
        assert embed_dim is not None, "embed_dim should be provided"
        super().__init__(**kwargs)
        if not isinstance(self.patch_embed, HybridEmbed):
            raise NotImplementedError(
                f"FANFPN only supports HybridEmbed. Received: self.patch_embed of type {type(self.patch_embed)}"
            )

        if isinstance(out_index, int):
            out_index = [out_index]
        self.out_index = out_index
        self.return_idx = return_idx

        for idx in self.return_idx:
            layer = nn.LayerNorm(out_channels[idx])
            layer_name = f"out_norm{idx}"
            self.add_module(layer_name, layer)
        self.out_channels = [out_channels[_i] for _i in return_idx]
        self.learnable_downsample = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=768,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            groups=1,
            bias=True,
        )

    def forward_feature_pyramid(self, x):
        """Forward"""
        outs = []
        B = x.shape[0]
        x, (Hp, Wp), out_list = self.patch_embed(x, return_feat=True)
        outs = outs + out_list

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
                x = checkpoint.checkpoint(blk, x)

            Hp, Wp = blk.H, blk.W

            if idx in self.out_index:
                outs.append(x.reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous())
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.cls_attn_blocks:
            x = blk(x)

        x = x[:, 1:, :].reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()
        x = self.learnable_downsample(x)
        outs.append(x)
        final_outs = []
        for i, out in enumerate(outs):
            if i in self.return_idx:
                out = out.permute(0, 2, 3, 1).contiguous()
                norm_layer = getattr(self, f"out_norm{i}")
                out = norm_layer(out)
                final_outs.append(out.permute(0, 3, 1, 2).contiguous())
        del outs
        return final_outs


@RTDETR_BACKBONE_REGISTRY.register()
def fan_tiny_8_p4_hybrid(out_indices=[1, 2, 3], activation_checkpoint=True, **kwargs):
    """FAN Hybrid Tiny

    Args:
        out_indices (list): List of block indices to return as feature
    """
    depth = 8
    sr_ratio = [1.0] * depth
    backbone = ConvNeXtFANBackbone(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False)
    return FANFPN(
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=192,
        depth=depth,
        backbone=backbone,
        num_heads=8,
        sr_ratio=sr_ratio,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        act_layer=None,
        norm_layer=None,
        cls_attn_layers=2,
        use_pos_embed=True,
        eta=1.0,
        tokens_norm=True,
        # FPN parameters.
        out_index=7,
        out_channels=[128, 256, 192, 768],
        return_idx=out_indices,
        activation_checkpoint=activation_checkpoint,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def fan_small_12_p4_hybrid(out_indices=[1, 2, 3], activation_checkpoint=True, **kwargs):
    """FAN Hybrid Small

    Args:
        out_indices (list): List of block indices to return as feature
    """
    depth = 10
    sr_ratio = [1.0] * depth
    backbone = ConvNeXtFANBackbone(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False)
    return FANFPN(
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=384,
        depth=depth,
        backbone=backbone,
        num_heads=8,
        sr_ratio=sr_ratio,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        act_layer=None,
        norm_layer=None,
        cls_attn_layers=2,
        use_pos_embed=True,
        eta=1.0,
        tokens_norm=True,
        # FPN parameters.
        out_index=9,
        out_channels=[128, 256, 384, 768],
        return_idx=out_indices,
        activation_checkpoint=activation_checkpoint,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def fan_base_12_p4_hybrid(out_indices=[1, 2, 3], activation_checkpoint=True, **kwargs):
    """FAN Hybrid Base

    Args:
        out_indices (list): List of block indices to return as feature
    """
    depth = 16
    sr_ratio = [1.0] * depth
    backbone = ConvNeXtFANBackbone(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False)
    return FANFPN(
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=448,
        depth=depth,
        backbone=backbone,
        num_heads=8,
        sr_ratio=sr_ratio,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        act_layer=None,
        norm_layer=None,
        cls_attn_layers=2,
        use_pos_embed=True,
        eta=1.0,
        tokens_norm=True,
        # FPN parameters.
        out_index=15,
        out_channels=[128, 256, 448, 768],
        return_idx=out_indices,
        activation_checkpoint=activation_checkpoint,
        **kwargs,
    )


@RTDETR_BACKBONE_REGISTRY.register()
def fan_large_12_p4_hybrid(out_indices=[1, 2, 3], activation_checkpoint=True, **kwargs):
    """FAN Hybrid Large

    Args:
        out_indices (list): List of block indices to return as feature
    """
    depth = 22
    sr_ratio = [1.0] * depth
    backbone = ConvNeXtFANBackbone(depths=[3, 5], dims=[128, 256, 512, 1024], use_head=False)
    return FANFPN(
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=480,
        depth=depth,
        backbone=backbone,
        num_heads=10,
        sr_ratio=sr_ratio,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        act_layer=None,
        norm_layer=None,
        cls_attn_layers=2,
        use_pos_embed=True,
        eta=1.0,
        tokens_norm=True,
        # FPN parameters.
        out_index=18,
        out_channels=[128, 256, 480, 768],
        return_idx=out_indices,
        activation_checkpoint=activation_checkpoint,
        **kwargs,
    )
