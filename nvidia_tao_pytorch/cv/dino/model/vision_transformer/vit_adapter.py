# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ViT Adatper backbone."""

from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

from timm.layers import trunc_normal_, SwiGLUPacked, to_2tuple

from nvidia_tao_pytorch.cv.backbone_v2.vit import VisionTransformer as TIMMVisionTransformer
from nvidia_tao_pytorch.cv.deformable_detr.model.ops.modules import MSDeformAttn
from nvidia_tao_pytorch.cv.dino.model.vision_transformer.adapter_modules import (SpatialPriorModule,
                                                                                 InteractionBlock,
                                                                                 deform_inputs)


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding. Returns H, W unlike timm for ViT-Adapter."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 norm_layer=None, flatten=True, bias=True):
        """Initialize PatchEmbed class"""
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """Forward function."""
        x = self.proj(x)
        _, _, H, W = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, H, W


class ViTAdapter(TIMMVisionTransformer):
    """ViT-Adapter from https://arxiv.org/abs/2205.08534."""

    def __init__(self,
                 img_size=224,
                 embed_dim=768,
                 patch_size=16,
                 in_chans=3,
                 depth=12,
                 num_heads=12,
                 conv_inplane=64,
                 n_points=4,
                 deform_num_heads=6,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 drop_rate=0.0,
                 drop_path_rate=0.0,
                 layer_scale=True,
                 init_values=0.,
                 interaction_indexes=None,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 deform_ratio=1.0,
                 add_vit_feature=True,
                 use_extra_extractor=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 out_indices=[0, 1, 2, 3],
                 activation_checkpoint=True,
                 **kwargs):
        """ViT-Adapter Constructor.

        Args:
            num_heads (int): The number of heads in attention modules.
            conv_inplane (int): The hidden dimension of Conv2D in SPM.
            n_points (int): The number of sampling points for
                each query in each head of MultiScaleDeformableAttention.
            deform_num_heads (int): Parallel attention heads of MultiScaleDeformableAttention.
            init_values (float): Init value of LayerScale.
            interaction_indexes (list): The indexes of each interaction block.
            with_cffn (bool): The option to use ffn for adapter. If True, it use ffn.
            cffn_ratio (float): The number of expansion ratio of feedforward
                network hidden layer channels of adapter.
            deform_ratio (float): The expansion ratio of value_proj.
            add_vit_feature (bool): The option to add vit feature to adapter
                feature. If True, it add vit feature.
            use_extra_extractor (bool): The option to use extra Extractor in
                InteractionBlock. If True, it use extra Extractor.
            out_indices (list): List of block indices to return as feature.
            activation_checkpoint (bool): Use activation checkpoint or not.
        """
        super().__init__(img_size=img_size,
                         embed_dim=embed_dim,
                         patch_size=patch_size,
                         in_chans=in_chans,
                         num_classes=0,
                         depth=depth,
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias,
                         drop_rate=drop_rate,
                         drop_path_rate=drop_path_rate,
                         init_values=init_values,
                         norm_layer=norm_layer,
                         activation_checkpoint=activation_checkpoint,
                         **kwargs)

        # remove self.norm and self.head
        delattr(self, 'norm')
        delattr(self, 'head')

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
        )
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.depth = depth
        self.cls_token = None
        self.num_block = len(self.blocks)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature

        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dim))
        self.spm = SpatialPriorModule(in_channel=3,
                                      patch_size=patch_size,
                                      inplanes=conv_inplane,
                                      embed_dim=self.embed_dim,
                                      out_indices=out_indices)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=self.embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((i == len(interaction_indexes) - 1) and use_extra_extractor),
                             with_cp=activation_checkpoint)
            for i in range(len(interaction_indexes))
        ])

        if 0 in out_indices:
            self.up = nn.ConvTranspose2d(self.embed_dim, self.embed_dim, 2, 2)
            self.up.apply(self._init_weights)
        else:
            self.up = None

        self.out_indices = out_indices

        for i_layer in self.out_indices:
            layer = nn.SyncBatchNorm(self.embed_dim)
            layer_name = f'out_norm{i_layer}'
            self.add_module(layer_name, layer)

        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.patch_embed.grid_size[0], self.patch_embed.grid_size[1], -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward_feature_pyramid(self, x):
        """Forward function."""
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, _, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        if self.up is not None:
            c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
            if len(self.out_indices) == 4:
                c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4
            else:
                c2, c3, c4 = c2 + x2, c3 + x3, c4 + x4

        outs = {}
        # Final Norm
        out_features = [c1, c2, c3, c4]
        for idx in self.out_indices:
            level = out_features[idx]
            norm_layer = getattr(self, f'out_norm{idx}')
            outs[f'p{idx}'] = norm_layer(level)
        return outs


def vit_large_nvdinov2(out_indices=[0, 1, 2, 3], resolution=1024, activation_checkpoint=True, **kwargs):
    """ViT-Large NV-DINOv2 model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.

    Return:
        model: ViT model.
    """
    model = ViTAdapter(img_size=resolution,
                       patch_size=16,
                       embed_dim=1024,
                       depth=24,
                       num_heads=16,
                       mlp_ratio=5472 / 1024,
                       drop_path_rate=0.4,
                       init_values=1e-5,
                       mlp_layer=SwiGLUPacked,
                       act_layer=nn.SiLU,
                       conv_inplane=56,
                       n_points=4,
                       deform_num_heads=16,
                       cffn_ratio=0.25,
                       deform_ratio=0.5,
                       interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
                       out_indices=out_indices,
                       activation_checkpoint=activation_checkpoint,
                       **kwargs)

    return model


def vit_large_dinov2(out_indices=[0, 1, 2, 3], resolution=1024, activation_checkpoint=True, **kwargs):
    """ViT-Large DINOv2 model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.

    Return:
        model: ViT model.
    """
    model = ViTAdapter(img_size=resolution,
                       patch_size=16,
                       embed_dim=1024,
                       depth=24,
                       num_heads=16,
                       drop_path_rate=0.4,
                       init_values=1e-5,
                       conv_inplane=56,
                       n_points=4,
                       deform_num_heads=16,
                       cffn_ratio=0.25,
                       deform_ratio=0.5,
                       interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
                       out_indices=out_indices,
                       activation_checkpoint=activation_checkpoint,
                       **kwargs)

    return model


vit_model_dict = {
    'vit_large_nvdinov2': vit_large_nvdinov2,
    'vit_large_dinov2': vit_large_dinov2,
}
