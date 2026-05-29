# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ViT Adapter backbone."""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import PatchEmbed, SwiGLUPacked, trunc_normal_
from torch.nn.init import normal_

from nvidia_tao_pytorch.cv.backbone_v2.radio import RADIO
from nvidia_tao_pytorch.cv.backbone_v2.vit import VisionTransformer
from nvidia_tao_pytorch.cv.deformable_detr.model.ops.modules import MSDeformAttn
from nvidia_tao_pytorch.cv.visual_changenet.backbone.adapter_modules import (
    InteractionBlock,
    SpatialPriorModule,
    deform_inputs,
)


class ViTAdapter(VisionTransformer):
    """ViT-Adapter from https://arxiv.org/abs/2205.08534."""

    def __init__(
        self,
        *args,
        pretrain_size=224,
        num_heads=12,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        init_values=0.0,
        interaction_indexes=None,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        add_vit_feature=True,
        use_extra_extractor=True,
        out_indices=[0, 1, 2, 3],
        activation_checkpoint=False,
        add_summary=True,
        export=False,
        **kwargs,
    ):
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
            add_summary (bool): Use summary token of backbone or not.
            export (bool): Whether to enable export mode. If `True`, replace BN with FrozenBN.
        """
        norm_layer = kwargs.get("norm_layer", None)
        patch_size = kwargs.get("patch_size", 16)
        drop_path_rate = kwargs.get("drop_path_rate", 0.0)
        super().__init__(
            num_classes=0,
            num_heads=num_heads,
            init_values=init_values,
            export=export,
            *args,
            **kwargs,
        )

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.add_summary = add_summary
        self.num_summary = 1
        self.patch_size = int(patch_size)
        self.norm_layer = norm_layer
        self.drop_path_rate = float(drop_path_rate)

        embed_dim = self.embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_summary, embed_dim))
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(in_channel=3,
                                      patch_size=self.patch_size,
                                      inplanes=conv_inplane,
                                      embed_dim=embed_dim,
                                      out_indices=out_indices)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((i == len(interaction_indexes) - 1) and use_extra_extractor),
                             with_cp=activation_checkpoint)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

        if self.add_summary:
            self.fc_summary = nn.Linear(self.num_summary * embed_dim, embed_dim)
            self.conv1 = nn.Conv2d(2 * embed_dim, embed_dim, 1)
            self.conv2 = nn.Conv2d(2 * embed_dim, embed_dim, 1)
            self.conv3 = nn.Conv2d(2 * embed_dim, embed_dim, 1)
            self.conv4 = nn.Conv2d(2 * embed_dim, embed_dim, 1)
            self.conv1.apply(self._init_weights)
            self.conv2.apply(self._init_weights)
            self.conv3.apply(self._init_weights)
            self.conv4.apply(self._init_weights)

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
            1,
            self.pretrain_size[0] // self.patch_size,
            self.pretrain_size[1] // self.patch_size,
            -1,
        ).permute(0, 3, 1, 2)
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
        x = self.patch_embed.proj(x)
        _, _, H, W = x.shape
        if self.patch_embed.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.patch_embed.norm(x)
        bs, _, dim = x.shape
        cls_token = self.cls_token.expand(bs, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        pos_embed_interpolate = self._get_pos_embed(self.pos_embed[:, self.num_summary:], H, W)
        pos_embed = torch.cat((self.pos_embed[:, :self.num_summary], pos_embed_interpolate), dim=1)
        x = self.pos_drop(x + pos_embed)

        # Interaction
        outs = []
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W, self.num_summary)
            outs.append(x[:, self.num_summary:].transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        if self.add_summary:
            summary = x[:, :1].view(bs, -1)
            summary = self.fc_summary(summary)
            summary = summary.unsqueeze(2).unsqueeze(3)
            c1 = torch.cat([summary.expand(-1, -1, c1.shape[2], c1.shape[3]), c1], dim=1)
            c2 = torch.cat([summary.expand(-1, -1, c2.shape[2], c2.shape[3]), c2], dim=1)
            c3 = torch.cat([summary.expand(-1, -1, c3.shape[2], c3.shape[3]), c3], dim=1)
            c4 = torch.cat([summary.expand(-1, -1, c4.shape[2], c4.shape[3]), c4], dim=1)
            c1 = self.conv1(c1)
            c2 = self.conv2(c2)
            c3 = self.conv3(c3)
            c4 = self.conv4(c4)

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]

    def load_state_dict(self, state_dict, **kwargs):
        """This function is used to load the pretrained weights."""
        from torch.nn.modules.module import _IncompatibleKeys

        _missing_keys, _unexpected_keys = super().load_state_dict(state_dict, **kwargs)
        # Remove the ViTAdapter keys.
        missing_keys = []
        ignore_keys = ["level_embed", "spm", "interactions", "up", "norm", "fc_summary", "conv"]
        for k in _missing_keys:
            if any(k.startswith(ignore_key) for ignore_key in ignore_keys):
                continue
            missing_keys.append(k)
        return _IncompatibleKeys(missing_keys, _unexpected_keys)

    def freeze_backbone(self):
        """Freeze specific parts of the backbone and batch normalization layers.

        Unfreeze the modules that are not part of the pretrained backbone.
        """
        super().freeze_backbone()

        modules = [
            self.cls_token,
            self.level_embed,
            self.spm,
            self.interactions,
            self.up,
            self.norm1,
            self.norm2,
            self.norm3,
            self.norm4,
        ]
        if self.add_summary:
            modules += [self.fc_summary, self.conv1, self.conv2, self.conv3, self.conv4]
        for m in modules:
            if isinstance(m, nn.Parameter):
                m.requires_grad = True
            elif isinstance(m, nn.Module):
                for param in m.parameters():
                    param.requires_grad = True
                m.train()


class CRADIOAdapter(nn.Module):
    """ViT-Adapter from https://arxiv.org/abs/2205.08534."""

    def __init__(
        self,
        model_name="vit_huge_patch16_224_mlpnorm",
        model_cfg=None,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        init_values=0.0,
        interaction_indexes=None,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        add_vit_feature=True,
        use_extra_extractor=True,
        out_indices=[0, 1, 2, 3],
        activation_checkpoint=False,
        add_summary=True,
        export=False,
        **kwargs,
    ):
        """ViT-Adapter Constructor.

        Args:
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
            add_summary (bool): Use summary token of backbone or not.
            export (bool): Whether to enable export mode. If `True`, replace BN with FrozenBN.
        """
        super().__init__()

        self.radio: RADIO = RADIO(
            num_classes=0,
            backbone=model_name,
            export=export,
            **model_cfg,
            **kwargs
        )
        self.num_block = len(self.radio.radio.radio.model.blocks)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.add_summary = add_summary
        self.num_summary = self.radio.radio.radio.num_summary_tokens
        embed_dim = self.radio.radio.radio.model.embed_dim
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(in_channel=3,
                                      patch_size=self.radio.radio.radio.model.patch_generator.patch_size,
                                      inplanes=conv_inplane,
                                      embed_dim=embed_dim,
                                      out_indices=out_indices)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=0.0,
                             norm_layer=nn.LayerNorm, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((i == len(interaction_indexes) - 1) and use_extra_extractor),
                             with_cp=activation_checkpoint)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

        if self.add_summary:
            self.fc_summary = nn.Linear(self.num_summary * embed_dim, embed_dim)
            self.conv1 = nn.Conv2d(2 * embed_dim, embed_dim, 1)
            self.conv2 = nn.Conv2d(2 * embed_dim, embed_dim, 1)
            self.conv3 = nn.Conv2d(2 * embed_dim, embed_dim, 1)
            self.conv4 = nn.Conv2d(2 * embed_dim, embed_dim, 1)
            self.conv1.apply(self._init_weights)
            self.conv2.apply(self._init_weights)
            self.conv3.apply(self._init_weights)
            self.conv4.apply(self._init_weights)

        if export:
            # Utilize the `_freeze_bn_norm` from `BackboneBase`.
            self.radio._freeze_bn_norm(self)

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
        x = self.radio.radio.radio.model.patch_generator(x)
        bs = x.shape[0]
        dim = self.radio.radio.radio.model.embed_dim
        H, W = self.radio.radio.radio.model.patch_generator.input_dims

        # Interaction
        outs = []
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.radio.radio.radio.model.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W, self.num_summary)
            outs.append(x[:, self.num_summary:].transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        if self.add_summary:
            summary = x[:, :self.num_summary].view(bs, -1)
            summary = self.fc_summary(summary)
            summary = summary.unsqueeze(2).unsqueeze(3)
            c1 = torch.cat([summary.expand(-1, -1, c1.shape[2], c1.shape[3]), c1], dim=1)
            c2 = torch.cat([summary.expand(-1, -1, c2.shape[2], c2.shape[3]), c2], dim=1)
            c3 = torch.cat([summary.expand(-1, -1, c3.shape[2], c3.shape[3]), c3], dim=1)
            c4 = torch.cat([summary.expand(-1, -1, c4.shape[2], c4.shape[3]), c4], dim=1)
            c1 = self.conv1(c1)
            c2 = self.conv2(c2)
            c3 = self.conv3(c3)
            c4 = self.conv4(c4)

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]

    def load_state_dict(self, state_dict, **kwargs):
        """This function is used to load the pretrained weights."""
        return self.radio.load_state_dict(state_dict, **kwargs)


def vit_large_nvdinov2(
    out_indices=[0, 1, 2, 3], resolution=1024, activation_checkpoint=False, use_summary_token=True, **kwargs
):
    """ViT-Large NV-DINOv2 model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.

    Return:
        model: ViT model.
    """
    return ViTAdapter(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        init_values=1e-5,
        img_size=resolution,
        drop_path_rate=0.4,
        mlp_layer=SwiGLUPacked,
        act_layer=nn.SiLU,
        mlp_ratio=5472 / 1024,
        embed_layer=partial(PatchEmbed, strict_img_size=False),
        global_pool="token",
        pretrain_size=resolution,
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        out_indices=out_indices,
        activation_checkpoint=activation_checkpoint,
        add_summary=use_summary_token,
        **kwargs,
    )


def vit_large_dinov2(
    out_indices=[0, 1, 2, 3], resolution=1024, activation_checkpoint=False, use_summary_token=True, **kwargs
):
    """ViT-Large DINOv2 model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.

    Return:
        model: ViT model.
    """
    return ViTAdapter(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        init_values=1e-5,
        img_size=resolution,
        drop_path_rate=0.4,
        pretrain_size=resolution,
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        out_indices=out_indices,
        activation_checkpoint=activation_checkpoint,
        add_summary=use_summary_token,
        **kwargs,
    )


def c_radio_p1_vit_huge_patch16_224_mlpnorm(
    out_indices=[0, 1, 2, 3], resolution=1024, activation_checkpoint=False, use_summary_token=True, **kwargs
):
    """ViT-Huge C-RADIO model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.

    Return:
        model: ViT model.
    """
    model_cfg = {
        "summary_idxs": [0, 1, 2],
        "window_size": None,
        "num_teacher": 3,
        "cpe_max_size": 2048,
        "register_multiple": 16,
    }
    return CRADIOAdapter(
        model_name="vit_huge_patch16_224_mlpnorm",
        model_cfg=model_cfg,
        img_size=resolution,
        drop_path_rate=0.4,
        init_values=1e-5,
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 7], [8, 15], [16, 23], [24, 31]],
        out_indices=out_indices,
        activation_checkpoint=activation_checkpoint,
        add_summary=use_summary_token,
        **kwargs
    )


def c_radio_p2_vit_huge_patch16_224_mlpnorm(
    out_indices=[0, 1, 2, 3], resolution=1024, activation_checkpoint=False, use_summary_token=True, **kwargs
):
    """ViT-Huge C-RADIO model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.

    Return:
        model: ViT model.
    """
    model_cfg = {
        "summary_idxs": [0, 1, 2, 3],
        "window_size": None,
        "num_teacher": 4,
        "cpe_max_size": 2048,
        "register_multiple": 16,
    }
    return CRADIOAdapter(
        model_name="vit_huge_patch16_224_mlpnorm",
        model_cfg=model_cfg,
        img_size=resolution,
        drop_path_rate=0.4,
        init_values=1e-5,
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 7], [8, 15], [16, 23], [24, 31]],
        out_indices=out_indices,
        activation_checkpoint=activation_checkpoint,
        add_summary=use_summary_token,
        **kwargs
    )


def c_radio_p3_vit_huge_patch16_224_mlpnorm(
    out_indices=[0, 1, 2, 3], resolution=1024, activation_checkpoint=False, use_summary_token=True, **kwargs
):
    """ViT-Huge C-RADIO model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.

    Return:
        model: ViT model.
    """
    model_cfg = {
        "summary_idxs": [0, 1, 2],
        "window_size": None,
        "num_teacher": 4,
        "cpe_max_size": 2048,
        "register_multiple": 16,
    }
    return CRADIOAdapter(
        model_name="vit_huge_patch16_224_mlpnorm",
        model_cfg=model_cfg,
        img_size=resolution,
        drop_path_rate=0.4,
        init_values=1e-5,
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 7], [8, 15], [16, 23], [24, 31]],
        out_indices=out_indices,
        activation_checkpoint=activation_checkpoint,
        add_summary=use_summary_token,
        **kwargs,
    )


def c_radio_v2_vit_huge_patch16_224(
    out_indices=[0, 1, 2, 3], resolution=1024, activation_checkpoint=False, use_summary_token=True, **kwargs
):
    """ViT-Huge C-RADIO model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.

    Return:
        model: ViT model.
    """
    model_cfg = {
        "summary_idxs": [0, 1, 2],
        "window_size": None,
        "num_teacher": 4,
        "cpe_max_size": 2048,
        "register_multiple": 8,
    }
    return CRADIOAdapter(
        model_name="vit_huge_patch16_224",
        model_cfg=model_cfg,
        img_size=resolution,
        drop_path_rate=0,
        init_values=1e-5,
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 7], [8, 15], [16, 23], [24, 31]],
        out_indices=out_indices,
        activation_checkpoint=activation_checkpoint,
        add_summary=use_summary_token,
        **kwargs,
    )


def c_radio_v2_vit_large_patch16_224(
    out_indices=[0, 1, 2, 3], resolution=1024, activation_checkpoint=False, use_summary_token=True, **kwargs
):
    """ViT-Huge C-RADIO model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.

    Return:
        model: ViT model.
    """
    model_cfg = {
        "summary_idxs": [0, 1, 2],
        "window_size": None,
        "num_teacher": 4,
        "cpe_max_size": 2048,
        "register_multiple": 8,
    }
    return CRADIOAdapter(
        model_name="vit_large_patch16_224",
        model_cfg=model_cfg,
        img_size=resolution,
        drop_path_rate=0,
        init_values=1e-5,
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        out_indices=out_indices,
        activation_checkpoint=activation_checkpoint,
        add_summary=use_summary_token,
        **kwargs,
    )


def c_radio_v2_vit_base_patch16_224(
    out_indices=[0, 1, 2, 3], resolution=1024, activation_checkpoint=False, use_summary_token=True, **kwargs
):
    """ViT-Base C-RADIO model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.

    Return:
        model: ViT model.
    """
    model_cfg = {
        "summary_idxs": [0, 1, 2],
        "window_size": None,
        "num_teacher": 4,
        "cpe_max_size": 2048,
        "register_multiple": 8,
    }
    return CRADIOAdapter(
        model_name="vit_base_patch16_224",
        model_cfg=model_cfg,
        img_size=resolution,
        drop_path_rate=0,
        init_values=1e-5,
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        out_indices=out_indices,
        activation_checkpoint=activation_checkpoint,
        add_summary=use_summary_token,
        **kwargs,
    )


vit_adapter_model_dict = {
    "vit_large_nvdinov2": vit_large_nvdinov2,
    "vit_large_dinov2": vit_large_dinov2,  # TODO: @zbhat check EVA/dinov2 support
    "c_radio_p1_vit_huge_patch16_224_mlpnorm": c_radio_p1_vit_huge_patch16_224_mlpnorm,
    "c_radio_p2_vit_huge_patch16_224_mlpnorm": c_radio_p2_vit_huge_patch16_224_mlpnorm,
    "c_radio_p3_vit_huge_patch16_224_mlpnorm": c_radio_p3_vit_huge_patch16_224_mlpnorm,
    "c_radio_v2_vit_huge_patch16_224": c_radio_v2_vit_huge_patch16_224,
    "c_radio_v2_vit_large_patch16_224": c_radio_v2_vit_large_patch16_224,
    "c_radio_v2_vit_base_patch16_224": c_radio_v2_vit_base_patch16_224,
}
