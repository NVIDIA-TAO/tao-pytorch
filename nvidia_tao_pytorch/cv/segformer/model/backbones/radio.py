# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RADIO backbone for Segformer."""

import math

import torch
import torch.nn.functional as F
from timm.layers import trunc_normal_
from torch import nn
from torch.nn.init import normal_

from nvidia_tao_pytorch.cv.backbone_v2.radio import RADIO
from nvidia_tao_pytorch.cv.deformable_detr.model.ops.modules import MSDeformAttn
from nvidia_tao_pytorch.cv.segformer.model.backbones.adapter_modules import (
    RADIOInteractionBlock,
    SpatialPriorModule,
    deform_inputs,
)


class RADIOAdapter(RADIO):
    """ViT-Adapter from https://arxiv.org/abs/2205.08534."""

    def __init__(
        self,
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
        add_summary=True,
        return_idx=[0, 1, 2, 3],
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
            add_summary (bool): Use summary token of backbone or not.
            return_idx (list): List of block indices to return as feature.
            **kwargs: Additional arguments for RADIO backbone.
        """
        super().__init__(**kwargs)

        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.add_summary = add_summary
        self.num_summary = self.radio.radio.num_summary_tokens

        embed_dim = self.radio.radio.model.embed_dim
        self.num_block = len(self.radio.radio.model.blocks)

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        normal_(self.level_embed)
        self.spm = SpatialPriorModule(
            in_channel=3,
            patch_size=self.radio.radio.patch_size,
            inplanes=conv_inplane,
            embed_dim=embed_dim,
            out_indices=return_idx,
        )
        self.spm.apply(self._init_weights)
        self.interactions = nn.Sequential(
            *[
                RADIOInteractionBlock(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=0.0,
                    norm_layer=nn.LayerNorm,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=((i == len(interaction_indexes) - 1) and use_extra_extractor),
                    with_cp=self.activation_checkpoint,
                )
                for i in range(len(interaction_indexes))
            ]
        )
        self.interactions.apply(self._init_weights)
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.up.apply(self._init_weights)
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)
        if self.add_summary:
            self.fc_summary = nn.Linear(self.num_summary * embed_dim, embed_dim)
            self.conv1 = nn.Conv2d(2 * embed_dim, embed_dim, 1)
            self.conv1.apply(self._init_weights)
            self.conv2 = nn.Conv2d(2 * embed_dim, embed_dim, 1)
            self.conv2.apply(self._init_weights)
            self.conv3 = nn.Conv2d(2 * embed_dim, embed_dim, 1)
            self.conv3.apply(self._init_weights)
            self.conv4 = nn.Conv2d(2 * embed_dim, embed_dim, 1)
            self.conv4.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
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

    def freeze_backbone(self):
        """Freeze the backbone while keeping adapter components trainable."""
        super().freeze_backbone()

        # Unfreezes all adapter-specific components to ensure they remain
        # trainable during fine-tuning.
        self.level_embed.requires_grad = True
        modules = [self.spm, self.interactions, self.up, self.norm1, self.norm2, self.norm3, self.norm4]
        if self.add_summary:
            modules.append(self.fc_summary)
            modules.append(self.conv1)
            modules.append(self.conv2)
            modules.append(self.conv3)
            modules.append(self.conv4)
        for m in modules:
            for p in m.parameters():
                p.requires_grad = True
            m.train()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward_feature_pyramid(self, x):
        """Forward pass through the backbone to extract intermediate feature maps."""
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x = self.radio.radio.model.patch_generator(x)
        bs = x.shape[0]
        dim = self.radio.radio.model.embed_dim
        H, W = self.radio.radio.model.patch_generator.input_dims

        # Interaction
        outs = []
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(
                x,
                c,
                self.radio.radio.model.blocks[indexes[0]: indexes[-1] + 1],
                deform_inputs1,
                deform_inputs2,
                H,
                W,
                self.num_summary,
            )
            outs.append(x[:, self.num_summary:].transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0: c2.size(1), :]
        c3 = c[:, c2.size(1): c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode="bilinear", align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode="bilinear", align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        if self.add_summary:
            summary = x[:, : self.num_summary].view(bs, -1)
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


def c_radio_v2_vit_base_patch16_224(
    return_idx=[0, 1, 2, 3], resolution=(1024, 1024), use_summary_token=True, **kwargs
):
    """ViT Base RADIO model.

    Args:
        return_idx (list): List of block indices to return as feature.
        resolution (tuple): Image size.
        use_summary_token (bool): Use summary token of backbone or not.
        **kwargs: Additional arguments for RADIO backbone.

    Return:
        model: RADIOAdapter model.
    """
    return RADIOAdapter(
        # RADIO.
        resolution=resolution,
        backbone="vit_base_patch16_224",
        summary_idxs=[0, 1, 2],
        window_size=None,
        num_teacher=4,
        cpe_max_size=2048,
        register_multiple=8,
        drop_path_rate=0,
        # Adapter.
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        init_values=1e-5,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        cffn_ratio=0.25,
        deform_ratio=0.5,
        add_summary=use_summary_token,
        return_idx=return_idx,
        **kwargs
    )


def c_radio_v2_vit_large_patch16_224(
    return_idx=[0, 1, 2, 3], resolution=(1024, 1024), use_summary_token=True, **kwargs
):
    """ViT Large RADIO model.

    Args:
        return_idx (list): List of block indices to return as feature.
        resolution (tuple): Image size.
        use_summary_token (bool): Use summary token of backbone or not.
        **kwargs: Additional arguments for RADIO backbone.

    Return:
        model: RADIOAdapter model.
    """
    return RADIOAdapter(
        # RADIO.
        resolution=resolution,
        backbone="vit_large_patch16_224",
        summary_idxs=[0, 1, 2],
        window_size=None,
        num_teacher=4,
        cpe_max_size=2048,
        register_multiple=8,
        drop_path_rate=0,
        # Adapter.
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        init_values=1e-5,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        cffn_ratio=0.25,
        deform_ratio=0.5,
        add_summary=use_summary_token,
        return_idx=return_idx,
        **kwargs
    )


def c_radio_v2_vit_huge_patch16_224(
    return_idx=[0, 1, 2, 3], resolution=(1024, 1024), use_summary_token=True, **kwargs
):
    """ViT Huge RADIO model.

    Args:
        return_idx (list): List of block indices to return as feature.
        resolution (tuple): Image size.
        use_summary_token (bool): Use summary token of backbone or not.
        **kwargs: Additional arguments for RADIO backbone.

    Return:
        model: RADIOAdapter model.
    """
    return RADIOAdapter(
        # RADIO.
        resolution=resolution,
        backbone="vit_huge_patch16_224",
        summary_idxs=[0, 1, 2],
        window_size=None,
        num_teacher=4,
        cpe_max_size=2048,
        register_multiple=8,
        drop_path_rate=0,
        # Adapter.
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        init_values=1e-5,
        interaction_indexes=[[0, 7], [8, 15], [16, 23], [24, 31]],
        cffn_ratio=0.25,
        deform_ratio=0.5,
        add_summary=use_summary_token,
        return_idx=return_idx,
        **kwargs
    )


def c_radio_v3_vit_large_patch16_reg4_dinov2(
    return_idx=[0, 1, 2, 3], resolution=(1024, 1024), use_summary_token=True, **kwargs
):
    """ViT Large RADIO model.

    Args:
        return_idx (list): List of block indices to return as feature.
        resolution (tuple): Image size.
        use_summary_token (bool): Use summary token of backbone or not.
        **kwargs: Additional arguments for RADIO backbone.

    Return:
        model: RADIOAdapter model.
    """
    return RADIOAdapter(
        # RADIO.
        resolution=resolution,
        backbone="vit_large_patch16_reg4_dinov2",
        summary_idxs=[0, 1, 2],
        window_size=None,
        num_teacher=4,
        cpe_max_size=2048,
        register_multiple=8,
        drop_path_rate=0,
        # Adapter.
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        init_values=1e-5,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        cffn_ratio=0.25,
        deform_ratio=0.5,
        add_summary=use_summary_token,
        return_idx=return_idx,
        **kwargs
    )
