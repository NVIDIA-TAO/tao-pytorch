# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOV2 backbone for Segformer."""

import math
from functools import partial

import torch
import torch.nn.functional as F
from timm.layers import PatchEmbed, SwiGLUPacked, trunc_normal_
from torch import nn
from torch.nn.init import normal_

from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.utils.pos_embed_interpolation import interpolate_patch_embed, interpolate_pos_embed
from nvidia_tao_pytorch.cv.backbone_v2.dino_v2 import DINOV2
from nvidia_tao_pytorch.cv.deformable_detr.model.ops.modules import MSDeformAttn
from nvidia_tao_pytorch.cv.segformer.model.backbones.adapter_modules import (
    InteractionBlock,
    SpatialPriorModule,
    deform_inputs,
)


def interpolate_vit_checkpoint(checkpoint, target_patch_size, target_resolution):
    """Interpolate ViT backbone position embedding and patch embedding

    Args:
        checkpoint (dict): pretrained ViT checkpoint
        target_patch_size (int): target patch size to interpolate to. ex: 14, 16, etc
        target_resolution (int): target image size to interpolate to. ex: 224, 512, 518, etc

    Returns:
        dict: interpolated model checkpoints
    """
    if checkpoint is None:
        return checkpoint
    if get_global_rank() == 0:
        logging.info("Do ViT pretrained backbone interpolation")
    checkpoint = interpolate_patch_embed(checkpoint=checkpoint, new_patch_size=target_patch_size)
    checkpoint = interpolate_pos_embed(
        checkpoint_model=checkpoint, new_resolution=target_resolution, new_patch_size=target_patch_size
    )
    return checkpoint


class DINOV2Adapter(DINOV2):
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
        drop_path_rate=0.0,
        add_vit_feature=True,
        use_extra_extractor=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        return_idx=[0, 1, 2, 3],
        **kwargs,
    ):
        """Initialize DINOV2 adapter.

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
            return_idx (list): List of block indices to return as feature.
            **kwargs: Additional arguments for DINOV2 backbone.
        """
        img_size = kwargs.get("img_size", 224)
        patch_size = kwargs.get("patch_size", 16)
        assert img_size % 32 == 0, f"Input img_size ({img_size}) should be divisible by 32."
        kwargs["img_size"] = img_size
        super().__init__(drop_path_rate=drop_path_rate, **kwargs)

        self.cls_token = None
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.img_size = img_size
        self.patch_size = patch_size

        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dim))
        normal_(self.level_embed)
        self.spm = SpatialPriorModule(
            in_channel=3,
            patch_size=self.patch_size,
            inplanes=conv_inplane,
            embed_dim=self.embed_dim,
            out_indices=return_idx,
        )
        self.spm.apply(self._init_weights)
        self.interactions = nn.Sequential(
            *[
                InteractionBlock(
                    dim=self.embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer,
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
        self.up = nn.ConvTranspose2d(self.embed_dim, self.embed_dim, 2, 2)
        self.up.apply(self._init_weights)
        self.norm1 = nn.BatchNorm2d(self.embed_dim)
        self.norm2 = nn.BatchNorm2d(self.embed_dim)
        self.norm3 = nn.BatchNorm2d(self.embed_dim)
        self.norm4 = nn.BatchNorm2d(self.embed_dim)
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
        for m in modules:
            for p in m.parameters():
                p.requires_grad = True
            m.train()

    def load_state_dict(self, state_dict, **kwargs):
        """Copy parameters and buffers from state_dict into this module and its descendants.

        Args:
            state_dict (dict): a dict containing parameters and persistent buffers.
            **kwargs: Additional arguments passed to `nn.Module.load_state_dict`.
        """
        state_dict = interpolate_vit_checkpoint(
            state_dict, target_patch_size=self.patch_size, target_resolution=self.img_size
        )
        return super().load_state_dict(state_dict, **kwargs)

    def _get_pos_embed(self, pos_embed: torch.Tensor, h: int, w: int):
        """Interpolate position embedding.

        Args:
            pos_embed (torch.Tensor): original position embedding
            h (int): input height
            w (int): input width

        Returns:
            torch.Tensor: interpolated position embedding
        """
        grid_size = self.patch_embed.grid_size
        if grid_size == (h, w):
            # no need to interpolate pos embedding if number of input patches is equal to model's num_patches
            return pos_embed

        # [1, seq, embed_dim] -> [1, h, w, embed_dim] -> [1, embed_dim, h, w]
        pos_embed = pos_embed.reshape(1, grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2)

        # [1, embed_dim, h, w] -> [1, embed_dim, H, W] -> [1, embed_dim, seq*] -> [1, seq*, embed_dim]
        pos_embed = (
            F.interpolate(pos_embed, size=(h, w), mode="bicubic", align_corners=False)
            .reshape(1, -1, h * w)
            .permute(0, 2, 1)
        )
        return pos_embed

    def get_patch_embed(self, x: torch.Tensor):
        """Get patch embedding from ViT.

        Args:
            x (torch.Tensor): input feature

        Returns:
            tuple: tuple of feature, input height and input width
        """
        # patches, patched_feature_H, patched_feature_W
        x = self.patch_embed(x)  # [B, S, embed_dim]
        h = w = int(x.shape[1] ** 0.5)

        # get rid of cls_token (prefix_token), only consider patches
        # reg_tokens is not consider here because default reg_tokens is 0 for TIMM VIT
        pos_embed = self._get_pos_embed(self.pos_embed[:, self.num_prefix_tokens:], h, w)
        x = self.pos_drop(x + pos_embed)
        return x, h, w

    def get_vit_blocks(self, start_idx: int, end_idx: int):
        """Get transformer blocks from ViT.

        Args:
            start_idx (int): block start index
            end_idx (int): block end index

        Returns:
            nn.Module: target ViT Transformer blocks
        """
        return self.blocks[start_idx:end_idx]

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward_feature_pyramid(self, x: torch.Tensor):
        """Forward pass through the backbone to extract intermediate feature maps."""
        deform_inputs1, deform_inputs2 = deform_inputs(x, patch_size=self.patch_size)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        _, _, H, W = x.shape

        # Downsampling in SPM ResNet backbone: 4, 8, 16, 32 -> c1, c2, c3, c4
        # c3 have downsample 4 times in ResNet backbone, hence divided by 16
        H, W = H // 16, W // 16

        x, patch_h, patch_w = self.get_patch_embed(x)
        bs, _, dim = x.shape

        # Interaction
        outs = []
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(
                x,
                c,
                self.get_vit_blocks(indexes[0], indexes[-1] + 1),
                deform_inputs1,
                deform_inputs2,
                H,
                W,
                batch_first=True,  # Atth input format [bs, seq_l, dim].
            )
            outs.append(x.transpose(1, 2).view(bs, dim, patch_h, patch_w).contiguous())

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
            x1 = F.interpolate(x1, size=c1.shape[-2:], mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, size=c2.shape[-2:], mode="bilinear", align_corners=False)
            x3 = F.interpolate(x3, size=c3.shape[-2:], mode="bilinear", align_corners=False)
            x4 = F.interpolate(x4, size=c4.shape[-2:], mode="bilinear", align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]


def vit_large_nvdinov2(return_idx=[0, 1, 2, 3], resolution=1024, **kwargs):
    """DINOV2 ViT Large model with SwiGLU activation."""
    return DINOV2Adapter(
        # DINOV2.
        img_size=resolution,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_layer=SwiGLUPacked,
        act_layer=nn.SiLU,
        mlp_ratio=5472 / 1024,
        drop_path_rate=0.4,
        embed_layer=partial(PatchEmbed, strict_img_size=False),
        global_pool="",
        num_classes=0,
        reg_tokens=0,
        # DINOV2Adapter.
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        init_values=1e-5,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        cffn_ratio=0.25,
        deform_ratio=0.5,
        return_idx=return_idx,
        **kwargs,
    )


def vit_giant_nvdinov2(return_idx=[0, 1, 2, 3], resolution=1024, **kwargs):
    """DINOV2 ViT Giant model with SwiGLU activation."""
    # Set `reg_tokens=0` for ViT-G because DINOV2Adapter would only take image patch and ignore
    # everything else, including cls_tokens` and `reg_tokens`.
    return DINOV2Adapter(
        # DINOV2.
        img_size=resolution,
        patch_size=14,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_layer=SwiGLUPacked,
        act_layer=nn.SiLU,
        mlp_ratio=8192 / 1536,
        drop_path_rate=0.4,
        embed_layer=partial(PatchEmbed, strict_img_size=False),
        global_pool="",
        num_classes=0,
        reg_tokens=0,
        # DINOV2Adapter.
        conv_inplane=56,
        n_points=4,
        deform_num_heads=16,
        init_values=1e-5,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        cffn_ratio=0.25,
        deform_ratio=0.5,
        return_idx=return_idx,
        **kwargs,
    )
