# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ViT Adapter backbone."""

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_
from timm.layers import trunc_normal_
from timm.layers.trace_utils import _assert
from functools import partial

from nvidia_tao_pytorch.cv.deformable_detr.model.ops.modules import MSDeformAttn
from nvidia_tao_pytorch.cv.segformer_old.model.backbones.vision_transformer.adapter_modules import (
    SpatialPriorModule, InteractionBlock, deform_inputs
)

logger = logging.getLogger(__name__)


class TIMMTransformerWrapper(nn.Module):
    """TIMM ViT model wrapper."""

    def __init__(self, model):
        """TIMM ViT model wrapper.

        Args:
            model (nn.Module): ViT model with TIMM backend.
        """
        super().__init__()
        self.model = model
        self.patch_size = self.model.patch_embed.patch_size[0]
        self.embed_dim = self.model.embed_dim
        self.grid_size = self.model.patch_embed.grid_size
        self.batch_first = True  # attn input format [bs, seq_l, dim]

    def get_patch_embed(self, x):
        """Method to get patch embedding from ViT

        Args:
            x (torch.Tensor): input feature

        Returns:
            tuple: tuple of feature, input height and input width
        """
        # patches, patched_feature_H, patched_feature_W
        x = self.model.patch_embed(x)  # [B, S, embed_dim]
        H = W = int(x.shape[1] ** 0.5)

        # get rid of cls_token (prefix_token), only consider patches
        # reg_tokens is not consider here because default reg_tokens is 0 for TIMM VIT
        pos_embed = self._get_pos_embed(self.model.pos_embed[:, self.model.num_prefix_tokens:], H, W)
        x = self.model.pos_drop(x + pos_embed)

        return x, H, W

    def get_vit_blocks(self, start_idx, end_idx):
        """Method to get transformer blocks from ViT

        Args:
            start_idx (int): block start index
            end_idx (int): block end index

        Returns:
            nn.Module: target ViT Transformer blocks
        """
        return self.model.blocks[start_idx:end_idx]

    def _get_pos_embed(self, pos_embed, H, W):
        """method to interpolate position embedding

        Args:
            pos_embed (torch.Tensor): original position embedding
            H (int): input height
            W (int): input width

        Returns:
            torch.Tensor: interpolated position embedding
        """
        if self.grid_size == (H, W):
            # no need to interpolate pos embedding if number of input patches is equal to model's num_patches
            return pos_embed

        # [1, seq, embed_dim] -> [1, h, w, embed_dim] -> [1, embed_dim, h, w]
        pos_embed = pos_embed.reshape(1, self.grid_size[0], self.grid_size[1], -1).permute(0, 3, 1, 2)

        # [1, embed_dim, h, w] -> [1, embed_dim, H, W] -> [1, embed_dim, seq*] -> [1, seq*, embed_dim]
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed


class OpenCLIPTransformerWrapper(nn.Module):
    """OpenCLIP ViT model wrapper."""

    def __init__(self, model):
        """OpenCLIP ViT model wrapper.

        Args:
            model (nn.Module): ViT model with OpenCLIP backend.
        """
        super().__init__()
        self.model = model
        self.patch_size = self.model.patch_size[0]
        self.embed_dim = self.model.transformer.width
        self.grid_size = self.model.grid_size

        # OpenCLIP use nn.MultiheadAttention for attn and by default batch_first is False
        # ref: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        self.batch_first = False  # attn input format [seq_l, bs, dim]

    def get_patch_embed(self, x):
        """Method to get patch embedding from ViT

        Args:
            x (torch.Tensor): input feature

        Returns:
            tuple: tuple of feature, input height and input width
        """
        # avoid weird things happen, we need to check if input size is divisible by patch_size
        _, _, h, w = x.shape

        _assert(h % self.patch_size == 0,
                f"Input height ({h}) should be divisible by patch size ({self.patch_size}).")
        _assert(w % self.patch_size == 0,
                f"Input width ({w}) should be divisible by patch size ({self.patch_size}).")

        # patches, patched_feature_H, patched_feature_W
        x = self.model.conv1(x)
        H, W = x.shape[-2:]

        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [bs, dim, seq_l] -> [bs, seq_l, dim]

        # get rid of cls_token (prefix_token), only consider patches
        # reg_tokens is not consider here because OpenCLIP ViT doesn't contain reg_tokens
        pos_embed = self._get_pos_embed(self.model.positional_embedding[1:, :], H, W)
        x = self.model.patch_dropout(x + pos_embed)
        x = self.model.ln_pre(x)

        return x, H, W

    def get_vit_blocks(self, start_idx, end_idx):
        """Method to get transformer blocks from ViT

        Args:
            start_idx (int): block start index
            end_idx (int): block end index

        Returns:
            nn.Module: target ViT Transformer blocks
        """
        return self.model.transformer.resblocks[start_idx:end_idx]

    def _get_pos_embed(self, pos_embed, H, W):
        """method to interpolate position embedding

        Args:
            pos_embed (torch.Tensor): original position embedding
            H (int): input height
            W (int): input width

        Returns:
            torch.Tensor: interpolated position embedding
        """
        if self.grid_size == (H, W):
            # no need to interpolate pos embedding if number of input patches is equal to model's num_patches
            return pos_embed

        # [1, seq, embed_dim] -> [1, h, w, embed_dim] -> [1, embed_dim, h, w]
        pos_embed = pos_embed.reshape(1, self.grid_size[0], self.grid_size[1], -1).permute(0, 3, 1, 2)

        # [1, embed_dim, h, w] -> [1, embed_dim, H, W] -> [1, embed_dim, seq*] -> [1, seq*, embed_dim]
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed


class ViTAdapter(nn.Module):
    """ViT-Adapter from https://arxiv.org/abs/2205.08534."""

    def __init__(self, vit_model, conv_inplane=64, n_points=4, drop_path_rate=0.,
                 deform_num_heads=6, init_values=0., interaction_indexes=None, with_cffn=True,
                 cffn_ratio=0.25, deform_ratio=1.0, add_vit_feature=True,
                 use_extra_extractor=True, out_indices=[0, 1, 2, 3], activation_checkpoint=False,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), resolution=512, **kwargs):
        """ViT-Adapter Constructor.

        Args:
            vit_model (nn.Module): Vision Transformer model
            conv_inplane (int): The hidden dimension of Conv2D in SPM.
            n_points (int): The number of sampling points for
                each query in each head of MultiScaleDeformableAttention.
            drop_path_rate (float) stochastic depth rate. Defaults to 0.
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
            norm_layer: (nn.Module): normalization layer
            resolution: (int): input image resolution
        """
        super().__init__()

        # we're using ResNet stem and multiple convs for SPM
        # make sure we have valid resolution_assert(h % self.patch_size == 0,
        _assert(resolution % 32 == 0, f"Input resolution ({resolution}) should be divisible by 32.")

        self.vision_transformer = vit_model
        self.cls_token = None
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature

        self.patch_size = self.vision_transformer.patch_size
        self.embed_dim = self.vision_transformer.embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dim))
        self.spm = SpatialPriorModule(in_channel=3,
                                      patch_size=self.patch_size,
                                      inplanes=conv_inplane,
                                      embed_dim=self.embed_dim,
                                      out_indices=out_indices)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=self.embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=drop_path_rate,
                             norm_layer=norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((i == len(interaction_indexes) - 1) and use_extra_extractor),
                             with_cp=activation_checkpoint)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(self.embed_dim, self.embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(self.embed_dim)
        self.norm2 = nn.SyncBatchNorm(self.embed_dim)
        self.norm3 = nn.SyncBatchNorm(self.embed_dim)
        self.norm4 = nn.SyncBatchNorm(self.embed_dim)

        self.up.apply(self._init_weights)
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

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        """Forward function."""
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

        x, patch_h, patch_w = self.vision_transformer.get_patch_embed(x)
        bs, _, dim = x.shape

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.vision_transformer.get_vit_blocks(indexes[0], indexes[-1] + 1),
                         deform_inputs1, deform_inputs2, H, W, self.vision_transformer.batch_first)
            outs.append(x.transpose(1, 2).view(bs, dim, patch_h, patch_w).contiguous())

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
            x1 = F.interpolate(x1, size=c1.shape[-2:], mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, size=c2.shape[-2:], mode='bilinear', align_corners=False)
            x3 = F.interpolate(x3, size=c3.shape[-2:], mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, size=c4.shape[-2:], mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]
