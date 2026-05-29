# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) Facebook, Inc. and its affiliates.
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
#
# Modified by: Shiyi Lan
# Mostly copy-paste from timm library.
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""CRADIO Transformer Backbone"""

from nvidia_tao_pytorch.cv.backbone.radio.base import RADIOBase
from nvidia_tao_pytorch.cv.backbone.radio.layers import ViTPatchGenerator

import torch
from torch import nn
from timm.models import register_model, create_model, checkpoint_seq
from timm.models.vision_transformer import (
    VisionTransformer,
    _create_vision_transformer,
    Mlp
)
from typing import Optional, Union, Tuple
from types import MethodType


@register_model
def vit_huge_patch16_224_mlpnorm(pretrained=False, **kwargs) -> VisionTransformer:
    """ViT-Huge model (ViT-H/16) from original paper (https://arxiv.org/abs/2010.11929).

    Args:
        pretrained (bool, optional): pretrained model. Defaults to False.

    Returns:
        VisionTransformer: Vision Transformer Model
    """
    model_args = dict(patch_size=16, embed_dim=1280, depth=32, num_heads=16)
    if pretrained:
        # There is no pretrained version of ViT-H/16, but we can adapt a ViT-H/14 for this purpose
        model = _create_vision_transformer('vit_huge_patch14_clip_336', pretrained=True, **dict(model_args, pre_norm=True, **kwargs))
    else:
        model = _create_vision_transformer('vit_huge_patch16_224', pretrained=False, **dict(model_args, **kwargs))

    for m in model.modules():
        if isinstance(m, Mlp) and not isinstance(m.norm, nn.LayerNorm):
            m.norm = nn.LayerNorm(m.fc1.out_features)

    return model


@register_model
def vit_huge_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ViT-Huge model (ViT-H/16) from original paper (https://arxiv.org/abs/2010.11929).

    Args:
        pretrained (bool, optional): pretrained model. Defaults to False.

    Returns:
        VisionTransformer: Vision Transformer Model
    """
    model_args = dict(patch_size=16, embed_dim=1280, depth=32, num_heads=16)
    if pretrained:
        # There is no pretrained version of ViT-H/16, but we can adapt a ViT-H/14 for this purpose
        model = _create_vision_transformer('vit_huge_patch14_clip_336', pretrained=True, **dict(model_args, pre_norm=True, **kwargs))
    else:
        model = _create_vision_transformer('vit_huge_patch16_224', pretrained=False, **dict(model_args, **kwargs))

    return model


class CRADIO(RADIOBase):
    """CRADIO model class
    """

    def __init__(
            self,
            backbone: str,
            summary_idxs: Optional[torch.Tensor] = None,
            window_size: int = None,
            in_chans: int = 3,
            num_teacher: int = 1,
            cpe_max_size: int = 2048,
            register_multiple: int = 16,
            **kwargs
    ):
        """CRADIO model class

        Args:
            backbone (str): backbone name
            summary_idxs (Optional[torch.Tensor], optional): sumary token index. Defaults to None.
            window_size (int, optional): Windown size for windowed attention. Defaults to None.
            in_chans (int, optional): input channels. Defaults to 3.
            num_teacher (int, optional): number of teachers. Defaults to 1.
            cpe_max_size (int, optional): cropped position embedding max size. Defaults to 2048.
            register_multiple (int, optional): number of extra tokens. Defaults to 16.
        """
        super().__init__()

        self.summary_idxs = summary_idxs
        self._window_size = window_size
        self.model_name = backbone
        self.model = self._build_model(backbone, in_chans, num_teacher,
                                       cpe_max_size, register_multiple)

    @property
    def num_summary_tokens(self) -> int:
        """Number of all extra tokens (class tokens + register tokens)
        """
        return self.model.patch_generator.num_skip

    @property
    def patch_size(self) -> int:
        """Patch size
        """
        return self.model.patch_generator.patch_size

    @property
    def window_size(self) -> int:
        """Window size for windowed attetion
        """
        return self._window_size

    def _build_model(self, model_name: str, in_chans: int = 3,
                     num_teacher: int = 4, cpe_max_size: int = 2048,
                     register_multiple: int = 16, **kwargs) -> nn.Module:
        """Method to build model

        Args:
            model_name (str): backbone model name
            in_chans (int, optional): input channels. Defaults to 3.
            num_teacher (int, optional): number of teachers. Defaults to 4.
            cpe_max_size (int, optional): cropped position embedding max size. Defaults to 2048.
            register_multiple (int, optional): number of extra tokens. Defaults to 16.

        Returns:
            nn.Module: RADIO model
        """
        model = create_model(
            model_name=model_name,
            pretrained=False,
            in_chans=in_chans,
            num_classes=0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            global_pool="token",
            weight_init="skip"
        )

        if hasattr(model, 'norm') and not getattr(kwargs, 'model_norm', False):
            model.norm = nn.Identity()

        model.head = nn.Identity()

        # enable cropped position embedding
        enable_cpe(model=model, max_img_size=cpe_max_size,
                   num_cls_tokens=num_teacher, register_multiple=register_multiple)

        return model

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """CRADIO model forward

        Args:
            x (torch.Tensor): Input features

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: output summary and features
        """
        y = self.model.forward_features(x)

        patch_gen = self.model.patch_generator
        all_summary = y[:, : patch_gen.num_cls_tokens]

        if self.summary_idxs is not None:
            bb_summary = all_summary[:, self.summary_idxs]
        else:
            bb_summary = all_summary

        bb_summary = bb_summary.flatten(1).to(torch.float32)

        all_feat = y[:, patch_gen.num_skip:]
        all_feat = all_feat.to(torch.float32)

        return bb_summary, all_feat


def _forward_cpe(self: VisionTransformer, x: torch.Tensor) -> torch.Tensor:
    """Internal function for CPE

    Args:
        self (VisionTransformer): ViT model
        x (torch.Tensor): input features

    Returns:
        torch.Tensor: output features
    """
    x = self.patch_generator(x)
    if self.grad_checkpointing and not torch.jit.is_scripting():
        x = checkpoint_seq(self.blocks, x)
    else:
        x = self.blocks(x)
    x = self.norm(x)
    return x


def enable_cpe(model: nn.Module,
               max_img_size: Union[int, Tuple[int, int]] = 1024,
               num_cls_tokens: int = 1,
               pos_dropout: float = 0.1,
               register_multiple: int = 0
               ):
    """Function to enable CPE

    Args:
        model (nn.Module): ViT model
        max_img_size (Union[int, Tuple[int, int]], optional): maximum image size. Defaults to 1024.
        num_cls_tokens (int, optional): number of class tokens. Defaults to 1.
        pos_dropout (float, optional): position embedding dropout rate. Defaults to 0.1.
        register_multiple (int, optional): number of extra tokens. Defaults to 0.

    Raises:
        ValueError: _description_
    """
    if not isinstance(model, VisionTransformer):
        raise ValueError("CPE only support for VisionTransformer models!")

    patch_size = model.patch_embed.patch_size[0]
    embed_dim = model.embed_dim
    input_dims = model.patch_embed.img_size
    normalize_patches = not isinstance(model.patch_embed.norm, nn.Identity)
    cls_token = model.cls_token is not None

    max_img_size = int(round(max_img_size / patch_size) * patch_size)

    patch_generator = ViTPatchGenerator(
        patch_size=patch_size,
        embed_dim=embed_dim,
        input_dims=input_dims,
        normalize_patches=normalize_patches,
        cls_token=cls_token,
        max_input_dims=max_img_size,
        pos_dropout=pos_dropout,
        num_cls_tokens=num_cls_tokens,
        register_multiple=register_multiple,
    )

    model.patch_generator = patch_generator
    model.patch_embed = None
    model.cls_token = None
    model.pos_embed = None
    model.pos_drop = None
    model.num_cls_tokens = num_cls_tokens
    model.num_registers = patch_generator.num_registers

    model.forward_features = MethodType(_forward_cpe, model)
