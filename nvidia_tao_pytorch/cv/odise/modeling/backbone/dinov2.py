# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from functools import partial

import torch
from timm.layers import PatchEmbed, SwiGLUPacked
from timm.models.vision_transformer import (
    VisionTransformer,
    build_model_with_cfg,
    checkpoint_filter_fn,
    register_model,
)
from torch import nn


class DinoV2VisionTransformer(VisionTransformer):
    """
    This class extends the VisionTransformer class from timm library so that we can
    handle different image sizes.
    """

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8

        w0, h0 = w0 + 0.1, h0 + 0.1

        # We need fp32 for the interpolation
        reshaped_pos_embed = patch_pos_embed.reshape(
            1, int(math.sqrt(N)), int(math.sqrt(N)), dim
        ).permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            reshaped_pos_embed,
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def _pos_embed(self, x):
        B, S, _ = x.shape
        w = h = int(math.sqrt(S)) * self.patch_embed.patch_size[0]

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)


@register_model
def vit_large_patch14_dinov2_swiglu(pretrained=False, **kwargs) -> VisionTransformer:
    """ViT-L/14 for DINOv2"""

    assert pretrained is False, (
        "You should load the checkpoint by using checkpoint_path argument "
        + "or manually load the checkpoint using torch.load()"
    )

    model_args = dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        init_values=1e-5,
        img_size=224,
        mlp_layer=SwiGLUPacked,
        act_layer=nn.SiLU,
        mlp_ratio=5472 / 1024,
        embed_layer=partial(PatchEmbed, strict_img_size=False),
    )

    model = build_model_with_cfg(
        DinoV2VisionTransformer,
        "vit_large_patch14_dinov2_swiglu",
        pretrained=False,
        pretrained_filter_fn=checkpoint_filter_fn,
        **dict(model_args, **kwargs),
    )

    return model
