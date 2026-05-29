# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transformer (ViT and FAN) builder."""

from functools import partial

import torch.nn as nn

from nvidia_tao_pytorch.cv.backbone_v2.fan import (
    fan_tiny_12_p16_224,
    fan_small_12_p16_224,
    fan_base_18_p16_224,
    fan_large_24_p16_224,
    fan_tiny_8_p4_hybrid,
    fan_small_12_p4_hybrid,
    fan_base_16_p4_hybrid,
    fan_large_16_p4_hybrid
)
from nvidia_tao_pytorch.cv.backbone_v2.vit import VisionTransformer


fan_dict = {
    "fan_tiny_12_p16_224": fan_tiny_12_p16_224,
    "fan_small_12_p16_224": fan_small_12_p16_224,
    "fan_base_18_p16_224": fan_base_18_p16_224,
    "fan_large_24_p16_224": fan_large_24_p16_224,
    "fan_tiny_8_p4_hybrid": fan_tiny_8_p4_hybrid,
    "fan_small_12_p4_hybrid": fan_small_12_p4_hybrid,
    "fan_base_16_p4_hybrid": fan_base_16_p4_hybrid,
    "fan_large_16_p4_hybrid": fan_large_16_p4_hybrid
}
# urls_dic = {
#     "vit-deit-tiny/16": "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
#     "vit-deit-small/16": "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
#     "vit-deit-base/16": "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
#     "vit-deit-base-distilled/16":  "https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
#     "vit-deit-iii-base-224/16": "https://dl.fbaipublicfiles.com/deit/deit_3_base_224_21k.pth",
#     "vit-mocov3-base/16": "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar",
#     "vit-mae-base/16": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
#     'vit-mae-large/16': "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth",
#     'vit-mae-huge/14': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth'
# }


def get_vit(cfg=None):
    """Build ViT models.

    Args:
        cfg (OmegaConfig): Hydra config

    Return:
        model: ViT model
    """
    arch = cfg.model.arch
    if '16' in arch:
        patch_size = 16
    elif '8' in arch:
        patch_size = 8
    elif '14' in arch:
        patch_size = 14
    else:
        raise ValueError("Only 8/14/16 are supported.")

    if 'tiny' in arch.lower():
        embed_dim = 192
        num_heads = 3
        depth = 12
    if 'small' in arch.lower():
        embed_dim = 384
        num_heads = 6
        depth = 12
    elif 'base' in arch.lower():
        embed_dim = 768
        num_heads = 12
        depth = 12
    elif 'large' in arch.lower():
        embed_dim = 1024
        num_heads = 16
        depth = 24
    elif 'huge' in arch.lower():
        embed_dim = 1280
        num_heads = 16
        depth = 32
    else:
        raise ValueError("Only tiny/small/base/large/huge are supported.")

    model = VisionTransformer(
        img_size=224, dynamic_img_size=True,
        patch_size=patch_size, embed_dim=embed_dim, depth=depth,
        num_heads=num_heads, mlp_ratio=4, qkv_bias=True, drop_path_rate=cfg.model.vit_dpr,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), freeze_at=cfg.model.frozen_stages)

    return model


def get_fan(cfg):
    """Build FAN models.

    Args:
        cfg (OmegaConfig): Hydra config

    Return:
        model: FAN model
    """
    arch = cfg.model.arch
    if arch in list(fan_dict.keys()):
        return fan_dict[arch]()
    raise ValueError(f"Only {list(fan_dict.keys())} are supported.")


def build_model(cfg):
    """Model builder.

    Args:
        cfg (OmegaConfig): Hydra config
    Return:
        backbone: either ViT or FAN model
    """
    if 'vit' in cfg.model.arch:
        backbone = get_vit(cfg)
    elif 'fan' in cfg.model.arch:
        backbone = get_fan(cfg)
    else:
        raise ValueError('Only vit and fan are supported.')
    return backbone
