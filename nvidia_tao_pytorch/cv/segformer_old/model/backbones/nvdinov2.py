# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


""" NVDINOv2 ViT Model Module """

from nvidia_tao_pytorch.core.utils.pos_embed_interpolation import (
    interpolate_pos_embed, interpolate_patch_embed
)
from nvidia_tao_pytorch.cv.segformer_old.model.backbones.vision_transformer.vit_adapter import (
    TIMMTransformerWrapper,
    ViTAdapter
)

import logging
import torch
import torch.nn as nn
from mmseg.registry import MODELS
from functools import partial

from timm.layers import SwiGLUPacked, PatchEmbed
from timm.models.vision_transformer import VisionTransformer

logger = logging.getLogger(__name__)


@MODELS.register_module()
class vit_large_nvdinov2(nn.Module):
    """ViT-Large NV-DINOv2 model."""

    def __init__(self, out_indices=[0, 1, 2, 3], resolution=(1024, 1024), init_cfg=None, **kwargs):
        """ViT-Large NV-DINOv2 model.

        Args:
            out_indices (list, optional): List of block indices to return as feature.. Defaults to [0, 1, 2, 3].
            resolution (tuple, optional): input resolution. Defaults to (1024, 1024).
            init_cfg (dict, optional): initial config. Defaults to None.
        """
        super().__init__()

        model_kwargs = dict(
            img_size=resolution[0],
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=5472 / 1024,
            drop_path_rate=0.4,
            mlp_layer=SwiGLUPacked,
            act_layer=nn.SiLU,
            init_values=1e-5,
            embed_layer=partial(PatchEmbed, strict_img_size=False),
            global_pool="",
            num_classes=0,
            reg_tokens=0
        )

        timm_vit_model = VisionTransformer(**model_kwargs)
        self.nvdinov2_vit_adapter = ViTAdapter(vit_model=TIMMTransformerWrapper(timm_vit_model),
                                               conv_inplane=56,
                                               n_points=4,
                                               drop_path_rate=0.4,
                                               deform_num_heads=16,
                                               init_values=1e-5,
                                               interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
                                               cffn_ratio=0.25,
                                               deform_ratio=0.5,
                                               out_indices=out_indices,
                                               resolution=resolution[0])

        # load pretrained backbone
        if init_cfg and init_cfg["checkpoint"]:
            pretrained_backbone_ckp = torch.load(init_cfg["checkpoint"],
                                                 map_location="cpu")
            # do vit interpolation
            pretrained_backbone_ckp = interpolate_vit_checkpoint(checkpoint=pretrained_backbone_ckp,
                                                                 target_patch_size=16,
                                                                 target_resolution=resolution[0])
            _tmp_st_output = self.nvdinov2_vit_adapter.vision_transformer.model.load_state_dict(pretrained_backbone_ckp, strict=False)

            logger.info(f"Loaded pretrained backbone weights from {init_cfg['checkpoint']}")
            logger.info(f"{_tmp_st_output}")

    def forward(self, x):
        """Forward function"""
        return self.nvdinov2_vit_adapter(x)


@MODELS.register_module()
class vit_giant_nvdinov2(nn.Module):
    """ViT-Giant NV-DINOv2 model.

    Args:
        out_indices (list): List of block indices to return as feature.
        activation_checkpoint (bool): flag to indicate if activation checkpoint is used.

    Return:
        model: ViT model.
    """

    def __init__(self, out_indices=[0, 1, 2, 3], resolution=(1024, 1024), init_cfg=None, **kwargs):
        """_summary_

        Args:
            out_indices (list, optional): List of block indices to return as feature.. Defaults to [0, 1, 2, 3].
            resolution (tuple, optional): input resolution. Defaults to (1024, 1024).
            init_cfg (dict, optional): initial config. Defaults to None.
        """
        super().__init__()

        # ingore reg_tokens for ViT-G because ViTAdapter would only take image patch and ignore
        # everything else, including cls_tokens and reg_tokens
        model_kwargs = dict(
            img_size=resolution[0],
            patch_size=14,
            embed_dim=1536,
            depth=40,
            num_heads=24,
            mlp_ratio=8192 / 1536,
            drop_path_rate=0.4,
            mlp_layer=SwiGLUPacked,
            act_layer=nn.SiLU,
            init_values=1e-5,
            embed_layer=partial(PatchEmbed, strict_img_size=False),
            global_pool="",
            num_classes=0,
            reg_tokens=0
        )

        timm_vit_model = VisionTransformer(**model_kwargs)
        self.nvdinov2_vit_adapter = ViTAdapter(vit_model=TIMMTransformerWrapper(timm_vit_model),
                                               conv_inplane=56,
                                               n_points=4,
                                               drop_path_rate=0.4,
                                               deform_num_heads=16,
                                               init_values=1e-5,
                                               interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
                                               cffn_ratio=0.25,
                                               deform_ratio=0.5,
                                               out_indices=out_indices,
                                               resolution=resolution[0])

        # load pretrained backbone
        if init_cfg and init_cfg["checkpoint"]:
            pretrained_backbone_ckp = torch.load(init_cfg["checkpoint"],
                                                 map_location="cpu")
            # do vit interpolation
            pretrained_backbone_ckp = interpolate_vit_checkpoint(checkpoint=pretrained_backbone_ckp,
                                                                 target_patch_size=14,
                                                                 target_resolution=resolution[0])
            _tmp_st_output = self.nvdinov2_vit_adapter.vision_transformer.model.load_state_dict(pretrained_backbone_ckp, strict=False)

            logger.info(f"Loaded pretrained backbone weights from {init_cfg['checkpoint']}")
            logger.info(f"{_tmp_st_output}")

    def forward(self, x):
        """Forward function"""
        return self.nvdinov2_vit_adapter(x)


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

    logger.info("Do ViT pretrained backbone interpolation")
    # interpolate patch embedding
    checkpoint = interpolate_patch_embed(checkpoint=checkpoint, new_patch_size=target_patch_size)

    # interpolate pos embedding
    checkpoint = interpolate_pos_embed(checkpoint_model=checkpoint,
                                       new_resolution=target_resolution,
                                       new_patch_size=target_patch_size)
    return checkpoint
