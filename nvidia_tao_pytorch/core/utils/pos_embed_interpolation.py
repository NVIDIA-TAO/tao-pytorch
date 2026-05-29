# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""script to interpolate n x n ViT to m x m patches"""

import torch
from nvidia_tao_pytorch.core.tlt_logging import logging


def interpolate_pos_embed(checkpoint_model,
                          orig_resolution=None,
                          orig_patch_size=14,
                          new_resolution=None,
                          new_patch_size=16):
    """Interpolate Positional Embedding from ViT.

    Args:
        checkpoint_model (dict): ViT state_dict
        orig_resolution (int, optional): Original resolution. Defaults to None.
        orig_patch_size (int, optional): original patch size. Defaults to 14.
        new_resolution (int, optional): target patch size. Defaults to None.
        new_patch_size (int, optional): target patch size. Defaults to 16.

    Returns:
        dict: checkpoint_model with updated state dict
    """
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_extra_tokens = 1

        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)  # 16

        if orig_resolution:
            assert orig_resolution == orig_size * orig_patch_size, \
                f"Resolution {orig_resolution} and patch size {orig_patch_size} do not match"
        else:
            orig_resolution = orig_size * orig_patch_size

        # height (== width) for the new position embedding
        if new_resolution:
            new_size = new_resolution // new_patch_size
        else:
            new_size = orig_resolution // new_patch_size

        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]

        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens.float(), size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        logging.info(f"Resolution: {orig_resolution}")
        logging.info(f"Old Pos Embed: {pos_embed_checkpoint.shape}")
        logging.info(f"New Pos Embed: {new_pos_embed.shape}")

    return checkpoint_model


def interpolate_patch_embed(checkpoint, new_patch_size=16):
    """Interpolate patch_embed.

    Args:
        checkpoint (dict): ViT state_dict
        new_patch_size (int, optional): new patch size. Defaults to 16.

    Returns:
        dict: checkpoint with updated state dict
    """
    patch_embed = checkpoint['patch_embed.proj.weight']
    patch_embed = torch.nn.functional.interpolate(
        patch_embed.float(), size=(new_patch_size, new_patch_size), mode='bicubic', align_corners=False)
    checkpoint['patch_embed.proj.weight'] = patch_embed
    return checkpoint
