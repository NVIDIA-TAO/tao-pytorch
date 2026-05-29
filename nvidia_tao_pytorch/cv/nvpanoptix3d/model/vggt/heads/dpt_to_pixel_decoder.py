# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DPT head to pixel decoder module for the VGGT model.
This module is used to convert the aggregated tokens from the DPT head to pixel decoder.
The input is a list of aggregated tokens from the DPT head, and the output is a dictionary of multi-scale features.
"""

import torch
import torch.nn as nn
from typing import List, Dict
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.vggt.heads.utils import \
    create_uv_grid, position_grid_to_embed


class DPTHead2PixelDecoder(nn.Module):
    """
    DPT head to pixel decoder

    Args:
        dim_in (int): Input dimension.
        patch_size (int): Patch size. Default is 14.
        out_channels (List[int]): Output channels. Default is [256, 512, 1024, 2048].
        intermediate_layer_idx (List[int]): Intermediate layer indices. Default is [4, 11, 17, 23].
        pos_embed (bool): Whether to use positional embedding. Default is True.

    Returns:
        DPTHead2PixelDecoder: DPT head to pixel decoder module.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        out_channels: List[int] = [256, 512, 1024, 2048],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
    ) -> None:
        """DPT head to pixel decoder constructor"""
        super(DPTHead2PixelDecoder, self).__init__()
        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self.intermediate_layer_idx = intermediate_layer_idx

        self.norm = nn.LayerNorm(dim_in)

        # Projection from ViT dim -> stage channels (before unifying to out_channels)
        self.projects = nn.ModuleList(
            [nn.Conv2d(dim_in, dim_in, kernel_size=1) for _ in intermediate_layer_idx]
        )

        # Resize layers to match desired stride outputs
        # With patch_size=14 and strides=(4,8,16,32) we adjust via conv-transpose or pooling
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(dim_in, dim_in, kernel_size=4, stride=4, padding=0),  # stride 4
                nn.ConvTranspose2d(dim_in, dim_in, kernel_size=2, stride=2, padding=0),  # stride 8
                nn.Identity(),  # stride 16
                nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=2, padding=1),  # stride 32
            ]
        )

        self.unify_layers = nn.ModuleList(
            [nn.Conv2d(dim_in, out_channels[idx], kernel_size=1) for idx in range(len(intermediate_layer_idx))]
        )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            aggregated_tokens_list: list of transformer layer token outputs
            images: [B, S, 3, H, W]
            patch_start_idx: starting index for patch tokens in token sequence

        Returns:
            dict: {"res2": stride 4, "res3": stride 8, "res4": stride 16, "res5": stride 32}
        """
        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        multi_scale_feats = {}
        names = ["res2", "res3", "res4", "res5"]

        for dpt_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            # Extract patch tokens only
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            x = x.reshape(B * S, -1, x.shape[-1])
            x = self.norm(x)

            # Tokens -> [B*S, C, H_patch, W_patch]
            x = x.permute(0, 2, 1).reshape(
                (x.shape[0], x.shape[-1], patch_h, patch_w)
            )

            # Project
            x = self.projects[dpt_idx](x)

            # Optional positional embedding
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)

            # Resize to correct stride stage
            x = self.resize_layers[dpt_idx](x)

            # Unify to out_channels
            x = self.unify_layers[dpt_idx](x)

            # Save in dict
            multi_scale_feats[names[dpt_idx]] = x

        return multi_scale_feats

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to feature maps.

        Args:
            x (torch.Tensor): Input feature tensor with shape [B*S, C, H_patch, W_patch].
            W (int): Original image width before patching.
            H (int): Original image height before patching.
            ratio (float): Scaling factor for positional embedding strength. Default is 0.1.

        Returns:
            torch.Tensor: Feature tensor with added positional embedding, same shape as input.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed
