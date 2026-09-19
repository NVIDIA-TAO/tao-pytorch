# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""VGGT module for the VGGT model."""

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from nvidia_tao_pytorch.cv.nvpanoptix3d.model.vggt.heads.camera_head import (
    CameraHead,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.vggt.heads.dpt_head import DPTHead

from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.vggt.models.aggregator import (
    Aggregator,
)


class VGGT(nn.Module, PyTorchModelHubMixin):
    """VGGT model

    Args:
        img_size (int): Input image size. Default is 518.
        patch_size (int): Patch size. Default is 14.
        embed_dim (int): Embedding dimension. Default is 1024.
        enable_camera (bool): Whether to enable the camera-pose head. Default is True.
        enable_point (bool): Whether to enable the world-point head. Default is True.
        enable_depth (bool): Whether to enable the depth head. Default is True.

    Note:
        Upstream VGGT also ships a point-tracking head. Neither NVPanoptix3Dv2
        variant tracks points, so that head is not vendored here and its weights
        are dropped when loading an upstream checkpoint.
    """

    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None

    def forward(self, images: torch.Tensor):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - dino_feats (torch.Tensor): DINOv2 patch features with shape [B, S, P, C]
                - vggt_feats (torch.Tensor): Final aggregator patch features with shape [B, S, P, 2C]
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)
        elif images.dim() != 5:
            raise ValueError(
                "images must be [S,3,H,W] or [B,S,3,H,W], got "
                f"shape {tuple(images.shape)}"
            )

        aggregated_tokens_list, patch_start_idx, dino_feats = self.aggregator(images)
        vggt_feats = aggregated_tokens_list[-1][:, :, patch_start_idx:]  # shape [B, S, P, 2C]

        predictions = {
            "dino_feats": dino_feats,
            "vggt_feats": vggt_feats,
        }

        with torch.amp.autocast("cuda", enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        return predictions
