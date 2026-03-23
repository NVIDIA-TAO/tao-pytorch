# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
# Portions of this code are based on the VGGT project by Facebook Research (Meta):
# https://github.com/facebookresearch/vggt

"""VGGT module for the VGGT model."""

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from nvidia_tao_pytorch.cv.nvpanoptix3d.model.vggt.models.aggregator import Aggregator
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.vggt.heads.camera_head import CameraHead
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.vggt.heads.dpt_head import DPTHead
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.vggt.heads.dpt_to_pixel_decoder import DPTHead2PixelDecoder

from addict import Dict


class VGGT(nn.Module, PyTorchModelHubMixin):
    """VGGT model

    Args:
        img_size (int): Input image size. Default is 518.
        patch_size (int): Patch size. Default is 14.
        embed_dim (int): Embedding dimension. Default is 1024.
        enable_camera (bool): Whether to enable camera head. Default is True.
        enable_depth (bool): Whether to enable depth head. Default is True.
    """

    def __init__(
        self, img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=True, enable_depth=True
    ) -> None:
        """VGGT constructor"""
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.depth_head = DPTHead(
            dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1"
        ) if enable_depth else None

        self._out_features = ["res2", "res3", "res4", "res5"]
        self.num_inter_features = [256, 512, 1024, 2048]
        self.dpt2pixeldecoder = DPTHead2PixelDecoder(dim_in=2 * embed_dim, out_channels=self.num_inter_features)

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.num_inter_features[0],
            "res3": self.num_inter_features[1],
            "res4": self.num_inter_features[2],
            "res5": self.num_inter_features[3],
        }

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2]
                (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.no_grad():
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

        if self.dpt2pixeldecoder is not None:
            predictions["multi_scale_features"] = self.dpt2pixeldecoder(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )

        if self.depth_head is not None:
            depth, depth_conf, multi_scale_depth_features = self.depth_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx,
                is_depth=True
            )
            predictions["depth"] = depth[..., 0]
            predictions["depth_conf"] = depth_conf
            predictions["multi_scale_depth_features"] = multi_scale_depth_features

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

    def freeze_modules(self):
        """Freeze modules"""
        # freeze the Aggregator:
        for param in self.aggregator.parameters():
            param.requires_grad = False

        # freeze the CameraHead:
        if self.camera_head is not None:
            for param in self.camera_head.parameters():
                param.requires_grad = False

        # unfreeze the DepthHead:
        if self.depth_head is not None:
            for param in self.depth_head.parameters():
                param.requires_grad = True

        # unfreeze the DPT2PixelDecoder:
        if self.dpt2pixeldecoder is not None:
            for param in self.dpt2pixeldecoder.parameters():
                param.requires_grad = True

    def output_shape(self):
        """Get output feature shape."""
        backbone_feature_shape = dict()
        for name in self._out_features:
            backbone_feature_shape[name] = Dict({
                'channel': self._out_feature_channels[name], 'stride': self._out_feature_strides[name]
            })
        return backbone_feature_shape
