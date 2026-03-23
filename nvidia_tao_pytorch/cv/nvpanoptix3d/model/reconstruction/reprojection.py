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

"""Sparse projection for NVPanoptix3D model using WarpConvNet."""

import torch
from torch import nn
from torch.nn import functional as F
from warpconvnet.geometry.types.voxels import Voxels

from nvidia_tao_pytorch.cv.mask2former.utils.point_features import point_sample
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.reconstruction.frustum import (
    generate_frustum,
    compute_camera2frustum_transform,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d.utils.sparse_utils import sparse_collate


class SparseProjection(nn.Module):
    """NVPanoptix3D model version of the sparse projection module."""

    def __init__(
        self, cfg, truncation: float = 3.0,
        sign_channel: bool = True,
        depth_min: float = 0.4, depth_max: float = 6.0,
        voxel_size: float = 0.03, frustum_dims: int = 256
    ):
        """Initialize the SparseProjection module."""
        super().__init__()
        self.truncation = truncation
        self.sign_channel = sign_channel
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.voxel_size = voxel_size
        self.register_buffer(
            "frustum_dimensions",
            torch.tensor([frustum_dims, frustum_dims, frustum_dims]),
            persistent=False,
        )

    @property
    def device(self):
        """Get the device of the SparseProjection module."""
        return self.frustum_dimensions.device

    @staticmethod
    def to_voxels(features, coordinates, stride=1):
        """Convert features and coordinates to Voxels."""
        batched_coords, batched_feats = sparse_collate(coordinates, features)
        spatial_coords = batched_coords[:, 1:].int()
        offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=spatial_coords.device),
                torch.bincount(batched_coords[:, 0].int()).cumsum(dim=0).to(torch.int64),
            ),
            dim=0,
        )
        return Voxels(
            batched_coordinates=spatial_coords,
            batched_features=batched_feats,
            offsets=offsets,
            tensor_stride=stride,
        )

    @staticmethod
    def projection(
        frustum, voxel_size, frustum_dimensions,
        truncation, intrinsic_inverse, depth,
        image_size, feat_size, near_clip, far_clip
    ):
        """
        Args:
            frustum: Frustum.
            voxel_size: Voxel size.
            frustum_dimensions: Frustum dimensions.
            truncation: Truncation.
            intrinsic_inverse: Inverse intrinsic matrix.
            depth: Depth.
            image_size: Image size.
            feat_size: Feature size.
            near_clip: Near clip.
            far_clip: Far clip.
        Returns:
            num repetition: number of repetition.
            segm sampling grid: segmentation sampling grid.
            feat sampling grid: feature sampling grid.
            flatten coordinates: flatten coordinates.
            coordinates z: coordinates z.
            voxel offsets: voxel offsets.
        """
        camera2frustum, padding_offsets = compute_camera2frustum_transform(
            frustum, voxel_size,
            frustum_dimensions=frustum_dimensions
        )

        depth = depth.clone()
        depth[depth < near_clip] = 0
        depth[depth > far_clip] = 0
        depth_pixels_xy = depth.nonzero(as_tuple=False)
        device = depth_pixels_xy.device

        if depth_pixels_xy.shape[0] == 0:
            depth_pixels_xy = torch.tensor(
                [[depth.shape[0] // 2, depth.shape[1] // 2]], device=device
            )
        depth_pixels_z = depth[depth_pixels_xy[:, 0], depth_pixels_xy[:, 1]].reshape(-1).float()

        depth_pixels_xy = depth_pixels_xy.flip(-1).float()
        normalized_depth_pixels_xy = depth_pixels_xy / torch.tensor(
            [depth.shape[-1], depth.shape[-2]], device=device
        )
        xv, yv = (normalized_depth_pixels_xy * torch.tensor(
            image_size, device=device) * depth_pixels_z[:, None]
        ).unbind(-1)
        # Use separate size for feature maps due to size divisibility padding
        feat_sampling_grid = depth_pixels_xy / torch.tensor(feat_size, device=device)

        depth_pixels = torch.stack([xv, yv, depth_pixels_z, torch.ones_like(depth_pixels_z)])
        pointcloud = torch.mm(intrinsic_inverse.float(), depth_pixels.float())
        grid_coordinates = torch.mm(camera2frustum.float(), pointcloud).t()[:, :3].contiguous()

        # projective sdf encoding
        # repeat truncation, add / subtract z-offset
        num_repetition = int(truncation * 2) + 1
        grid_coordinates = grid_coordinates.unsqueeze(1).repeat(1, num_repetition, 1)
        voxel_offsets = torch.arange(-truncation, truncation + 1, 1.0, device=device).view(1, -1, 1)
        coordinates_z = grid_coordinates[:, :, 2].clone()
        grid_coordinates[:, :, 2] += voxel_offsets[:, :, 0]

        num_points = grid_coordinates.size(0)

        flatten_coordinates = grid_coordinates.view(num_points * num_repetition, 3)
        # pad to 256,256,256
        flatten_coordinates = flatten_coordinates + padding_offsets
        return num_repetition, normalized_depth_pixels_xy, \
            feat_sampling_grid, flatten_coordinates, coordinates_z, voxel_offsets

    def forward(self, multi_scale_features, encoder_features, batched_inputs):
        """Forward pass
        Args:
            multi_scale_features: List of multi-scale features.
            encoder_features: List of encoder features.
            batched_inputs: List of batch inputs.
        Returns:
            sparse_ms_features: List of sparse multi-scale features.
            sparse_enc_features: List of sparse encoder features.
        """
        sparse_ms_coordinates = [[] for _ in range(len(multi_scale_features))]
        sparse_ms_features = [[] for _ in range(len(multi_scale_features))]
        sparse_enc_features = []
        sparse_enc_coordinates = []

        # Process each sample in the batch individually
        for idx, inputs in enumerate(batched_inputs):
            # Get GT intrinsic matrix
            intrinsic = inputs["intrinsic"].to(self.device)
            image_size = inputs["image_size"]  # (width, height)
            padded_size = inputs["padded_size"]
            intrinsic_inverse = torch.inverse(intrinsic)

            frustum = generate_frustum(
                image_size,
                intrinsic_inverse,
                self.depth_min,
                self.depth_max
            )

            num_repetition, segm_sampling_grid, feat_sampling_grid, \
                flatten_coordinates, coordinates_z, voxel_offsets = \
                self.projection(
                    frustum, self.voxel_size,
                    self.frustum_dimensions, self.truncation,
                    intrinsic_inverse, inputs["depth"],
                    image_size, (padded_size[0] // 2, padded_size[1] // 2),
                    self.depth_min, self.depth_max
                )

            df_values = coordinates_z - coordinates_z.int()
            df_values = df_values + voxel_offsets.squeeze(-1)
            df_values.unsqueeze_(-1)

            # encode sign and values in 2 different channels
            if self.sign_channel:
                sign = torch.sign(df_values)
                value = torch.abs(df_values)
                df_values = torch.cat([sign, value], dim=-1)

            # segm features
            semantic_seg = inputs["sem_seg"]
            sampled_segm_features = point_sample(
                semantic_seg[None], segm_sampling_grid[None], align_corners=False
            )[0]

            # encoder features
            sampled_enc_features = point_sample(
                encoder_features[[idx]],
                feat_sampling_grid[None],
                align_corners=False,
            )[0]
            sampled_enc_features = torch.cat([sampled_enc_features, sampled_segm_features], dim=0)
            sampled_enc_features = sampled_enc_features.permute(1, 0).unsqueeze(1).repeat(
                1, num_repetition, 1
            )
            sampled_enc_features = torch.cat([df_values, sampled_enc_features], dim=-1)

            flat_features = sampled_enc_features.flatten(0, -2)
            sparse_enc_coordinates.append(flatten_coordinates)
            sparse_enc_features.append(flat_features)

            # multi-scale features
            for lvl, feat in enumerate(multi_scale_features):
                ratio = feat.shape[-1] / encoder_features.shape[-1]
                level_depth = F.interpolate(
                    inputs["depth"][None, None], scale_factor=ratio, mode="nearest"
                ).squeeze()
                num_repetition, segm_sampling_grid, feat_sampling_grid, flatten_coordinates, *__ = \
                    self.projection(
                        frustum, self.voxel_size / ratio, self.frustum_dimensions * ratio,
                        round(ratio * self.truncation), intrinsic_inverse, level_depth,
                        image_size, (feat.shape[-1] * 2, feat.shape[-2] * 2),
                        self.depth_min, self.depth_max,
                    )
                sampled_features = point_sample(
                    feat[[idx]],
                    feat_sampling_grid[None],
                    align_corners=False,
                )[0]
                sampled_features = \
                    sampled_features.permute(1, 0).unsqueeze(1).repeat(1, num_repetition, 1).flatten(0, -2)
                sparse_ms_features[lvl].append(sampled_features)
                # Resize feature volume
                sparse_ms_coordinates[lvl].append(flatten_coordinates.clone())

        sparse_enc_features = self.to_voxels(sparse_enc_features, sparse_enc_coordinates)
        strides = [2, 4, 8]
        sparse_ms_features = [
            self.to_voxels(feats, coords, stride=stride)
            for feats, coords, stride in zip(
                sparse_ms_features, sparse_ms_coordinates, strides,
            )
        ]
        return sparse_ms_features, sparse_enc_features
