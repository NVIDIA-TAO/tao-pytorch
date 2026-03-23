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
# Portions of this code are based on the BUOL model:
# https://github.com/chtsy/buol

"""Back projection module."""

import torch
from torch import nn
from nvidia_tao_pytorch.cv.nvpanoptix3d.utils.frustum import (
    generate_frustum,
    generate_frustum_volume,
    compute_camera2frustum_transform
)


class BackProjection(nn.Module):
    """
    Back projection module for projecting 3D frustum voxels to 2D image coordinates.

    This module computes the mapping between 3D voxel coordinates in a frustum volume
    and their corresponding 2D pixel coordinates in the image plane. It handles camera
    intrinsics, depth ranges, and frustum mask filtering.
    """

    def __init__(self, cfg):
        """Initialize the back projection module.

        Args:
            cfg: Configuration object containing:
                - dataset.depth_size: Image dimensions for depth estimation
                - dataset.depth_min: Minimum depth value in meters
                - dataset.depth_max: Maximum depth value in meters
                - model.projection.voxel_size: Size of each voxel in meters
                - model.frustum3d.frustum_dims: Dimensions of the frustum volume
        """
        super().__init__()
        self.image_size = (cfg.dataset.depth_size[1], cfg.dataset.depth_size[0])
        # self.image_size = cfg.dataset.depth_size
        self.depth_min = cfg.dataset.depth_min
        self.depth_max = cfg.dataset.depth_max
        self.voxel_size = cfg.model.projection.voxel_size
        self.frustum_dimensions = torch.tensor([cfg.model.frustum3d.frustum_dims] * 3)

    def forward(
        self, shape, intrinsics, frustum_masks=None, room_masks=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project 3D frustum voxels to 2D image coordinates and compute valid mappings.

        Args:
            shape: Target image shape as (height, width)
            intrinsics: Camera intrinsic matrices
                - Shape: (N, 4, 4) for batch mode or (4, 4) for single view
            frustum_masks: Binary masks indicating valid frustum voxels (optional)
                - Shape: (N, D, D, D) for batch mode or (D, D, D) for single view
                - where D is frustum_dimensions (e.g., 256)
                - If None, assumes all voxels are valid
            room_masks: Binary masks for room segmentation (optional)
                - Shape: (N, 1, H, W)

        Returns:
            tuple containing:
                - kepts: Binary masks indicating valid voxels after back-projection
                    - Shape: (N, D, D, D) for batch or (D, D, D) for single view
                - mappings: Mapping from 3D voxel coords to 2D pixel coords and depth
                    - Shape: (N, D, D, D, 5) for batch or (D, D, D, 4) for single view
                    - Last dimension: [batch_idx, u, v, depth] (batch) or [u, v, depth] (single)
        """
        device = intrinsics.device
        if frustum_masks is None:
            frustum_masks = torch.ones(
                [len(intrinsics), *self.frustum_dimensions],
                dtype=torch.bool, device=device
            )
        len_shape = len(frustum_masks.shape)
        if len_shape == 3:
            frustum_masks = frustum_masks[None]
            intrinsics = intrinsics[None]

        kepts, mappings = [], []
        for bi, (intrinsic, frustum_mask) in enumerate(zip(intrinsics, frustum_masks)):
            camera2frustum = compute_camera2frustum_transform(
                intrinsic.cpu(), self.image_size, self.depth_min,
                self.depth_max, self.voxel_size
            ).to(device)
            intrinsic_inverse = torch.inverse(intrinsic)
            coordinates = torch.nonzero(frustum_mask)
            grid_coordinates = coordinates.clone()
            # NOTE: do not flip XY voxel indices; flipping mirrors the projection and misaligns 3D lifting.
            # grid_coordinates[:, :2] = 256 - grid_coordinates[:, :2]

            padding_offsets = self.compute_frustum_padding(intrinsic_inverse)
            grid_coordinates = grid_coordinates - padding_offsets - torch.tensor([1., 1., 1.], device=device)
            grid_coordinates = torch.cat([
                grid_coordinates, torch.ones(len(grid_coordinates), 1, device=device)], 1
            )
            pointcloud = torch.mm(torch.inverse(camera2frustum), grid_coordinates.t())
            depth_pixels = torch.mm(intrinsic, pointcloud)

            depth = depth_pixels[2]
            coord = depth_pixels[0:2] / depth
            coord = torch.cat([coord, coordinates[:, 2][None]], 0).permute(1, 0)

            kept = (depth <= self.depth_max) * \
                   (depth >= self.depth_min) * \
                   (coord[:, 0] < shape[1]) * (coord[:, 1] < shape[0])
            coordinates = coordinates[kept]
            depth = depth[kept, None]

            # Mapping tensor uses the configured frustum voxel grid resolution (D x D x D).
            # D is derived from cfg.model.frustum3d.frustum_dims (typically 256), not depth.
            frustum_dim = int(self.frustum_dimensions[0].item())
            mapping = torch.zeros(
                frustum_dim, frustum_dim, frustum_dim, 5,
                device=depth.device, dtype=depth.dtype
            ) - 1.
            mapping[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = \
                torch.cat([torch.ones_like(depth) * bi, coord[kept], depth], -1)

            kept = (mapping >= 0).all(-1)

            if room_masks is not None:
                mapping_kept = mapping[kept].long()
                kept[kept.clone()] = room_masks[bi, 0, mapping_kept[:, 2], mapping_kept[:, 1]]

            kepts.append(kept)
            mappings.append(mapping)

        if len_shape == 3:
            kepts = kepts[0]
            mappings = mappings[0][..., 1:]
        else:
            kepts = torch.stack(kepts, 0)
            mappings = torch.stack(mappings, 0)

        return kepts, mappings

    def compute_frustum_padding(self, intrinsic_inverse: torch.Tensor) -> torch.Tensor:
        """Compute padding offsets to center the frustum volume.

        Calculates the difference between the target frustum dimensions and the
        actual frustum volume dimensions, then computes symmetric padding offsets
        to center the frustum within the voxel grid.

        Args:
            intrinsic_inverse: Inverse of camera intrinsic matrix
                - Shape: (4, 4)

        Returns:
            padding_offsets: Padding values for each dimension (x, y, z)
                - Shape: (3,)
        """
        frustum = generate_frustum(
            self.image_size, intrinsic_inverse.cpu().numpy(), self.depth_min, self.depth_max
        )
        dimensions, _ = generate_frustum_volume(frustum, self.voxel_size)
        difference = (
            self.frustum_dimensions - torch.tensor(dimensions)
        ).float().to(intrinsic_inverse.device)
        padding_offsets = difference / 2
        return padding_offsets
