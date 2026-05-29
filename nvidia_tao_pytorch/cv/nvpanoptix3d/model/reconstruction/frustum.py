# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frustum generation for NVPanoptix-3D model."""

from typing import Optional
import torch


def generate_frustum(image_size, intrinsic_inv, depth_min, depth_max, transform=None):
    """
    Args:
        image_size: Image size.
        intrinsic_inv: Inverse intrinsic matrix.
        depth_min: Minimum depth.
        depth_max: Maximum depth.
        transform: Transform matrix.
    Returns:
        Frustum.
    """
    x = image_size[0]
    y = image_size[1]

    eight_points = torch.tensor([[0 * depth_min, 0 * depth_min, depth_min, 1.0],
                                 [0 * depth_min, y * depth_min, depth_min, 1.0],
                                 [x * depth_min, y * depth_min, depth_min, 1.0],
                                 [x * depth_min, 0 * depth_min, depth_min, 1.0],
                                 [0 * depth_max, 0 * depth_max, depth_max, 1.0],
                                 [0 * depth_max, y * depth_max, depth_max, 1.0],
                                 [x * depth_max, y * depth_max, depth_max, 1.0],
                                 [x * depth_max, 0 * depth_max, depth_max, 1.0]],
                                device=intrinsic_inv.device, dtype=intrinsic_inv.dtype)

    frustum = intrinsic_inv @ eight_points.T

    if transform is not None:
        frustum = transform @ frustum

    frustum = frustum.T

    return frustum[:, :3]


def generate_frustum_volume(frustum, voxel_size):
    """
    Args:
        frustum: Frustum.
        voxel_size: Voxel size.
    Returns:
        Frustum volume.
    """
    max_x = torch.max(frustum[:, 0]) / voxel_size
    max_y = torch.max(frustum[:, 1]) / voxel_size
    max_z = torch.max(frustum[:, 2]) / voxel_size
    min_x = torch.min(frustum[:, 0]) / voxel_size
    min_y = torch.min(frustum[:, 1]) / voxel_size
    min_z = torch.min(frustum[:, 2]) / voxel_size

    dim_x = torch.ceil(max_x - min_x)
    dim_y = torch.ceil(max_y - min_y)
    dim_z = torch.ceil(max_z - min_z)

    camera2frustum = torch.as_tensor([[1.0 / voxel_size, 0, 0, -min_x],
                                      [0, 1.0 / voxel_size, 0, -min_y],
                                      [0, 0, 1.0 / voxel_size, -min_z],
                                      [0, 0, 0, 1.0]], dtype=frustum.dtype, device=frustum.device)

    return torch.stack((dim_x, dim_y, dim_z)), camera2frustum


def compute_camera2frustum_transform(
        frustum,
        voxel_size: float,
        frustum_dimensions: Optional[torch.Tensor] = None
):
    """
    Args:
        frustum: Frustum.
        voxel_size: Voxel size.
        frustum_dimensions: Frustum dimensions.
    Returns:
        Camera to frustum transform.
    """
    dimensions, camera2frustum = generate_frustum_volume(frustum, voxel_size)
    if frustum_dimensions is not None:
        difference = (frustum_dimensions - dimensions).float()
        padding_offsets = torch.div(difference, 2, rounding_mode="floor")
        return camera2frustum, padding_offsets
    else:
        return camera2frustum
