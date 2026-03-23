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

""" Frustum utilities, mostly using numpy. """

import math
import torch
import numpy as np
from typing import Tuple


def frustum2planes(frustum: np.ndarray) -> dict:
    """Convert frustum to planes.
    Args:
        frustum: Frustum.
    Returns:
        Planes.
    """
    planes = {}
    # normal towards inside
    # near
    edge_vec_1 = frustum[3] - frustum[0]
    edge_vec_2 = frustum[1] - frustum[0]
    plane_normal = np.cross(edge_vec_1, edge_vec_2)
    plane_offset = -np.dot(plane_normal, frustum[0])
    planes["near"] = np.array([plane_normal[0], plane_normal[1], plane_normal[2], plane_offset])

    # far
    edge_vec_1 = frustum[5] - frustum[4]
    edge_vec_2 = frustum[7] - frustum[4]
    plane_normal = np.cross(edge_vec_1, edge_vec_2)
    plane_offset = -np.dot(plane_normal, frustum[4])
    planes["far"] = np.array([plane_normal[0], plane_normal[1], plane_normal[2], plane_offset])

    # left
    edge_vec_1 = frustum[5] - frustum[1]
    edge_vec_2 = frustum[0] - frustum[1]
    plane_normal = np.cross(edge_vec_1, edge_vec_2)
    plane_offset = -np.dot(plane_normal, frustum[1])
    planes["left"] = np.array([plane_normal[0], plane_normal[1], plane_normal[2], plane_offset])

    # right
    edge_vec_1 = frustum[3] - frustum[2]
    edge_vec_2 = frustum[6] - frustum[2]
    plane_normal = np.cross(edge_vec_1, edge_vec_2)
    plane_offset = -np.dot(plane_normal, frustum[2])
    planes["right"] = np.array([plane_normal[0], plane_normal[1], plane_normal[2], plane_offset])

    # top
    edge_vec_1 = frustum[4] - frustum[0]
    edge_vec_2 = frustum[3] - frustum[0]
    plane_normal = np.cross(edge_vec_1, edge_vec_2)
    plane_offset = -np.dot(plane_normal, frustum[0])
    planes["top"] = np.array([plane_normal[0], plane_normal[1], plane_normal[2], plane_offset])

    # bottom
    edge_vec_1 = frustum[2] - frustum[1]
    edge_vec_2 = frustum[5] - frustum[1]
    plane_normal = np.cross(edge_vec_1, edge_vec_2)
    plane_offset = -np.dot(plane_normal, frustum[1])
    planes["bottom"] = np.array([plane_normal[0], plane_normal[1], plane_normal[2], plane_offset])

    return planes


def frustum_culling(points: np.ndarray, frustum: np.ndarray) -> np.ndarray:
    """Cull points outside frustum.
    Args:
        points: Points.
        frustum: Frustum.
    Returns:
        Points inside frustum.
    """
    frustum_planes = frustum2planes(frustum)
    points = np.concatenate([points, np.ones((len(points), 1))], 1)
    flags = np.ones(len(points))
    for _, plane in frustum_planes.items():
        flag = np.dot(points, plane) >= 0
        flags = np.logical_and(flags, flag)

    return points[flags][:, :3]


def frustum_transform(frustum: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform frustum.
    Args:
        frustum: Frustum.
        transform: Transform matrix.
    Returns:
        Transformed frustum.
    """
    eight_points = np.concatenate([frustum, np.ones((8, 1))], 1).transpose()
    frustum = np.dot(transform, eight_points).transpose()
    return frustum[:, :3]


def generate_frustum(
    image_size: Tuple, intrinsic_inv: np.ndarray,
    depth_min: float, depth_max: float,
    transform: np.ndarray = None
) -> np.ndarray:
    """Generate frustum.
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

    eight_points = np.array([[0.0, 0.0, depth_min, 1.0],
                             [0.0, y * depth_min, depth_min, 1.0],
                             [x * depth_min, y * depth_min, depth_min, 1.0],
                             [x * depth_min, 0.0, depth_min, 1.0],
                             [0.0, 0.0, depth_max, 1.0],
                             [0.0, y * depth_max, depth_max, 1.0],
                             [x * depth_max, y * depth_max, depth_max, 1.0],
                             [x * depth_max, 0.0, depth_max, 1.0]]).transpose()

    frustum = np.dot(intrinsic_inv, eight_points)

    if transform is not None:
        frustum = np.dot(transform, frustum)

    frustum = frustum.transpose()

    return frustum[:, :3]


def generate_frustum_volume(frustum: np.ndarray, voxel_size: float) -> Tuple:
    """Generate frustum volume.
    Args:
        frustum: Frustum.
        voxel_size: Voxel size.
    Returns:
        Frustum volume.
        Camera-to-frustum transform.
    """
    max_x = np.max(frustum[:, 0]) / voxel_size
    max_y = np.max(frustum[:, 1]) / voxel_size
    max_z = np.max(frustum[:, 2]) / voxel_size
    min_x = np.min(frustum[:, 0]) / voxel_size
    min_y = np.min(frustum[:, 1]) / voxel_size
    min_z = np.min(frustum[:, 2]) / voxel_size

    dim_x = math.ceil(max_x - min_x)
    dim_y = math.ceil(max_y - min_y)
    dim_z = math.ceil(max_z - min_z)

    camera2frustum = np.array([[1.0 / voxel_size, 0, 0, -min_x],
                               [0, 1.0 / voxel_size, 0, -min_y],
                               [0, 0, 1.0 / voxel_size, -min_z],
                               [0, 0, 0, 1.0]])

    return (dim_x, dim_y, dim_z), camera2frustum


def compute_camera2frustum_transform(
    intrinsic: torch.Tensor, image_size: Tuple,
    depth_min: float, depth_max: float,
    voxel_size: float
) -> torch.Tensor:
    """Compute camera-to-frustum transform.
    Args:
        intrinsic: Intrinsic matrix.
        image_size: Image size.
        depth_min: Minimum depth.
        depth_max: Maximum depth.
        voxel_size: Voxel size.
    Returns:
        Camera-to-frustum transform.
    """
    frustum = generate_frustum(image_size, torch.inverse(intrinsic).numpy(), depth_min, depth_max)
    _, camera2frustum = generate_frustum_volume(frustum, voxel_size)
    camera2frustum = torch.from_numpy(camera2frustum).float()

    return camera2frustum
