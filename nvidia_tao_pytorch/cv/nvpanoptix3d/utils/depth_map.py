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

"""Depth map utilities."""

import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from nvidia_tao_pytorch.cv.nvpanoptix3d.utils.io import write_pointcloud


class DepthMap(object):
    """Utility wrapper around a depth map and camera intrinsics.

    This class provides methods to:
    - Load/save depth maps
    - Apply masks
    - Convert depth to a pointcloud using the camera intrinsic matrix
    - Estimate per-pixel normals from the reconstructed pointcloud
    """

    def __init__(self, depth_map=None, intrinsic_matrix=None):
        """Create a DepthMap container.

        Args:
            depth_map: Either a torch tensor/NumPy array containing depth values, or
                a string path to a depth image on disk. If a path is provided, the
                image is loaded and converted to meters by dividing by 1000.0.
            intrinsic_matrix: Camera intrinsic matrix used for back-projection.
                Expected shape is ``(4, 4)`` (homogeneous) as used by this module.
        """
        if isinstance(depth_map, str):
            depth_map = torch.from_numpy(np.array(Image.open(depth_map))).float() / 1000.0
        self.depth_map = depth_map
        self.intrinsic_matrix = intrinsic_matrix

    def load_from(self, filename):
        """Load a depth map from an image file and store it in meters.

        Args:
            filename: Path to a depth image file. The loaded values are interpreted
                as millimeters and converted to meters by dividing by 1000.0.
        """
        depth_image = torch.from_numpy(np.array(Image.open(filename))).float()
        self.depth_map = depth_image / 1000.0

    def get_tensor(self):
        """Return a cloned torch tensor copy of the stored depth map."""
        return self.depth_map.clone()

    def set_intrinsic(self, intrinsic_matrix):
        """Set the camera intrinsic matrix used for back-projection.

        Args:
            intrinsic_matrix: Camera intrinsic matrix (expected ``(4, 4)``).
        """
        self.intrinsic_matrix = intrinsic_matrix

    def get_intrinsic(self):
        """Return a cloned copy of the camera intrinsic matrix."""
        return self.intrinsic_matrix.clone()

    def save(self, filename):
        """Save the depth map visualization to disk.

        Args:
            filename: Output path for the saved image.

        Notes:
            This writes a visualization using Matplotlib's ``rainbow`` colormap and
            does not preserve original depth encoding/units.
        """
        plt.imsave(filename, self.depth_map.numpy(), cmap="rainbow")

    def mask_out(self, mask):
        """Apply an elementwise mask to the depth map.

        Args:
            mask: Tensor/array broadcastable to the depth map shape. Typically a
                boolean or 0/1 mask.
        """
        self.depth_map = self.depth_map * mask

    def to_pointcloud(self, filename):
        """Write the depth-derived pointcloud to a PLY file (no colors).

        Args:
            filename: Output path for the pointcloud PLY.
        """
        pointcloud, _ = self.compute_pointcloud()
        write_pointcloud(pointcloud, None, filename)

    def to_pointcloud_with_colors(self, colors, filename):
        """Write the depth-derived pointcloud to a PLY file with per-point colors.

        Args:
            colors: Color image array/tensor indexable by pixel coordinates as
                ``colors[v, u]`` for each valid depth pixel.
            filename: Output path for the colored pointcloud PLY.
        """
        pointcloud, coords = self.compute_pointcloud()
        color_values = colors[coords[:, 0], coords[:, 1]]
        write_pointcloud(pointcloud, color_values, filename)

    def compute_pointcloud(self):
        """Back-project non-zero depth pixels into a 3D pointcloud.

        Returns:
            A tuple ``(pointcloud, coords2d)`` where:
            - ``coords2d``: ``(N, 2)`` integer pixel coordinates ``(v, u)`` for
              pixels where depth is non-zero.
            - ``pointcloud``: ``(N, 3)`` float XYZ points in camera coordinates.

        Notes:
            The implementation assumes a homogeneous 4x4 intrinsic matrix and uses:
            ``inverse(K) @ [u*z, v*z, z, 1]^T``.
        """
        coords2d = self.depth_map.nonzero(as_tuple=False)
        depth_map = self.depth_map[coords2d[:, 0], coords2d[:, 1]].reshape(-1)

        yv = coords2d[:, 0].reshape(-1).float() * depth_map.float()
        xv = coords2d[:, 1].reshape(-1).float() * depth_map.float()

        coords3d = torch.stack([xv, yv, depth_map.float(), torch.ones_like(depth_map).float()])
        pointcloud = torch.mm(
            torch.inverse(self.intrinsic_matrix.float()),
            coords3d.float()
        ).t()[:, :3]

        return pointcloud, coords2d

    def compute_normal(self):
        """Estimate per-pixel surface normals from the depth-derived pointcloud.

        Returns:
            A tensor of shape ``(3, H, W)`` containing estimated normals.

        Notes:
            Pixels with zero depth (and their immediate 4-neighbors) are zeroed out
            in the returned normal map.
        """
        # linearize
        width = self.depth_map.shape[1]
        height = self.depth_map.shape[0]
        depth_map = self.depth_map.reshape(-1).float()

        yv, xv = torch.meshgrid([torch.arange(height),
                                 torch.arange(width)])

        yv = yv.reshape(-1).float() * depth_map.float()
        xv = xv.reshape(-1).float() * depth_map.float()
        coords3d = torch.stack([xv, yv, depth_map.float(), torch.ones_like(depth_map).float()])
        pointcloud = torch.mm(
            torch.inverse(self.intrinsic_matrix.float()),
            coords3d.float()
        ).t()[:, :3]

        '''
           MC
        CM-CC-CP
           PC
        '''
        output_normals = torch.zeros((3, height, width))

        y, x = torch.meshgrid([torch.arange(1, height - 1),
                               torch.arange(1, width - 1)])
        y = y.reshape(-1)
        x = x.reshape(-1)

        # CC = pointcloud[(y + 0) * width + (x + 0)]
        PC = pointcloud[(y + 1) * width + (x + 0)]
        CP = pointcloud[(y + 0) * width + (x + 1)]
        MC = pointcloud[(y - 1) * width + (x + 0)]
        CM = pointcloud[(y + 0) * width + (x - 1)]

        n = torch.cross(PC - MC, CP - CM).transpose(1, 0)
        n_norm = torch.norm(n, dim=0)
        output_normals[:, y, x] = n / (-n_norm)

        # filter1: zero_depth and their neighbouring
        zeros = (self.depth_map == 0).nonzero()
        output_normals[:, zeros[:, 0], zeros[:, 1]] = 0

        zeros_height_lower = zeros.clone()
        zeros_height_lower[:, 0] -= 1
        zeros_height_lower[:, 0] = torch.clamp(zeros_height_lower[:, 0], min=0, max=height - 1)
        output_normals[:, zeros_height_lower[:, 0], zeros_height_lower[:, 1]] = 0

        zeros_height_upper = zeros.clone()
        zeros_height_upper[:, 0] += 1
        zeros_height_upper[:, 0] = torch.clamp(zeros_height_upper[:, 0], min=0, max=height - 1)
        output_normals[:, zeros_height_upper[:, 0], zeros_height_upper[:, 1]] = 0

        zeros_width_lower = zeros.clone()
        zeros_width_lower[:, 1] -= 1
        zeros_width_lower[:, 1] = torch.clamp(zeros_width_lower[:, 1], min=0, max=width - 1)
        output_normals[:, zeros_width_lower[:, 0], zeros_width_lower[:, 1]] = 0

        zeros_width_upper = zeros.clone()
        zeros_width_upper[:, 1] += 1
        zeros_width_upper[:, 1] = torch.clamp(zeros_width_upper[:, 1], min=0, max=width - 1)
        output_normals[:, zeros_width_upper[:, 0], zeros_width_upper[:, 1]] = 0

        return output_normals
