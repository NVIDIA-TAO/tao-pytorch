# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Four-corner bounds tracking through spatial transforms.

``Quad`` tracks the four corners of an image as 2D vertices.  Each
spatial transform (translate, scale, rotate, flip, arbitrary STM) is
applied to the vertices so that downstream code (e.g. ``equivariant_collate``)
can determine which region of the output canvas contains valid pixels.
"""

import torch


class Quad:
    """Axis-aligned quadrilateral that tracks image bounds.

    Initialised from an image tensor's spatial dimensions.  Methods
    mirror those on ``DeferImage`` so both can be transformed in lockstep.

    Attributes:
        bounds: ``(4, 2)`` float32 tensor of corner coordinates
            ``[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]``.
    """

    def __init__(self, image: torch.Tensor):
        self.bounds = torch.tensor([
            [0, 0],
            [image.shape[-1], 0],
            [image.shape[-1], image.shape[-2]],
            [0, image.shape[-2]],
        ], dtype=torch.float32)

    def apply_stm(self, stm: torch.Tensor, **kwargs):
        """Apply a 3x3 homogeneous transformation matrix to the bounds."""
        self.bounds = _apply_single_stm(self.bounds, stm)

    def translate(self, delta_vector: torch.Tensor):
        """Translate the quad bounds by ``delta_vector``."""
        self.bounds += delta_vector

    def scale(self, scale_vector: torch.Tensor, **kwargs):
        """Scale the quad bounds by ``scale_vector`` (per-axis)."""
        self.bounds *= scale_vector

    def rotate(self, rot_mat: torch.Tensor):
        """Rotate the quad bounds by the 2x2 matrix ``rot_mat``."""
        self.bounds = self.bounds @ rot_mat.T

    def flip(self, x: float):
        """Reflect the quad bounds horizontally about the line ``X = x``."""
        self.bounds[:, 0] -= x
        self.bounds[:, 0] *= -1
        self.bounds[:, 0] += x


def _apply_single_stm(vertices: torch.Tensor, stm: torch.Tensor):
    """Apply a 3x3 homogeneous transformation to 2D vertices.

    Args:
        vertices: ``(N, 2)`` tensor of 2D points.
        stm: ``(3, 3)`` homogeneous transformation matrix.

    Returns:
        ``(N, 2)`` transformed vertices (perspective-divided).
    """
    homogenous_vertices = torch.cat((vertices, torch.ones(vertices.shape[0], 1)), dim=1)
    transformed = torch.matmul(homogenous_vertices, stm)
    norm_factor = 1.0 / transformed[:, 2:]
    norm_factor[transformed[:, 2:] == 0] = 0
    return transformed[:, :2].contiguous() * norm_factor
