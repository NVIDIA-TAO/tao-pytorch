# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-PyTorch homography grid generation for spatial transforms.

Generates a sampling grid from a batch of 3x3 homography matrices.
The grid maps output pixels to normalised source coordinates and can be
fed directly to ``torch.nn.functional.grid_sample``.
"""

import torch


BASE_GRID_CACHE = dict()


def generate_homography_grid(homography: torch.Tensor, size):
    """Build a sampling grid for a batch of homography matrices.

    Args:
        homography: ``(N, 3, 3)`` batch of homography matrices.
        size: ``(N, C, H, W)`` tuple describing the output spatial size.

    Returns:
        ``(N, H, W, 2)`` float32 sampling grid in ``[-1, 1]`` normalised
        coordinates (perspective-divided).
    """
    N, _C, H, W = size
    if size not in BASE_GRID_CACHE:
        base_grid = homography.new(N, H, W, 3)
        linear_points = torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1])
        base_grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(base_grid[:, :, :, 0])
        linear_points = torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])
        base_grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(base_grid[:, :, :, 1])
        base_grid[:, :, :, 2] = 1
        BASE_GRID_CACHE[size] = base_grid
    else:
        base_grid = BASE_GRID_CACHE[size]

    grid = torch.bmm(base_grid.view(N, H * W, 3), homography.transpose(1, 2))
    grid = grid.view(N, H, W, 3)
    grid[:, :, :, 0] = grid[:, :, :, 0] / grid[:, :, :, 2]
    grid[:, :, :, 1] = grid[:, :, :, 1] / grid[:, :, :, 2]
    grid = grid[:, :, :, :2].float()
    return grid
