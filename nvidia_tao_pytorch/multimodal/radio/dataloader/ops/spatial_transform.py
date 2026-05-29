# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Perspective warp using 3x3 homography matrices.

Uses a C++ CPU/CUDA kernel (ported from EVFM) when available; falls back
to a pure-PyTorch path using F.grid_sample otherwise.

The C++ kernel dispatches internally:
  - tensor.is_cuda() -> CUDA kernel (8x8 thread blocks)
  - else             -> CPU kernel (OpenMP parallelized)
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_cpp_spatial_transform = None
try:
    from nvidia_tao_pytorch.multimodal.radio.dataloader.ops.SpatialTransformOps import (
        spatial_transform as _fn,
    )
    _cpp_spatial_transform = _fn
except ImportError:
    logger.info("SpatialTransformOps C++ extension not built; using Python fallback.")


def spatial_transform_fallback(
    image: torch.Tensor,
    stm: torch.Tensor,
    out_width: int,
    out_height: int,
    mode: str = "bilinear",
    pad_color: float = 0.0,
    verbose: bool = False,
) -> torch.Tensor:
    """Pure-PyTorch replacement for the C++ ``spatial_transform`` kernel.

    Matches the C++ signature: ``(inputs, stms, output_width, output_height,
    method, background, verbose)``.  The *verbose* flag is accepted for
    compatibility but ignored.

    The C++ kernel operates in pixel coordinates with a half-pixel center
    convention::

        For output pixel (x, y):
          p = [x + 0.5, y + 0.5, 1]
          [mx, my, mz] = p @ M          (M = stm, row-vector @ matrix)
          src_x = mx / mz - 0.5
          src_y = my / mz - 0.5
          bilinear sample at (src_x, src_y)

    This function reproduces that coordinate mapping, then delegates the
    actual sampling to ``F.grid_sample`` (with ``align_corners=True``).

    Args:
        image: ``(N, C, H, W)`` input batch.
        stm: ``(N, 3, 3)`` homography matrices (already column-major / transposed).
        out_width: desired output width.
        out_height: desired output height.
        mode: interpolation mode (``'bilinear'`` or ``'nearest'``).
        pad_color: value to fill out-of-bounds pixels.  ``F.grid_sample``
            only natively supports 0; for non-zero values we post-fill.
        verbose: accepted for C++ signature compatibility; ignored.

    Returns:
        ``(N, C, out_height, out_width)`` transformed batch.
    """
    N, _C, H_in, W_in = image.shape
    out_h = int(out_height)
    out_w = int(out_width)

    if stm.dim() == 2:
        stm = stm.unsqueeze(0).expand(N, -1, -1)

    y = torch.arange(out_h, dtype=torch.float32, device=image.device) + 0.5
    x = torch.arange(out_w, dtype=torch.float32, device=image.device) + 0.5
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    ones = torch.ones_like(gx)
    pixel_grid = torch.stack([gx, gy, ones], dim=-1)
    pixel_grid = pixel_grid.reshape(1, out_h * out_w, 3).expand(N, -1, -1)

    transformed = torch.bmm(pixel_grid, stm)
    src_x = transformed[..., 0] / transformed[..., 2] - 0.5
    src_y = transformed[..., 1] / transformed[..., 2] - 0.5

    if W_in > 1:
        norm_x = 2.0 * src_x / (W_in - 1) - 1.0
    else:
        norm_x = torch.zeros_like(src_x)
    if H_in > 1:
        norm_y = 2.0 * src_y / (H_in - 1) - 1.0
    else:
        norm_y = torch.zeros_like(src_y)

    sample_grid = torch.stack([norm_x, norm_y], dim=-1).reshape(N, out_h, out_w, 2)

    result = F.grid_sample(
        image, sample_grid, mode=mode, padding_mode="zeros", align_corners=True
    )

    if pad_color != 0.0:
        oob_x = (src_x < -0.5) | (src_x >= W_in - 0.5)
        oob_y = (src_y < -0.5) | (src_y >= H_in - 0.5)
        oob = (oob_x | oob_y).reshape(N, 1, out_h, out_w).expand_as(result)
        result[oob] = pad_color

    return result


def spatial_transform(
    inputs: torch.Tensor,
    stms: torch.Tensor,
    output_width: int,
    output_height: int,
    method: str = "bilinear",
    background: float = 0.0,
    verbose: bool = False,
) -> torch.Tensor:
    """Apply perspective warp to a batch of images using 3x3 homography matrices.

    Uses the C++ kernel if available, otherwise falls back to
    ``spatial_transform_fallback`` (pure PyTorch / F.grid_sample).

    Args:
        inputs: Input images [B, C, H, W].
        stms: Spatial transform matrices [B, 3, 3].
        output_width: Output width.
        output_height: Output height.
        method: Interpolation method ("nearest", "bilinear", "bicubic").
        background: Background fill value for out-of-bounds pixels.
        verbose: Print debug info.

    Returns:
        Transformed images [B, C, output_height, output_width].
    """
    if _cpp_spatial_transform is not None:
        return _cpp_spatial_transform(
            inputs, stms, output_width, output_height, method, background, verbose
        )

    return spatial_transform_fallback(
        inputs, stms, output_width, output_height,
        mode=method, pad_color=background, verbose=verbose,
    )
