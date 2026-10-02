# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the stereo geometry pyramids built by CombinedGeoEncodingVolume.

The pyramids halve the last dim with sliced averaging instead of F.avg_pool2d (whose
TensorRT/cuDNN kernels cap the batch axis, here B * H/4 * W/4). These tests pin the
replacement to F.avg_pool2d(..., [1, 2], stride=[1, 2]) exactly, including odd widths.
"""

import pytest
import torch
import torch.nn.functional as F

from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.geometry import (
    CombinedGeoEncodingVolume,
    _halve_last_dim,
)


def _avg_pool_pyramid(level0, num_levels):
    """Reference pyramid: F.avg_pool2d with kernel [1, 2], stride [1, 2], applied iteratively."""
    pyramid = [level0]
    for _ in range(num_levels - 1):
        pyramid.append(F.avg_pool2d(pyramid[-1], [1, 2], stride=[1, 2]))
    return pyramid


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.model
@pytest.mark.parametrize("width", [2, 3, 4, 5, 8, 9, 48, 184, 185])
def test_halve_last_dim_matches_avg_pool2d(width):
    """_halve_last_dim equals avg_pool2d([1, 2], stride [1, 2]) for even and odd widths."""
    generator = torch.Generator().manual_seed(width)
    x = torch.randn(6, 3, 1, width, generator=generator)

    expected = F.avg_pool2d(x, [1, 2], stride=[1, 2])
    actual = _halve_last_dim(x)

    assert actual.shape == expected.shape == (6, 3, 1, width // 2)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.model
def test_halve_last_dim_rejects_width_one_like_avg_pool2d():
    """A width-1 last dim cannot be pooled by a size-2 kernel; both implementations raise."""
    x = torch.randn(4, 2, 1, 1)
    with pytest.raises(RuntimeError):
        F.avg_pool2d(x, [1, 2], stride=[1, 2])
    with pytest.raises(RuntimeError, match="need at least 2 elements"):
        _halve_last_dim(x)


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.model
@pytest.mark.parametrize("depth", [20, 11], ids=["D_even", "D_odd"])
@pytest.mark.parametrize("width2", [20, 13], ids=["W2_even", "W2_odd"])
def test_geo_encoding_pyramids_match_avg_pool2d(depth, width2):
    """Every level of both pyramids matches iterated avg_pool2d, in value and shape.

    With four levels, widths 20 -> 10 -> 5 -> 2, 11 -> 5 -> 2 -> 1 and 13 -> 6 -> 3 -> 1 cover
    pooling both even and odd widths, including an odd width produced by an earlier level.
    """
    num_levels = 4
    batch, feat_channels, geo_channels, height, width1 = 2, 16, 8, 3, 5
    generator = torch.Generator().manual_seed(depth * 100 + width2)
    fmap1 = torch.randn(batch, feat_channels, height, width1, generator=generator)
    fmap2 = torch.randn(batch, feat_channels, height, width2, generator=generator)
    geo_volume = torch.randn(batch, geo_channels, depth, height, width1, generator=generator)

    encoder = CombinedGeoEncodingVolume(fmap1, fmap2, geo_volume, num_levels=num_levels)

    rows = batch * height * width1
    geo_level0 = geo_volume.permute(0, 3, 4, 1, 2).reshape(rows, geo_channels, 1, depth)
    corr_level0 = CombinedGeoEncodingVolume.corr(fmap1, fmap2).reshape(rows, 1, 1, width2)
    expected_geo = _avg_pool_pyramid(geo_level0, num_levels)
    expected_corr = _avg_pool_pyramid(corr_level0, num_levels)

    assert len(encoder.geo_volume_pyramid) == num_levels
    assert len(encoder.init_corr_pyramid) == num_levels
    for level in range(num_levels):
        geo, corr = encoder.geo_volume_pyramid[level], encoder.init_corr_pyramid[level]
        assert geo.shape == (rows, geo_channels, 1, depth // 2 ** level), f"geo level {level}"
        assert corr.shape == (rows, 1, 1, width2 // 2 ** level), f"init_corr level {level}"
        assert geo.shape == expected_geo[level].shape
        assert corr.shape == expected_corr[level].shape
        torch.testing.assert_close(geo, expected_geo[level], atol=1e-6, rtol=0,
                                   msg=lambda m, lvl=level: f"geo level {lvl}: {m}")
        torch.testing.assert_close(corr, expected_corr[level], atol=1e-6, rtol=0,
                                   msg=lambda m, lvl=level: f"init_corr level {lvl}: {m}")
