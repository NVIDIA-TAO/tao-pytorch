# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the stereo bilinear_sampler cuDNN grid_sample batch-limit chunking."""

import types

import pytest
import torch
import torch.nn.functional as F

from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo import utils
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.utils import (
    GRID_SAMPLE_MAX_ROWS,
    bilinear_sampler,
)


def _make_inputs(rows, channels=2, width=8, w_out=3, device="cpu", seed=0):
    """Build an (N, C, 1, W) volume and (N, 1, W_out, 2) pixel coords with constant y."""
    gen = torch.Generator().manual_seed(seed)
    img = torch.randn(rows, channels, 1, width, generator=gen).to(device)
    x = torch.rand(rows, 1, w_out, 1, generator=gen) * (width + 1) - 1  # some out of range
    coords = torch.cat([x, torch.zeros_like(x)], dim=-1).to(device)
    return img, coords


def _reference(img, coords):
    """Single direct grid_sample on the grid bilinear_sampler builds, plus its mask."""
    width = img.shape[-1]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (width - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1).to(img.dtype)
    mask = ((xgrid > -1) & (xgrid < 1) & (ygrid > -1) & (ygrid < 1)).float()
    return grid, mask


@pytest.fixture
def grid_sample_rows(monkeypatch):
    """Record the batch size (N) of every grid_sample call made by utils."""
    # Replace the module-level `F` that utils binds, not torch.nn.functional.grid_sample
    # itself (that would patch torch globally). bilinear_sampler only uses F.grid_sample.
    rows = []

    def _spy(inp, grid, *args, **kwargs):
        rows.append(inp.shape[0])
        return F.grid_sample(inp, grid, *args, **kwargs)

    monkeypatch.setattr(utils, "F", types.SimpleNamespace(grid_sample=_spy))
    return rows


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.model
@pytest.mark.parametrize("low_memory", [False, True])
@pytest.mark.parametrize("mask", [False, True])
@pytest.mark.parametrize("rows, expected", [
    (65536, [65535, 1]),
    (2 * 65535 + 10, [65535, 65535, 10]),
])
def test_eager_chunks_under_limit(grid_sample_rows, mask, low_memory, rows, expected):
    """Eager calls above the limit are chunked and bitwise equal to one direct call."""
    # Literal 65535: the bound must not drift up (65,536 rows already fails cuDNN for C>=41, W_out=9).
    assert GRID_SAMPLE_MAX_ROWS == 65535
    img, coords = _make_inputs(rows)
    grid, ref_mask = _reference(img, coords)
    ref = F.grid_sample(img, grid, align_corners=True)

    result = bilinear_sampler(img, coords, mask=mask, low_memory=low_memory)

    assert grid_sample_rows == expected
    out = result[0] if mask else result
    assert torch.equal(out, ref)
    if mask:
        assert torch.equal(result[1], ref_mask)


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.model
@pytest.mark.parametrize("low_memory", [False, True])
def test_eager_single_call_at_limit(grid_sample_rows, low_memory):
    """Exactly 65,535 rows still use one call."""
    img, coords = _make_inputs(65535)
    grid, _ = _reference(img, coords)

    out = bilinear_sampler(img, coords, low_memory=low_memory)

    assert grid_sample_rows == [65535]
    assert torch.equal(out, F.grid_sample(img, grid, align_corners=True))


class _Sampler(torch.nn.Module):
    """Wrap bilinear_sampler for torch.jit.trace."""

    def __init__(self, low_memory):
        super().__init__()
        self.low_memory = low_memory

    def forward(self, img, coords):
        """Sample img at coords."""
        return bilinear_sampler(img, coords, low_memory=self.low_memory)


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.model
@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
@pytest.mark.parametrize("low_memory, expected", [
    (False, [131080]),
    (True, [102400, 131080 - 102400]),
])
def test_traced_graph_not_chunked(grid_sample_rows, low_memory, expected):
    """Tracing keeps the original calls: one call, or the historical 102,400-row low_memory chunks."""
    img, coords = _make_inputs(131080)
    grid, _ = _reference(img, coords)

    traced = torch.jit.trace(_Sampler(low_memory), (img, coords), check_trace=False)

    assert grid_sample_rows == expected
    assert torch.equal(traced(img, coords), F.grid_sample(img, grid, align_corners=True))


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.model
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("channels, rows", [
    (28, 196606),  # FFS bp2 geo volume, one row above its measured cuDNN cap (196,605)
    (64, 65536),  # C>=64 shapes, one row above the 65,535 minimum cap
])
def test_cudnn_batch_limit_cuda(channels, rows):
    """A direct grid_sample one row above the cuDNN cap fails; bilinear_sampler does not."""
    if not (torch.backends.cudnn.is_available() and torch.backends.cudnn.enabled):
        pytest.skip("requires cuDNN")
    img, coords = _make_inputs(rows, channels=channels, width=48, w_out=9, device="cuda")
    grid, _ = _reference(img, coords)

    with pytest.raises(RuntimeError):
        F.grid_sample(img, grid, align_corners=True)

    half = rows // 2
    ref = torch.cat([
        F.grid_sample(img[:half], grid[:half], align_corners=True),
        F.grid_sample(img[half:], grid[half:], align_corners=True),
    ], dim=0)
    assert torch.equal(bilinear_sampler(img, coords), ref)
