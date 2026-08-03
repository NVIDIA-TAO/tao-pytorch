# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 Phase 1 (high-res Gram) unit tests — refresh cadence, grid pooling, config (CPU)."""
import pytest
import torch
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.dinov3.model.pl_model import DinoV3PlModel


@pytest.mark.config
@pytest.mark.ssl_unit
def test_gram_phase1_config_fields():
    """Gram config exposes the Phase 1 knobs with sensible defaults."""
    cfg = OmegaConf.structured(ExperimentConfig())
    assert cfg.model.gram.refresh_interval == 0      # never refresh -> Phase 0 behavior
    assert cfg.model.gram.teacher_scale == 1.0       # memory-safe default (paper 2.0; spec may raise)
    assert cfg.model.gram.teacher_source == "pretrained"


@pytest.mark.ssl_unit
def test_gram_refresh_due():
    """EMA-refresh fires only at positive multiples of the interval, once per step."""
    due = DinoV3PlModel._gram_refresh_due
    assert due(0, 100, -1) is False          # never at step 0
    assert due(100, 100, -1) is True         # first multiple
    assert due(100, 100, 100) is False       # already refreshed at 100
    assert due(150, 100, 100) is False       # not a multiple
    assert due(200, 100, 100) is True        # next multiple
    assert due(100, 0, -1) is False          # interval 0 disables
    assert due(100, -1, -1) is False         # guard against negative


@pytest.mark.ssl_unit
def test_pool_tokens_identity_when_grids_match():
    """Pooling to the same grid is a no-op."""
    x = torch.randn(2, 16, 8)
    out = DinoV3PlModel._pool_tokens(x, (4, 4), (4, 4))
    assert out.shape == x.shape
    assert torch.equal(out, x)


@pytest.mark.ssl_unit
def test_pool_tokens_downsamples_grid():
    """A 4x4 teacher grid average-pools to a 2x2 student grid with the right block means."""
    C = 3
    # Build a [1, 4*4, C] grid whose value encodes its (row, col) so we can check block means.
    grid = torch.zeros(1, 4, 4, C)
    for r in range(4):
        for c in range(4):
            grid[0, r, c, :] = r * 10 + c
    tokens = grid.reshape(1, 16, C)
    out = DinoV3PlModel._pool_tokens(tokens, (4, 4), (2, 2))
    assert out.shape == (1, 4, C)
    # Top-left 2x2 block: rows {0,1}, cols {0,1} -> values {0,1,10,11} -> mean 5.5
    assert torch.allclose(out[0, 0], torch.full((C,), 5.5))
    # Bottom-right 2x2 block: rows {2,3}, cols {2,3} -> {22,23,32,33} -> mean 27.5
    assert torch.allclose(out[0, 3], torch.full((C,), 27.5))
