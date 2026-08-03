# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the Sinkhorn-Knopp centering DINOv3 uses (CPU).

DINOv3 centers teacher outputs with Sinkhorn-Knopp (SwAV) instead of DINOv2's softmax. That
path in ``DinoV2Loss`` was previously marked untested, so these tests confirm it produces valid
soft-assignment simplices, stays finite across the spec's teacher-temperature range, and is the
DINOv3 default. ``DinoHead`` L2-normalizes features against unit-norm prototypes, so teacher
logits are cosine similarities in ``[-1, 1]`` — the inputs are generated in that range to match.
"""
import pytest
import torch
from omegaconf import OmegaConf

from nvidia_tao_pytorch.ssl.nvdinov2.model.loss import DinoV2Loss
from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig

K = 64    # prototypes
B = 32    # batch (DINO CLS path)
P = 900   # masked patch tokens (iBOT path)


@pytest.mark.config
@pytest.mark.ssl_unit
def test_dinov3_default_centering_is_sinkhorn():
    """The DINOv3 config defaults to Sinkhorn-Knopp centering."""
    cfg = OmegaConf.structured(ExperimentConfig())
    assert cfg.model.centering_method == "sinkhorn"


@pytest.mark.ssl_unit
def test_sinkhorn_produces_valid_simplex():
    """Sinkhorn output is a non-negative probability simplex per sample (rows sum to 1)."""
    torch.manual_seed(0)
    loss = DinoV2Loss(num_prototypes=K, centering_method="sinkhorn")
    teacher_output = torch.empty(B, K).uniform_(-1, 1)  # cosine-range logits
    out = loss.centering(teacher_output, teacher_temp=0.04)
    assert out.shape == (B, K)
    assert torch.isfinite(out).all(), "sinkhorn produced non-finite values"
    assert (out >= 0).all(), "sinkhorn produced negative assignments"
    assert torch.allclose(out.sum(dim=-1), torch.ones(B), atol=1e-4)


@pytest.mark.ssl_unit
def test_sinkhorn_finite_across_teacher_temp_range():
    """No NaN/inf across the spec's teacher-temp range (0.04 start .. 0.07 base)."""
    torch.manual_seed(1)
    loss = DinoV2Loss(num_prototypes=K, centering_method="sinkhorn")
    teacher_output = torch.empty(B, K).uniform_(-1, 1)
    for temp in (0.04, 0.07):
        out = loss.centering(teacher_output, teacher_temp=temp)
        assert torch.isfinite(out).all(), f"non-finite at teacher_temp={temp}"
        assert torch.allclose(out.sum(dim=-1), torch.ones(B), atol=1e-4)


@pytest.mark.ssl_unit
def test_sinkhorn_ibot_patch_shape():
    """The iBOT path (many masked patch tokens) keeps the simplex property."""
    torch.manual_seed(2)
    loss = DinoV2Loss(num_prototypes=K, centering_method="sinkhorn")
    teacher_output = torch.empty(P, K).uniform_(-1, 1)
    out = loss.centering(teacher_output, teacher_temp=0.07)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.sum(dim=-1), torch.ones(P), atol=1e-4)
