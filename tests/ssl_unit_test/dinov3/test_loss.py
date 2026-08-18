# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 GramLoss unit tests (CPU; pure-torch, no GPU/xformers needed)."""
import pytest
import torch

from nvidia_tao_pytorch.ssl.dinov3.model.loss import GramLoss

BATCH = 2
N_PATCHES = 16
CHANNELS = 32


@pytest.mark.ssl_unit
def test_gram_loss_zero_for_identical_inputs():
    """Identical student/teacher patch tokens give exactly zero Gram loss."""
    torch.manual_seed(0)
    x = torch.randn(BATCH, N_PATCHES, CHANNELS)
    loss = GramLoss()(x, x.clone())
    assert loss.ndim == 0
    assert torch.allclose(loss, torch.zeros(()), atol=1e-6)


@pytest.mark.ssl_unit
def test_gram_loss_invariant_to_per_token_scale():
    """The Gram matrix is cosine-based, so scaling either side per-token is a no-op."""
    torch.manual_seed(1)
    student = torch.randn(BATCH, N_PATCHES, CHANNELS)
    teacher = torch.randn(BATCH, N_PATCHES, CHANNELS)
    scale = torch.rand(BATCH, N_PATCHES, 1) + 0.5  # strictly positive per-token scale
    base = GramLoss()(student, teacher)
    scaled = GramLoss()(student * scale, teacher)
    assert torch.allclose(base, scaled, atol=1e-5)


@pytest.mark.ssl_unit
def test_gram_loss_positive_for_different_inputs():
    """Different student/teacher features give a strictly positive loss."""
    torch.manual_seed(2)
    student = torch.randn(BATCH, N_PATCHES, CHANNELS)
    teacher = torch.randn(BATCH, N_PATCHES, CHANNELS)
    assert GramLoss()(student, teacher) > 0


@pytest.mark.ssl_unit
def test_gram_loss_computes_in_fp32():
    """fp16 inputs are upcast to fp32 internally; the loss is finite and fp32."""
    torch.manual_seed(3)
    student = torch.randn(BATCH, N_PATCHES, CHANNELS, dtype=torch.float16)
    teacher = torch.randn(BATCH, N_PATCHES, CHANNELS, dtype=torch.float16)
    loss = GramLoss()(student, teacher)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
