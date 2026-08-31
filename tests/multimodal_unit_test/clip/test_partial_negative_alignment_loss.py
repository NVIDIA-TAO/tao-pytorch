# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nvidia_tao_pytorch.multimodal.clip.loss.partial_negative_alignment_loss import (
    partial_negative_alignment_loss,
)


def test_pa_loss_is_finite_and_differentiable():
    torch.manual_seed(3)
    images = torch.randn(8, 16, requires_grad=True)
    texts = torch.randn(8, 16, requires_grad=True)
    loss = partial_negative_alignment_loss(images, texts, 0.2, 10.0, 0.5)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(images.grad).all()
    assert torch.isfinite(texts.grad).all()


def test_top_ratio_one_matches_equation_for_diagonal_positives():
    torch.manual_seed(5)
    images = torch.randn(7, 12)
    texts = torch.randn(7, 12)
    pa = partial_negative_alignment_loss(images, texts, 0.2, 10.0, 1.0)
    scores = torch.nn.functional.normalize(images, dim=-1) @ (
        torch.nn.functional.normalize(texts, dim=-1).mT
    )

    def direction(values):
        excluded = torch.eye(len(values), dtype=torch.bool)
        negative = values.masked_fill(excluded, float("-inf"))
        smooth_negative = torch.logsumexp(10.0 * negative, dim=1) / 10.0
        return torch.relu(0.2 - values.diag() + smooth_negative).mean()

    reference = 0.5 * (direction(scores) + direction(scores.mT))
    assert torch.allclose(pa, reference, atol=1e-6, rtol=1e-6)


def test_compatible_pairs_are_positives_not_negatives():
    images = torch.eye(4, requires_grad=True)
    texts = torch.roll(torch.eye(4), shifts=1, dims=0).requires_grad_()
    compatible = torch.zeros(4, 4, dtype=torch.bool)
    compatible[0, 1] = True
    masked = partial_negative_alignment_loss(
        images, texts, 0.2, 10.0, 0.5, compatible_mask=compatible
    )
    unmasked = partial_negative_alignment_loss(images, texts, 0.2, 10.0, 0.5)
    assert masked < unmasked
    masked.backward()
    assert torch.isfinite(images.grad).all()


def test_all_compatible_pairs_return_zero():
    features = torch.eye(3)
    compatible = torch.ones(3, 3, dtype=torch.bool)
    loss = partial_negative_alignment_loss(
        features, features, 0.2, 10.0, 0.5, compatible_mask=compatible
    )
    assert loss.item() == 0.0


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_invalid_temperature_rejected(temperature):
    features = torch.eye(3)
    with pytest.raises(ValueError, match="inverse temperature"):
        partial_negative_alignment_loss(features, features, 0.2, temperature, 0.5)


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1])
def test_invalid_ratio_rejected(ratio):
    features = torch.eye(3)
    with pytest.raises(ValueError, match="top_ratio"):
        partial_negative_alignment_loss(features, features, 0.2, 10.0, ratio)


def test_invalid_margin_and_mismatched_batches_rejected():
    features = torch.eye(3)
    with pytest.raises(ValueError, match="margin"):
        partial_negative_alignment_loss(features, features, -0.1, 10.0, 0.5)
    with pytest.raises(ValueError, match="equal image and text"):
        partial_negative_alignment_loss(features, features[:2], 0.2, 10.0, 0.5)
