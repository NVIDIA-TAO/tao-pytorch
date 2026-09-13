# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Sparse4D deformable aggregation CUDA operator."""

import pytest
import torch

from nvidia_tao_pytorch.cv.sparse4d.model.ops.deformable_aggregation import (
    deformable_aggregation_function,
)


@pytest.mark.cv_unit
@pytest.mark.sparse4d
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Deformable aggregation requires CUDA",
)
def test_deformable_aggregation_rejects_nonfinite_locations():
    """NaN and infinite locations must contribute no values or gradients."""
    device = torch.device("cuda")
    features = torch.ones((1, 4, 2), device=device, requires_grad=True)
    spatial_shape = torch.tensor([[[2, 2]]], dtype=torch.int32, device=device)
    scale_start_index = torch.tensor([[0]], dtype=torch.int32, device=device)
    sampling_locations = torch.tensor(
        [
            [
                [[[0.5, 0.5]]],
                [[[float("nan"), 0.5]]],
                [[[0.5, float("nan")]]],
                [[[float("inf"), 0.5]]],
                [[[0.5, -float("inf")]]],
                [[[0.0, 0.5]]],
                [[[1.0, 0.5]]],
            ]
        ],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    weights = torch.ones(
        (1, 7, 1, 1, 1, 1), device=device, requires_grad=True
    )

    output = deformable_aggregation_function(
        features,
        spatial_shape,
        scale_start_index,
        sampling_locations,
        weights,
    )

    torch.testing.assert_close(output[:, :1], torch.ones_like(output[:, :1]))
    assert torch.equal(output[:, 1:], torch.zeros_like(output[:, 1:]))
    assert torch.isfinite(output).all()

    output.sum().backward()
    for gradient in (features.grad, sampling_locations.grad, weights.grad):
        assert torch.isfinite(gradient).all()
    assert torch.equal(
        sampling_locations.grad[:, 1:],
        torch.zeros_like(sampling_locations.grad[:, 1:]),
    )
    assert torch.equal(weights.grad[:, 1:], torch.zeros_like(weights.grad[:, 1:]))


@pytest.mark.cv_unit
@pytest.mark.sparse4d
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Deformable aggregation requires CUDA",
)
def test_deformable_aggregation_uses_strict_open_coordinate_bounds():
    """Small in-bounds values pass while boundaries and outsiders are rejected."""
    device = torch.device("cuda")
    inside_low = torch.finfo(torch.float32).tiny
    inside_high = 1.0 - torch.finfo(torch.float32).eps
    features = torch.ones((1, 4, 1), device=device, requires_grad=True)
    spatial_shape = torch.tensor([[[2, 2]]], dtype=torch.int32, device=device)
    scale_start_index = torch.tensor([[0]], dtype=torch.int32, device=device)
    sampling_locations = torch.tensor(
        [
            [
                [[[inside_low, 0.5]]],
                [[[inside_high, 0.5]]],
                [[[0.5, inside_low]]],
                [[[0.5, inside_high]]],
                [[[0.0, 0.5]]],
                [[[1.0, 0.5]]],
                [[[0.5, 0.0]]],
                [[[0.5, 1.0]]],
                [[[-0.1, 0.5]]],
                [[[1.1, 0.5]]],
                [[[0.5, -0.1]]],
                [[[0.5, 1.1]]],
            ]
        ],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    weights = torch.ones(
        (1, 12, 1, 1, 1, 1), device=device, requires_grad=True
    )

    output = deformable_aggregation_function(
        features,
        spatial_shape,
        scale_start_index,
        sampling_locations,
        weights,
    )

    assert torch.all(output[:, :4] > 0)
    assert torch.equal(output[:, 4:], torch.zeros_like(output[:, 4:]))

    output.sum().backward()
    for gradient in (features.grad, sampling_locations.grad, weights.grad):
        assert torch.isfinite(gradient).all()
    assert torch.equal(
        sampling_locations.grad[:, 4:],
        torch.zeros_like(sampling_locations.grad[:, 4:]),
    )
    assert torch.equal(
        weights.grad[:, 4:],
        torch.zeros_like(weights.grad[:, 4:]),
    )
