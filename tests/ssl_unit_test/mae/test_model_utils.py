# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from nvidia_tao_pytorch.ssl.mae.model.utils import LayerNorm, GRN

# Define constants for test cases
BATCH_SIZE = 4
HEIGHT = 8
WIDTH = 8
CHANNELS = 32
EPS = 1e-6
DIM = CHANNELS // 2

@pytest.fixture
def input_data():
    # Randomly generate a batch of images for testing
    return torch.randn(BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)

def test_layer_norm(input_data):
    layer_norm = LayerNorm(CHANNELS, EPS, "channels_last")
    output = layer_norm(input_data)

    # Check if the shape of the output is correct.
    assert output.shape == input_data.shape

    # Check if the mean and variance are close to 0 and 1 respectively.
    mean = torch.mean(output, dim=(1, 2), keepdim=True)
    var = torch.var(output, dim=(1, 2), unbiased=False, keepdim=True)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1)
    assert torch.allclose(var, torch.ones_like(var), atol=1)

def test_layer_norm_channels_first(input_data):
    input_data = input_data.permute(0, 3, 1, 2)  # Change to 'channels_first' format
    layer_norm = LayerNorm(CHANNELS, EPS, "channels_first")
    output = layer_norm(input_data)

    # Similar checks as in test_layer_norm.

def test_grn(input_data):
    grn = GRN(DIM)
    input_data = torch.randn(BATCH_SIZE, HEIGHT, WIDTH, DIM)  # Input data with reduced channels for GRN layer
    output = grn(input_data)

    # Check if the shape of the output is correct.
    assert output.shape == input_data.shape

    # Check if the output has been normalized correctly.
    Gx = torch.norm(input_data, p=2, dim=(1, 2), keepdim=True)
    Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
    expected_output = grn.gamma * (input_data * Nx) + grn.beta + input_data
    assert torch.allclose(output, expected_output, atol=1e-6)
