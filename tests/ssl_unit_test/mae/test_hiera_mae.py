# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
from nvidia_tao_pytorch.ssl.mae.model.hiera_mae import (
    MaskedAutoencoderHiera,
    mae_hiera_tiny_224, mae_hiera_small_224,
    mae_hiera_base_224, mae_hiera_large_224, mae_hiera_huge_224
)


@pytest.fixture
def model():
    return MaskedAutoencoderHiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3), q_pool=2, mask_ratio=0.5)

@pytest.fixture
def input_data():
    batch_size = 4
    channels = 3
    height = 224
    width = 224
    return torch.randn(batch_size, channels, height, width)

def test_forward_encoder(model, input_data):
    mask_ratio = 0.6
    output = model.forward_encoder(input_data, mask_ratio)
    assert len(output) == 2
    latent, mask = output
    assert latent.shape[1:] == (19, 2, 2, 768)
    assert mask.shape[1:] == (49,)

def test_forward_decoder(model, input_data):
    mask_ratio = 0.6
    latent, mask = model.forward_encoder(input_data, mask_ratio)
    output = model.forward_decoder(latent, mask)
    assert len(output) == 2
    pred, pred_mask = output
    assert pred.shape[1:] == (196, 768)
    assert pred_mask.shape[1:] == (196,)

def test_forward_loss(model, input_data):
    mask_ratio = 0.6
    latent, mask = model.forward_encoder(input_data, mask_ratio)
    pred, pred_mask = model.forward_decoder(latent, mask)
    loss, pred, label = model.forward_loss(input_data, pred, ~pred_mask)
    assert loss
    assert pred.shape[1:] == (768,)
    assert label.shape[1:] == (768,)

def test_forward(model, input_data):
    output = model(input_data)
    assert len(output) == 3  # refactoring output to 3 since the forward function was edited.
    loss, pred, mask = output
    assert loss.shape == ()
    assert pred.shape[1:] == (768,)
    assert mask.shape[1:] == (49,)

@pytest.mark.parametrize("model_func", [mae_hiera_tiny_224, mae_hiera_small_224, mae_hiera_base_224, mae_hiera_large_224, mae_hiera_huge_224])
def test_pretrained_models(model_func):
    model = model_func()
    input_data = torch.randn(4, 3, 224, 224)
    output = model(input_data)
    assert len(output) == 3
    loss, pred, mask = output
