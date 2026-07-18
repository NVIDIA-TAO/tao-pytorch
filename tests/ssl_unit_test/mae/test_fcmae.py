# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nvidia_tao_pytorch.ssl.mae.model.fcmae import *

def test_fcmae_init():
    model = FCMAE(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320])
    assert isinstance(model, torch.nn.Module)
    assert model.depths == [2, 2, 6, 2]
    assert model.dims == [40, 80, 160, 320]

def test_fcmae_forward_encoder():
    model = FCMAE(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320])
    imgs = torch.randn(1, 3, 224, 224)
    mask_ratio = 0.5
    x, mask = model.forward_encoder(imgs, mask_ratio)
    assert x.shape == (1, 320, 7, 7)
    assert mask.shape == (1, 49)

def test_fcmae_forward_decoder():
    model = FCMAE(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320])
    x = torch.randn(1, 320, 7, 7)
    mask = torch.ones(1, 49)
    pred = model.forward_decoder(x, mask)
    assert pred.shape == (1, 3072, 7, 7)

def test_fcmae_forward_loss():
    model = FCMAE(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320])
    imgs = torch.randn(1, 3, 224, 224)
    pred = torch.randn(1, 3072, 7, 7)
    mask = torch.ones(1, 49)
    loss = model.forward_loss(imgs, pred, mask)
    assert isinstance(loss, torch.Tensor)

def test_fcmae_forward():
    model = FCMAE(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320])
    imgs = torch.randn(1, 3, 224, 224)
    loss, pred, mask = model.forward(imgs)
    assert isinstance(loss, torch.Tensor)
    assert pred.shape == (1, 3072, 7, 7)
    assert mask.shape == (1, 49)

def test_mae_convnextv2_atto():
    model = mae_convnextv2_atto()
    assert isinstance(model, FCMAE)
    assert model.depths == [2, 2, 6, 2]
    assert model.dims == [40, 80, 160, 320]

def test_mae_convnextv2_femto():
    model = mae_convnextv2_femto()
    assert isinstance(model, FCMAE)
    assert model.depths == [2, 2, 6, 2]
    assert model.dims == [48, 96, 192, 384]

def test_mae_convnextv2_pico():
    model = mae_convnextv2_pico()
    assert isinstance(model, FCMAE)
    assert model.depths == [2, 2, 6, 2]
    assert model.dims == [64, 128, 256, 512]

def test_mae_convnextv2_nano():
    model = mae_convnextv2_nano()
    assert isinstance(model, FCMAE)
    assert model.depths == [2, 2, 8, 2]
    assert model.dims == [80, 160, 320, 640]

def test_mae_convnextv2_tiny():
    model = mae_convnextv2_tiny()
    assert isinstance(model, FCMAE)
    assert model.depths == [3, 3, 9, 3]
    assert model.dims == [96, 192, 384, 768]

def test_mae_convnextv2_base():
    model = mae_convnextv2_base()
    assert isinstance(model, FCMAE)
    assert model.depths == [3, 3, 27, 3]
    assert model.dims == [128, 256, 512, 1024]

def test_mae_convnextv2_large():
    model = mae_convnextv2_large()
    assert isinstance(model, FCMAE)
    assert model.depths == [3, 3, 27, 3]
    assert model.dims == [192, 384, 768, 1536]

def test_mae_convnextv2_huge():
    model = mae_convnextv2_huge()
    assert isinstance(model, FCMAE)
    assert model.depths == [3, 3, 27, 3]
    assert model.dims == [352, 704, 1408, 2816]

@pytest.mark.parametrize("model_fn", [
    mae_convnextv2_atto,
    mae_convnextv2_femto,
    mae_convnextv2_pico,
    mae_convnextv2_nano,
    mae_convnextv2_tiny,
    mae_convnextv2_base,
    mae_convnextv2_large,
])
def test_model_fn(model_fn):
    model = model_fn()
    assert isinstance(model, FCMAE)
