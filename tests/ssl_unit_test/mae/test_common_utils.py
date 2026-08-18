# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

import torch

from nvidia_tao_pytorch.ssl.mae.utils.lr_decay  import (
    add_weight_decay, param_groups_vit, get_layer_id_for_vit,
    get_layer_id_for_convnextv2, param_groups_convnextv2
)
from nvidia_tao_pytorch.ssl.mae.utils.pos_embed import ( 
    get_2d_sincos_pos_embed, 
    get_1d_sincos_pos_embed_from_grid,
    interpolate_pos_embed
)

# Sample model for testing
class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = torch.nn.Linear(10, 5)
        self.conv = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.blocks = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(5, 5),
            torch.nn.ReLU()
        ) for _ in range(4)])

    def forward(self, x):
        pass

@pytest.fixture
def model():
    return DummyModel()

def test_add_weight_decay(model):
    param_groups = add_weight_decay(model, weight_decay=1e-5)
    assert len(param_groups) == 2

def test_param_groups_vit(model):
    param_groups = param_groups_vit(model, weight_decay=1e-5)
    assert len(param_groups) > 0

def test_get_layer_id_for_vit():
    layer_id = get_layer_id_for_vit("stages.2.blocks.3", 1)
    assert isinstance(layer_id, int) and layer_id <= 12

def test_get_layer_id_for_convnextv2():
    layer_id = get_layer_id_for_convnextv2("stages.2.3")
    assert isinstance(layer_id, int) and layer_id <= 12

def test_param_groups_convnextv2(model):
    param_groups = param_groups_convnextv2(model, weight_decay=1e-5)
    assert len(param_groups) > 0


# Fixtures for common data used in multiple tests
@pytest.fixture
def embed_dim():
    return 64

@pytest.fixture
def grid_size():
    return 8

@pytest.fixture
def grid(grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    return np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

@pytest.fixture
def pos():
    return np.arange(32, dtype=np.float32)

# Test for get_2d_sincos_pos_embed function
def test_get_2d_sincos_pos_embed(embed_dim, grid_size):
    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
    assert isinstance(pos_embed, np.ndarray)
    assert pos_embed.shape == (grid_size * grid_size, embed_dim)

# Test for get_1d_sincos_pos_embed_from_grid function
def test_get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    emb = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (pos.shape[0], embed_dim)
