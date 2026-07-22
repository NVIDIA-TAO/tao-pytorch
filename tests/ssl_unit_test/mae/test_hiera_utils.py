# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn
from nvidia_tao_pytorch.ssl.mae.model.hiera_utils import Reroll, Unroll


def test_unroll_2d():
    unroll = Unroll((10, 12), (2, 3), [(1, 1), (1, 1)])
    x = torch.randn(1, 5 * 4, 3)
    output = unroll(x)
    assert output.shape == (1, 5 * 4, 3)


def test_unroll_3d():
    unroll = Unroll((8, 12, 16), (2, 3, 4), [(1, 1, 1), (1, 1, 1)])
    x = torch.randn(1, 4 * 4 * 4, 3)
    output = unroll(x)
    assert output.shape == (1, 4 * 4 * 4, 3)


def test_reroll_2d():
    reroll = Reroll((10, 12), (2, 3), [(1, 1)], [0], 1)
    x = torch.randn(1, 20, 3)
    output = reroll(x, block_idx=0)
    assert output.shape == (1, 5, 4, 3)


def test_reroll_3d():
    reroll = Reroll((8, 12, 16), (2, 3, 4), [(1, 1, 1)], [0], 1)
    x = torch.randn(1, 4 * 4 * 4, 3)
    output = reroll(x, block_idx=0)
    assert output.shape == (1, 4, 4, 4, 3)


def test_reroll_masked_2d():
    reroll = Reroll((10, 12), (2, 3), [(1, 1)], [0], 1)
    x = torch.randn(1, 20, 3)
    mask = torch.ones_like(x[:, :, 0], dtype=torch.bool)
    output = reroll(x, block_idx=0, mask=mask)
    assert output.shape == (1, 20, 1, 1, 3)


def test_reroll_masked_3d():
    reroll = Reroll((8, 12, 16), (2, 3, 4), [(1, 1, 1)], [0], 1)
    x = torch.randn(1, 4 * 4 * 4, 3)
    mask = torch.ones_like(x[:, :, 0], dtype=torch.bool)
    output = reroll(x, block_idx=0, mask=mask)
    assert output.shape == (1, 64, 1, 1, 1, 3)


def test_unroll_reroll_consistency():
    unroll = Unroll((10, 12), (2, 3), [(1, 1)])
    reroll = Reroll((10, 12), (2, 3), [(1, 1)], [0], 1)
    x = torch.randn(1, 5 * 4, 3)
    output_unroll = unroll(x)
    output_reroll = reroll(output_unroll, block_idx=0)
    assert torch.allclose(x, output_reroll.reshape(1, -1, 3))
