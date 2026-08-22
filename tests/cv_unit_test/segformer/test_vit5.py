# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ViT-5 SegFormer adapter."""

import pytest
import torch

pytest.importorskip("fvcore")
from nvidia_tao_pytorch.cv.segformer.model.backbones import vit5


def test_vit5_rope_is_device_safe_and_shape_preserving():
    rope = vit5.VisionRotaryEmbedding(dim=8, pt_seq_len=14)
    tokens = torch.randn(2, 16, 3, 16)
    output = rope(tokens)
    assert output.shape == tokens.shape
    assert output.device == tokens.device
    assert output.dtype == tokens.dtype


def test_vit5_patch_tokens_keep_official_order_and_interpolate_positions():
    model = vit5.ViT5Adapter(
        embed_dim=64,
        depth=4,
        num_heads=4,
        pretrained_resolution=64,
        resolution=128,
        conv_inplane=8,
        deform_num_heads=8,
        interaction_indexes=[[0, 0], [1, 1], [2, 2], [3, 3]],
    )
    image = torch.randn(1, 3, 128, 128)
    tokens, patch_h, patch_w = model._patch_tokens(image)
    assert (patch_h, patch_w) == (8, 8)
    assert tokens.shape == (1, 1 + 64 + 4, 64)
    assert torch.equal(tokens[:, :1], model.cls_token)
    assert torch.equal(tokens[:, -4:], model.reg_token)


def test_vit5_large_factory_matches_released_architecture(monkeypatch):
    captured = {}

    def fake_adapter(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(vit5, "ViT5Adapter", fake_adapter)
    result = vit5.vit5_large_patch16_224(
        resolution=1024,
        freeze_at="all",
        activation_checkpoint=True,
    )
    assert result["embed_dim"] == 1024
    assert result["depth"] == 24
    assert result["num_heads"] == 16
    assert result["pretrained_resolution"] == 224
    assert result["interaction_indexes"] == [[0, 5], [6, 11], [12, 17], [18, 23]]
    assert result["freeze_at"] == "all"
    assert result["activation_checkpoint"] is True
