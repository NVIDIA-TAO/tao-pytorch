# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for C-RADIO v4 backbone additions."""

from types import SimpleNamespace

import pytest

from nvidia_tao_pytorch.cv.backbone_v2.registry import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2 import radio


@pytest.mark.cv_unit
def test_c_radio_v4_so400m_config():
    """Test the C-RADIO v4 SO400M ViT config added for runtime support."""
    cfg = radio.radio_model_cfg["vit_so400m_patch16_224"]

    assert cfg["img_size"] == 224
    assert cfg["patch_size"] == 16
    assert cfg["embed_dim"] == 1152
    assert cfg["depth"] == 27
    assert cfg["num_heads"] == 16
    assert cfg["mlp_ratio"] == 4304 / 1152


@pytest.mark.cv_unit
def test_c_radio_v4_factories_use_released_teacher_summary_slots(monkeypatch):
    """Test C-RADIO v4 factories select the two exposed teacher summary tokens."""
    calls = []

    def fake_radio(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(radio, "RADIO", fake_radio)

    huge = radio.c_radio_v4_vit_huge_patch16()
    so400m = radio.c_radio_v4_vit_so400m_patch16()

    assert huge.backbone == "vit_huge_patch16_224"
    assert so400m.backbone == "vit_so400m_patch16_224"

    for model in (huge, so400m):
        assert model.summary_idxs == [0, 1]
        assert model.window_size is None
        assert model.num_teacher == 3
        assert model.cpe_max_size == 2048
        assert model.register_multiple == 10

    assert len(calls) == 2


@pytest.mark.cv_unit
def test_c_radio_v4_factories_are_registered():
    """Test the C-RADIO v4 factories are available through the backbone registry."""
    assert BACKBONE_REGISTRY.get("c_radio_v4_vit_huge_patch16") is radio.c_radio_v4_vit_huge_patch16
    assert BACKBONE_REGISTRY.get("c_radio_v4_vit_so400m_patch16") is radio.c_radio_v4_vit_so400m_patch16
