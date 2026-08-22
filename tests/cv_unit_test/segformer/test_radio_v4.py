# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for C-RADIOv4 SegFormer backbone registration."""

import pytest

pytest.importorskip("fvcore")
from nvidia_tao_pytorch.cv.segformer.model.backbones import radio


def test_c_radio_v4_segformer_factories(monkeypatch):
    """The adapters must match the released v4 trunks and teacher tokens."""
    calls = []

    def fake_adapter(**kwargs):
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(radio, "RADIOAdapter", fake_adapter)

    huge = radio.c_radio_v4_vit_huge_patch16_224(activation_checkpoint=True)
    so400m = radio.c_radio_v4_vit_so400m_patch16_224(activation_checkpoint=True)

    assert huge["backbone"] == "vit_huge_patch16_224"
    assert huge["summary_idxs"] == [0, 1]
    assert huge["num_teacher"] == 3
    assert huge["register_multiple"] == 10
    assert huge["interaction_indexes"] == [[0, 7], [8, 15], [16, 23], [24, 31]]

    assert so400m["backbone"] == "vit_so400m_patch16_224"
    assert so400m["summary_idxs"] == [0, 1]
    assert so400m["num_teacher"] == 3
    assert so400m["register_multiple"] == 10
    assert so400m["interaction_indexes"] == [[0, 6], [7, 13], [14, 20], [21, 26]]
    assert all(call["activation_checkpoint"] for call in calls)
