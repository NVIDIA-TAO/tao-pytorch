# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RADIO runtime-fix additions."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from nvidia_tao_pytorch.multimodal.radio.config.default_config import (
    DataPathFormat,
    TeacherConfig,
)
from nvidia_tao_pytorch.multimodal.radio.dataloader.filters.native_resolution_filter import (
    NativeResolutionFilter,
)
from nvidia_tao_pytorch.multimodal.radio.distillation import loss as loss_module
from nvidia_tao_pytorch.multimodal.radio.distillation.distiller import MultiTeacherDistiller
from nvidia_tao_pytorch.multimodal.radio.distillation.loss import (
    AttnFDHead,
    DistillationLoss,
)


class _Image:
    """Small image stand-in that exposes the PIL ``size`` contract."""

    def __init__(self, width, height):
        self.size = (width, height)


class _FakeRADIO(nn.Module):
    """Minimal RADIO stand-in for summary-token helper tests."""

    def __init__(self, num_features=8, summary_idxs=(0, 1)):
        super().__init__()
        self.num_features = num_features
        self.summary_idxs = list(summary_idxs)


def _summary_loss(summary_token_idx):
    """Create a DistillationLoss instance without running full model setup."""
    loss = DistillationLoss.__new__(DistillationLoss)
    nn.Module.__init__(loss)
    loss.summary_token_idx = summary_token_idx
    return loss


@pytest.mark.multimodal_unit
def test_native_resolution_filter_keeps_only_images_within_bounds():
    """Test native-resolution filtering before resize/crop."""
    samples = [
        ("keep", _Image(640, 480)),
        ("short-side-too-small", _Image(640, 128)),
        ("area-too-small", _Image(256, 256)),
        ("aspect-too-extreme", _Image(2000, 400)),
    ]
    stage = NativeResolutionFilter(
        image_tuple_idx=1,
        min_short_side=256,
        min_long_side=512,
        min_area=640 * 480,
        max_aspect_ratio=3.0,
    )

    kept = list(stage.run(samples))

    assert kept == [samples[0]]
    assert stage.num_seen == len(samples)
    assert stage.num_filtered == len(samples) - 1


@pytest.mark.multimodal_unit
def test_radio_config_defaults_include_runtime_fix_fields():
    """Test config fields added for native-resolution and C-RADIO v4 distillation."""
    data_cfg = DataPathFormat()
    teacher_cfg = TeacherConfig()

    assert data_cfg.native_resolution_filter is None
    assert teacher_cfg.summary_token_idx is None
    assert teacher_cfg.spatial_mlp_version == "v2"
    assert teacher_cfg.spatial_num_inner is None


@pytest.mark.multimodal_unit
def test_distillation_loss_selects_flattened_radio_summary_token(monkeypatch):
    """Test per-teacher RADIO summary-token selection from flattened summaries."""
    monkeypatch.setattr(loss_module, "RADIO", _FakeRADIO)
    loss = _summary_loss(summary_token_idx=2)
    model = _FakeRADIO(num_features=8, summary_idxs=(0, 2))
    summary = torch.arange(16).reshape(2, 8)

    selected = loss._select_summary_token(summary, model)

    assert loss._summary_feature_dim(model, fallback_dim=8) == 4
    assert torch.equal(selected, summary.reshape(2, 2, 4)[:, 1])


@pytest.mark.multimodal_unit
def test_distillation_loss_selects_rank3_radio_summary_token(monkeypatch):
    """Test per-teacher RADIO summary-token selection from tokenized summaries."""
    monkeypatch.setattr(loss_module, "RADIO", _FakeRADIO)
    loss = _summary_loss(summary_token_idx=1)
    model = _FakeRADIO(num_features=8, summary_idxs=(0, 1))
    summary = torch.arange(24).reshape(2, 3, 4)

    selected = loss._select_summary_token(summary, model)

    assert torch.equal(selected, summary[:, 1])


@pytest.mark.multimodal_unit
def test_distillation_loss_rejects_unknown_radio_summary_token(monkeypatch):
    """Test invalid RADIO summary-token indices fail explicitly."""
    monkeypatch.setattr(loss_module, "RADIO", _FakeRADIO)
    loss = _summary_loss(summary_token_idx=1)
    model = _FakeRADIO(num_features=8, summary_idxs=(0, 2))

    with pytest.raises(ValueError, match="summary_token_idx=1"):
        loss._select_summary_token(torch.zeros(2, 8), model)


@pytest.mark.multimodal_unit
def test_attn_fd_head_preserves_sequence_and_projects_channels():
    """Test the attention-based C-RADIO v4 feature-distillation head."""
    head = AttnFDHead(
        input_size=8,
        hidden_size=16,
        output_size=6,
        num_inner=0,
        num_blocks=1,
        num_heads=2,
    )

    out = head(torch.randn(2, 5, 8))

    assert out.shape == (2, 5, 6)


@pytest.mark.multimodal_unit
def test_extract_and_lookup_upstream_token_slots():
    """Test checkpoint teacher token-slot helpers."""
    args = {
        "cls_token_per_teacher": True,
        "teachers": [
            {"name": "siglip2", "token_slot": 2},
            SimpleNamespace(name="dinov3", token_slot=0),
        ],
    }

    slots = MultiTeacherDistiller._extract_upstream_token_slots(args)

    assert slots == {"siglip2": 2, "dinov3": 0}
    assert MultiTeacherDistiller._lookup_upstream_token_slot("siglip-2", slots) == 2
    assert MultiTeacherDistiller._lookup_upstream_token_slot("missing", slots) is None


@pytest.mark.multimodal_unit
def test_extract_upstream_token_slots_without_per_teacher_cls_tokens():
    """Test checkpoint token slots collapse to zero when no per-teacher CLS token exists."""
    args = {
        "cls_token_per_teacher": False,
        "teachers": [{"name": "sam3", "token_slot": 5}],
    }

    assert MultiTeacherDistiller._extract_upstream_token_slots(args) == {"sam3": 0}


@pytest.mark.multimodal_unit
def test_detect_upstream_head_info_infers_attn_projection_and_summary_slot(tmp_path):
    """Test upstream checkpoint inspection for C-RADIO v4 projection heads."""
    ckpt_path = tmp_path / "radio.pt"
    torch.save(
        {
            "args": {
                "cls_token_per_teacher": True,
                "teachers": [{"name": "siglip2", "token_slot": 1}],
            },
            "state_dict": {
                "_heads.siglip2.weight": torch.ones(1),
                "_feature_projections.siglip2.blocks.0.attn.qkv.weight": torch.ones(3, 3),
                "unrelated.weight": torch.zeros(1),
            },
        },
        ckpt_path,
    )
    distiller = object.__new__(MultiTeacherDistiller)
    object.__setattr__(
        distiller,
        "teacher_configs",
        [{
            "model_config": SimpleNamespace(
                backbone=SimpleNamespace(type="siglip2_so400m_patch16_512")
            )
        }],
    )
    object.__setattr__(distiller, "_pretrained_head_sd", None)

    MultiTeacherDistiller._detect_upstream_head_info(distiller, str(ckpt_path))

    teacher_config = distiller.teacher_configs[0]
    assert teacher_config["upstream_name"] == "siglip2"
    assert teacher_config["summary_token_idx"] == 1
    assert teacher_config["spatial_mlp_version"] == "attn"
    assert teacher_config["spatial_num_inner"] == 0
    assert set(distiller._pretrained_head_sd) == {
        "_heads.siglip2.weight",
        "_feature_projections.siglip2.blocks.0.attn.qkv.weight",
    }


@pytest.mark.multimodal_unit
def test_warmstart_projection_heads_loads_matching_upstream_weights():
    """Test warm-start copying for detected upstream projection heads."""
    summary_head = nn.Linear(2, 2)
    spatial_head = nn.Linear(2, 2)
    loss_fn = SimpleNamespace(
        projection_layer_summary=summary_head,
        projection_layer=spatial_head,
    )
    distiller = object.__new__(MultiTeacherDistiller)
    object.__setattr__(distiller, "distillation_loss_fns", [loss_fn])
    object.__setattr__(distiller, "teacher_configs", [{"upstream_name": "siglip2"}])

    head_sd = {
        "_heads.siglip2.weight": torch.full_like(summary_head.weight, 2.0),
        "_heads.siglip2.bias": torch.full_like(summary_head.bias, 3.0),
        "_feature_projections.siglip2.weight": torch.full_like(spatial_head.weight, 4.0),
        "_feature_projections.siglip2.bias": torch.full_like(spatial_head.bias, 5.0),
    }

    MultiTeacherDistiller._warmstart_projection_heads(distiller, head_sd)

    assert torch.allclose(summary_head.weight, head_sd["_heads.siglip2.weight"])
    assert torch.allclose(summary_head.bias, head_sd["_heads.siglip2.bias"])
    assert torch.allclose(spatial_head.weight, head_sd["_feature_projections.siglip2.weight"])
    assert torch.allclose(spatial_head.bias, head_sd["_feature_projections.siglip2.bias"])
