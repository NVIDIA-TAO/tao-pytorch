# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for bug 6459926 on the DINOv3 backbone builder.

``DinoV3PlModel.build_backbone`` is a ``classmethod`` that builds a ViT without a model
instance. An earlier revision read ``train_config.use_custom_attention`` directly, which
bypassed the GPU capability gate that :class:`DinoV2PlModel` applies in ``__init__``. On
Hopper (SM90A) the xformers ``memory_efficient_attention`` kernel fails to launch, so the
ungated flag crashed the process with SIGSEGV (pytest exited 139) instead of falling back
to SDPA. These tests freeze the gated behavior with a mocked device capability - no Hopper
GPU needed, and no CUDA forward is run.
"""
import pytest
import torch
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.dinov3.model.pl_model import DinoV3PlModel


class _FakeProps:
    def __init__(self, major, minor):
        self.major = major
        self.minor = minor


@pytest.fixture
def _fake_capability(monkeypatch):
    def _set(major, minor):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx=0: _FakeProps(major, minor))
    return _set


def _small_config():
    """A minimal DINOv3 experiment config using the smallest ViT for a cheap build."""
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.model.backbone.teacher_type = "vit_s"
    cfg.model.backbone.student_type = "vit_s"
    cfg.train.use_custom_attention = True
    return cfg


@pytest.mark.ssl_unit
@pytest.mark.parametrize(
    "capability,expected",
    [
        ((8, 0), True),    # A100 (validated platform)
        ((8, 9), True),    # Ada
        ((9, 0), False),   # H100 SM90A: the 6459926 regression
        ((10, 0), False),  # Blackwell: FA3 unsupported
    ],
)
def test_build_backbone_honors_capability_gate(_fake_capability, capability, expected):
    """build_backbone must gate the xformers path on GPU capability, not the raw flag."""
    _fake_capability(*capability)
    cfg = _small_config()

    backbone = DinoV3PlModel.build_backbone(cfg.model.backbone, cfg.train)

    assert backbone.blocks[0].use_custom_attention is expected


@pytest.mark.ssl_unit
def test_build_backbone_never_enables_custom_attention_when_disabled(_fake_capability):
    """An explicitly disabled flag stays disabled even on a supported GPU."""
    _fake_capability(8, 0)
    cfg = _small_config()
    cfg.train.use_custom_attention = False

    backbone = DinoV3PlModel.build_backbone(cfg.model.backbone, cfg.train)

    assert backbone.blocks[0].use_custom_attention is False


@pytest.mark.ssl_unit
def test_build_backbone_gates_explicit_use_custom_attention_override(_fake_capability):
    """An explicitly requested True is still AND-ed with the capability gate."""
    _fake_capability(9, 0)  # Hopper: the crashing arch
    cfg = _small_config()

    backbone = DinoV3PlModel.build_backbone(
        cfg.model.backbone, cfg.train, use_custom_attention=True
    )

    assert backbone.blocks[0].use_custom_attention is False
