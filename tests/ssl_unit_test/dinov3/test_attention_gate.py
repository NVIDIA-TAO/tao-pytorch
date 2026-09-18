# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for bug 6459926 on the DINOv3 backbone-build path.

``DinoV2PlModel.__init__`` force-disables the xformers ``memory_efficient_attention``
path on Hopper (``(9, x)``) and Blackwell (``>= (10, 0)``), because the kernel fails to
launch there. ``DinoV3PlModel.build_backbone`` must honor that gate: when it read the raw
``train.use_custom_attention`` flag instead of the arch-gated value, the backbone was built
with the crashing kernel enabled and the first real CUDA forward aborted the whole pytest
process with SIGSEGV (exit 139) rather than raising a Python exception.

These tests freeze the gate with a mocked device capability - no Hopper/Blackwell GPU and
no CUDA forward needed.
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
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda idx=0: _FakeProps(major, minor)
        )
    return _set


def _small_cfg():
    """Smallest supported v3 backbone, so the build stays cheap on CPU."""
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.model.backbone.teacher_type = "vit_s"
    cfg.model.backbone.student_type = "vit_s"
    cfg.train.use_custom_attention = True
    cfg.train.num_prototypes = 16
    cfg.model.head.num_layers = 1
    cfg.model.head.hidden_dim = 32
    cfg.model.head.bottleneck_dim = 16
    return cfg


@pytest.mark.ssl_unit
@pytest.mark.parametrize(
    "capability,expected",
    [
        ((8, 0), True),    # A100: custom path is supported
        ((8, 9), True),    # Ada
        ((9, 0), False),   # H100 SM90A: kernel fails to launch (bug 6459926)
        ((10, 0), False),  # Blackwell: FA3 unsupported
        ((12, 0), False),  # beyond Blackwell
    ],
)
def test_build_backbone_applies_arch_gate(_fake_capability, capability, expected):
    """build_backbone must gate the configured flag on the GPU arch, not trust it blindly."""
    _fake_capability(*capability)
    cfg = _small_cfg()
    backbone = DinoV3PlModel.build_backbone(cfg.model.backbone, cfg.train)
    assert backbone.blocks[0].use_custom_attention is expected


@pytest.mark.ssl_unit
def test_full_model_build_propagates_arch_gate(_fake_capability):
    """The training constructor must pass its resolved gate into both backbones."""
    _fake_capability(9, 0)
    model = DinoV3PlModel(_small_cfg())
    assert model.use_custom_attention is False
    assert model.student.backbone.blocks[0].use_custom_attention is False
    assert model.teacher.backbone.blocks[0].use_custom_attention is False


@pytest.mark.ssl_unit
def test_build_backbone_without_cuda_honors_config(monkeypatch):
    """CPU-only construction preserves the configured attention choice."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = _small_cfg()
    backbone = DinoV3PlModel.build_backbone(cfg.model.backbone, cfg.train)
    assert backbone.blocks[0].use_custom_attention is True


@pytest.mark.ssl_unit
@pytest.mark.parametrize("capability", [(8, 0), (9, 0), (10, 0)])
def test_build_backbone_honors_explicit_override(_fake_capability, capability):
    """An explicit choice (what the training path passes) wins over the config flag."""
    _fake_capability(*capability)
    cfg = _small_cfg()
    backbone = DinoV3PlModel.build_backbone(
        cfg.model.backbone, cfg.train, use_custom_attention=False
    )
    assert backbone.blocks[0].use_custom_attention is False


@pytest.mark.ssl_unit
def test_build_backbone_respects_disabled_config(_fake_capability):
    """A pre-Hopper GPU still honors an explicitly disabled config flag."""
    _fake_capability(8, 0)
    cfg = _small_cfg()
    cfg.train.use_custom_attention = False
    backbone = DinoV3PlModel.build_backbone(cfg.model.backbone, cfg.train)
    assert backbone.blocks[0].use_custom_attention is False
