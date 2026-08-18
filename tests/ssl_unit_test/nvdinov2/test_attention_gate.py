# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for bug 6459926: xformers custom attention crashes on Hopper (SM90A).

The xformers ``memory_efficient_attention`` kernel fails to launch on Hopper (compute
capability ``(9, x)``) with ``cudaErrorLaunchFailure``, but the old gate only force-disabled
the custom path on Blackwell (``>= (10, 0)``), so H100 ran the crashing kernel by default.
The policy now enables the custom path only on pre-Hopper GPUs (``< (9, 0)``). These tests
freeze that policy with a mocked device capability - no Hopper GPU needed.
"""
import pytest
import torch
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.nvdinov2.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.nvdinov2.model.pl_model import DinoV2PlModel


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


@pytest.mark.ssl_unit
@pytest.mark.parametrize(
    "capability,expected",
    [
        ((8, 0), True),    # A100 (validated platform)
        ((8, 6), True),    # Ampere consumer
        ((8, 9), True),    # Ada
        ((9, 0), False),   # H100 SM90A: the 6459926 regression
        ((10, 0), False),  # Blackwell: FA3 unsupported (pre-existing policy)
        ((12, 0), False),  # beyond Blackwell
    ],
)
def test_custom_attention_supported_policy(_fake_capability, capability, expected):
    """The gate enables the xformers custom path only on pre-Hopper GPUs."""
    _fake_capability(*capability)
    model = object.__new__(DinoV2PlModel)  # bypass __init__; test the policy in isolation
    assert model._custom_attention_supported() is expected


@pytest.mark.ssl_unit
def test_custom_attention_supported_no_cuda(monkeypatch):
    """With no CUDA device to probe, the configured flag is honored (no crash)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = object.__new__(DinoV2PlModel)
    assert model._custom_attention_supported() is True


@pytest.mark.ssl_unit
@pytest.mark.parametrize(
    "capability,expected",
    [((8, 0), True), ((9, 0), False), ((10, 0), False)],
)
def test_model_use_custom_attention_wiring(_fake_capability, capability, expected):
    """With use_custom_attention=True in config, the built model honors the capability policy."""
    _fake_capability(*capability)
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.model.backbone.teacher_type = "vit_s"
    cfg.model.backbone.student_type = "vit_s"
    cfg.train.use_custom_attention = True
    model = DinoV2PlModel(cfg)
    assert model.use_custom_attention is expected
