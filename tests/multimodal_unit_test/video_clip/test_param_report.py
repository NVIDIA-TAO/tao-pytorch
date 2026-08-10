# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the model parameter-summary report (CPU, no model/weights).

Guards against the misleading 'frozen' status on a component that still has
trainable params (e.g. a frozen ViT backbone with a trainable alignment head).
"""

import re
from types import SimpleNamespace

import pytest
import torch

from nvidia_tao_pytorch.multimodal.video_clip.model.adapters import base as base_mod
from nvidia_tao_pytorch.multimodal.video_clip.model.adapters.base import BaseCLIPAdapter


class _CapLog:
    """Minimal stand-in for the module logger that captures info() messages."""

    def __init__(self):
        self.msgs = []

    def info(self, msg, *args, **kwargs):
        self.msgs.append(str(msg))

    def warning(self, *args, **kwargs):
        pass


def _summary_stub():
    """A lightweight object exposing what _log_model_summary touches."""
    return SimpleNamespace(
        _format_params=BaseCLIPAdapter._format_params,
        _param_status=BaseCLIPAdapter._param_status,
        logit_scale=torch.zeros([]),
        logit_bias=torch.zeros([]),
    )


@pytest.mark.multimodal_unit
class TestParamStatus:
    """Status must be derived from the actual counts, never contradict them."""

    def test_status_from_counts(self):
        s = BaseCLIPAdapter._param_status
        assert s(0, 100) == "frozen"
        assert s(100, 100) == "trainable"
        assert s(40, 100) == "partial"
        assert s(0, 0) == "frozen"


@pytest.mark.multimodal_unit
class TestModelSummaryTable:
    """_log_model_summary rendering: count-based status + vision sub-rows."""

    def test_split_vision_rows_backbone_frozen_head_trainable(self, monkeypatch):
        cap = _CapLog()
        monkeypatch.setattr(base_mod, "logging", cap)
        BaseCLIPAdapter._log_model_summary(
            _summary_stub(),
            model_name="InternVideo2-CLIP L14",
            vision_total=309_300_000,
            vision_trainable=4_337_408,
            text_total=63_400_000,
            text_trainable=0,
            freeze_vision=True,
            freeze_text=True,
            vision_subrows=[
                ("Vision backbone", 0, 305_000_000),
                ("Vision align head", 4_337_408, 4_337_408),
            ],
        )
        out = "\n".join(cap.msgs)
        assert "Vision backbone" in out and "Vision align head" in out
        # backbone shows frozen, align head shows trainable -- no contradiction
        assert re.search(r"Vision backbone\s+\S+\s+\S+\s+frozen", out)
        assert re.search(r"Vision align head\s+\S+\s+\S+\s+trainable", out)
        # the misleading single "Vision encoder" row is gone
        assert "Vision encoder" not in out
        # text fully frozen
        assert re.search(r"Text encoder\s+\S+\s+\S+\s+frozen", out)

    def test_single_row_uses_partial_when_partially_trainable(self, monkeypatch):
        cap = _CapLog()
        monkeypatch.setattr(base_mod, "logging", cap)
        BaseCLIPAdapter._log_model_summary(
            _summary_stub(),
            model_name="X",
            vision_total=100,
            vision_trainable=40,
            text_total=10,
            text_trainable=0,
            freeze_vision=True,
            freeze_text=True,
        )
        out = "\n".join(cap.msgs)
        assert re.search(r"Vision encoder\s+\S+\s+\S+\s+partial", out)
