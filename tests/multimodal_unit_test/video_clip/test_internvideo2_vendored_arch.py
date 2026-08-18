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

"""Regression tests for the in-repo InternVideo2-CLIP architecture (CPU only).

The InternVideo2-CLIP arch used to be loaded from an external OpenGVLab
``multi_modality`` checkout pointed to by the ``INTERNVIDEO2_ROOT`` env var,
with ``sys.path``/``cwd`` mutated at import time. It is now vendored under
``video_clip.model.backbones.internvideo2`` and imported normally. These tests
guard that behavior without building the real 300M model or downloading
weights: the backbone and the HF asset resolver are stubbed.

The export-side detection (that ``export.py`` keys off ``is_internvideo2``) is
covered separately in ``test_scripts.py``; here we assert the adapter exposes
the attributes that contract depends on.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

from nvidia_tao_pytorch.multimodal.video_clip.model.adapters import internvideo2clip
from nvidia_tao_pytorch.multimodal.video_clip.utils.internvideo2_assets import AttrDict


class _StubBackbone(nn.Module):
    """Tiny stand-in for InternVideo2_CLIP_small (no weights, CPU)."""

    def __init__(self, config=None, is_pretrain=True):
        super().__init__()
        self.temp = nn.Parameter(torch.ones([]))
        self.tokenizer = object()
        self.vision_encoder = nn.Linear(2, 2)
        self.vision_align = nn.Linear(2, 2)
        self.text_encoder = nn.Linear(2, 2)
        paths = (
            config.model.vision_ckpt_path,
            config.model.text_ckpt_path,
            config.model.get("extra_ckpt_path", None),
        ) if config is not None else ()
        if any(paths):
            self.load_checkpoint(*paths)

    def load_checkpoint(self, *args, **kwargs):
        return None


@pytest.fixture
def stubbed_adapter(monkeypatch):
    """Build InternVideo2CLIP with the heavy arch + HF resolve stubbed out."""
    monkeypatch.setattr(internvideo2clip, "InternVideo2_CLIP_small", _StubBackbone)
    monkeypatch.setattr(
        internvideo2clip,
        "resolve_internvideo2_l14_assets",
        lambda model_cfg: AttrDict(
            {"vision_ckpt": None, "text_ckpt": None, "extra_ckpt": None}
        ),
    )

    def _build():
        return internvideo2clip.InternVideo2CLIP(num_frames=8, image_size=224)

    return _build


@pytest.mark.multimodal_unit
class TestVendoredInternVideo2Arch:
    """The adapter no longer needs INTERNVIDEO2_ROOT or sys.path/cwd mutation."""

    def test_constructs_without_env_var(self, stubbed_adapter, monkeypatch):
        monkeypatch.delenv("INTERNVIDEO2_ROOT", raising=False)
        model = stubbed_adapter()
        assert isinstance(model, internvideo2clip.InternVideo2CLIP)
        assert not hasattr(model, "internvideo2_root")

    def test_bogus_env_var_is_ignored(self, stubbed_adapter, monkeypatch):
        # A bogus value would have raised under the old root-pointer loader.
        monkeypatch.setenv("INTERNVIDEO2_ROOT", "/nonexistent/bogus/path")
        model = stubbed_adapter()
        assert model.is_internvideo2 is True

    def test_construction_does_not_mutate_syspath_or_cwd(
        self, stubbed_adapter, monkeypatch
    ):
        monkeypatch.delenv("INTERNVIDEO2_ROOT", raising=False)
        syspath_before = list(sys.path)
        cwd_before = os.getcwd()
        stubbed_adapter()
        assert sys.path == syspath_before
        assert os.getcwd() == cwd_before

    def test_exposes_export_detection_contract(self, stubbed_adapter, monkeypatch):
        # export.py keys video-input shaping off these two attributes.
        monkeypatch.delenv("INTERNVIDEO2_ROOT", raising=False)
        model = stubbed_adapter()
        assert model.is_internvideo2 is True
        assert isinstance(model.num_frames, int) and model.num_frames == 8

    def test_all_null_sources_build_without_component_loading(self, monkeypatch):
        monkeypatch.setattr(
            internvideo2clip, "InternVideo2_CLIP_small", _StubBackbone
        )
        monkeypatch.setattr(
            _StubBackbone,
            "load_checkpoint",
            lambda *args, **kwargs: pytest.fail(
                "architecture-only construction loaded component weights"
            ),
        )

        model = internvideo2clip.InternVideo2CLIP(
            internvideo2clip_hf_id=None,
            vision_encoder=None,
            text_encoder=None,
            clip_head=None,
            pretrained_ckpt=None,
        )

        assert isinstance(model.backbone, _StubBackbone)

    def test_complete_checkpoint_normalizes_lightning_prefixes(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            internvideo2clip, "InternVideo2_CLIP_small", _StubBackbone
        )
        expected = _StubBackbone()
        for parameter in expected.parameters():
            nn.init.constant_(parameter, 0.25)
        checkpoint = tmp_path / "complete.pth"
        torch.save({
            "state_dict": {
                f"model.backbone.{key}": value
                for key, value in expected.state_dict().items()
            }
        }, checkpoint)

        model = internvideo2clip.InternVideo2CLIP(
            pretrained_ckpt=str(checkpoint)
        )

        for key, value in expected.state_dict().items():
            torch.testing.assert_close(model.backbone.state_dict()[key], value)
