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

"""Tests that build_model rejects model types the 5D video path cannot feed.

The video dataloader emits ``[B, T, C, H, W]`` and only the InternVideo2-CLIP
adapter consumes 5D input. C-RADIO / SigLIP2 / OpenCLIP are image-only, so
without a guard they build fine and then die on the first forward pass with a
shape error that reads like a data bug.
"""

from types import SimpleNamespace

import pytest

import nvidia_tao_pytorch.multimodal.video_clip.model.video_clip as video_clip_module
from nvidia_tao_pytorch.multimodal.video_clip.model.video_clip import build_model


def _experiment_config(model_type):
    """Minimal config carrying every field build_model reads before dispatch."""
    return SimpleNamespace(
        model=SimpleNamespace(
            type=model_type,
            image_size=224,
            init_logit_scale=None,
            init_logit_bias=None,
            freeze_vision_encoder=False,
            freeze_text_encoder=False,
            canonicalize_text=False,
        ),
        dataset=SimpleNamespace(augmentation=None),
        train=SimpleNamespace(loss_type="internvideo2_vtc"),
        peft=SimpleNamespace(enabled=False),
    )


@pytest.mark.multimodal_unit
class TestModelTypeGuard:
    """No weights are fetched: every case stops at or before the builder call."""

    @pytest.mark.parametrize("model_type", [
        "c-radio_v3-l",                 # radio_model_configs branch
        "siglip2-so400m-patch16-256",   # siglip2_model_configs branch
        "ViT-L-14-SigLIP-CLIPA-224",    # openclip_model_configs branch
        "definitely-not-a-model",       # open_clip catch-all `else` branch
    ])
    def test_image_only_types_are_rejected(self, model_type):
        """Each reachable non-InternVideo2 branch raises before building."""
        with pytest.raises(ValueError) as excinfo:
            build_model(_experiment_config(model_type))
        message = str(excinfo.value)
        assert model_type in message
        assert "internvideo2-clip-l14" in message

    def test_supported_type_still_reaches_the_builder(self, monkeypatch):
        """The guard must not over-reject the one type that does work."""
        monkeypatch.setattr(
            video_clip_module,
            "build_internvideo2clip_model",
            lambda **kwargs: ("model", "preprocess_train", "preprocess_val", "tokenizer"),
        )
        built = build_model(_experiment_config("internvideo2-clip-l14"))
        assert built.model == "model"
        assert built.tokenizer == "tokenizer"

    def test_peft_suppresses_pre_injection_parameter_report(self, monkeypatch):
        """PEFT runs must not log counts before LoRA modules are injected."""
        builder_kwargs = {}

        def _build(**kwargs):
            builder_kwargs.update(kwargs)
            return "model", "preprocess_train", "preprocess_val", "tokenizer"

        monkeypatch.setattr(
            video_clip_module,
            "build_internvideo2clip_model",
            _build,
        )
        config = _experiment_config("internvideo2-clip-l14")
        config.peft.enabled = True
        build_model(config)

        assert builder_kwargs["log_parameters"] is False
