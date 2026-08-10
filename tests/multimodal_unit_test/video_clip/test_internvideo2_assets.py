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

"""Tests for InternVideo2 asset resolution."""

from types import SimpleNamespace

import pytest
import torch

from nvidia_tao_pytorch.multimodal.video_clip.model.adapters.internvideo2clip import InternVideo2Tokenizer

from nvidia_tao_pytorch.multimodal.video_clip.utils.internvideo2_assets import (
    AttrDict,
    build_internvideo2_l14_config,
    resolve_internvideo2_l14_assets,
)


@pytest.mark.multimodal_unit
class TestInternVideo2Assets:
    """Test InternVideo2 L14 asset utilities."""

    def test_attrdict_supports_nested_attribute_access(self):
        """Nested dicts should become attribute-readable configs."""
        cfg = AttrDict({"model": {"vision_encoder": {"img_size": 224}}})

        assert cfg.model.vision_encoder.img_size == 224
        cfg.model.vision_encoder.num_frames = 8
        assert cfg["model"]["vision_encoder"]["num_frames"] == 8

    def test_resolve_local_paths_without_hf_download(self, tmp_path):
        """Explicit local paths should bypass HuggingFace download."""
        vision = tmp_path / "vision.bin"
        text = tmp_path / "mobileclip.pt"
        extra = tmp_path / "align.bin"
        for path in (vision, text, extra):
            path.write_bytes(b"ckpt")

        assets = resolve_internvideo2_l14_assets(SimpleNamespace(
            vision_encoder=str(vision),
            text_encoder=str(text),
            clip_head=str(extra),
            internvideo2clip_hf_id="unused/repo",
        ))

        assert assets == {
            "vision_ckpt": str(vision),
            "text_ckpt": str(text),
            "extra_ckpt": str(extra),
        }

    def test_resolve_hf_paths_for_missing_vision_and_extra(
        self, tmp_path, monkeypatch
    ):
        """Missing vision/extra paths should resolve through hf_hub_download."""
        text = tmp_path / "mobileclip.pt"
        text.write_bytes(b"ckpt")

        def fake_hf_hub_download(repo_id, filename):
            del repo_id
            path = tmp_path / filename.replace("/", "_")
            path.write_bytes(b"hf")
            return str(path)

        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download",
            fake_hf_hub_download,
        )

        assets = resolve_internvideo2_l14_assets(SimpleNamespace(
            vision_encoder=None,
            text_encoder=str(text),
            clip_head=None,
            internvideo2clip_hf_id=(
                "OpenGVLab/InternVideo2_distillation_models"
            ),
        ))

        assert assets["text_ckpt"] == str(text)
        assert assets["vision_ckpt"].endswith(
            "stage1_L14_L14_dist_1B_stage2_pytorch_model.bin"
        )
        assert assets["extra_ckpt"].endswith("clip_L14_pytorch_model.bin")

    def test_all_null_sources_return_architecture_only_assets(self):
        """All-null is the explicit random/architecture-only configuration."""
        assets = resolve_internvideo2_l14_assets(SimpleNamespace(
            vision_encoder=None,
            text_encoder=None,
            clip_head=None,
            internvideo2clip_hf_id=None,
        ))

        assert assets == {
            "vision_ckpt": None,
            "text_ckpt": None,
            "extra_ckpt": None,
        }

    def test_partial_sources_still_require_text_encoder(self):
        """A partial pretrained configuration must not become random init."""
        with pytest.raises(ValueError, match="requires model.text_encoder"):
            resolve_internvideo2_l14_assets(SimpleNamespace(
                vision_encoder="repo/vision",
                text_encoder=None,
                clip_head=None,
                internvideo2clip_hf_id=None,
            ))

    def test_build_l14_config_contains_expected_checkpoint_paths(self):
        """OpenGVLab config should expose resolved checkpoints."""
        assets = {
            "vision_ckpt": "/tmp/vision.bin",
            "text_ckpt": "/tmp/mobileclip.pt",
            "extra_ckpt": "/tmp/align.bin",
        }

        cfg = build_internvideo2_l14_config(
            assets, num_frames=4, image_size=224
        )

        assert cfg.model.vision_encoder.num_frames == 4
        assert cfg.model.vision_ckpt_path == "/tmp/vision.bin"
        assert cfg.model.text_ckpt_path == "/tmp/mobileclip.pt"
        assert cfg.model.extra_ckpt_path == "/tmp/align.bin"


@pytest.mark.multimodal_unit
class TestInternVideo2Tokenizer:
    """Test TAO tokenizer wrapper shape normalization."""

    def test_single_tensor_token_output_is_per_sample(self):
        """Single captions should collate to [batch, seq], not [batch, 1, seq]."""

        def source_tokenizer(text):
            assert isinstance(text, list)
            return torch.ones(len(text), 4, dtype=torch.long)

        tokenizer = InternVideo2Tokenizer(source_tokenizer)

        single = tokenizer("caption")[0]
        batch = tokenizer(["caption", "other"])[0]
        collated = torch.stack([single, single], dim=0)

        assert single.shape == (4,)
        assert batch.shape == (2, 4)
        assert collated.shape == (2, 4)
