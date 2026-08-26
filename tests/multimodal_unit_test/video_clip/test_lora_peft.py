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

"""Unit tests for LoRA PEFT wiring in the video_clip task.

The LoRA building blocks (LoRALinear, inject_lora, merge_lora, PreservationLoss)
live in ``multimodal.clip.model.lora`` / ``preservation_loss`` and are exercised
thoroughly by the clip unit suite. These tests cover the video_clip-specific
contract: the video_clip adapter base ``get_encoder_blocks``, injection driven by
the real ``VideoCLIPPEFTConfig`` defaults against InternVideo2/MobileCLIP-style
fused-QKV blocks, and that the two config copies stay field-identical.
"""

import pytest
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.multimodal.video_clip.model.adapters.base import (
    BaseCLIPAdapter,
)
from nvidia_tao_pytorch.multimodal.clip.model.lora import (
    LoRALinear,
    inject_lora,
    merge_lora,
)
from nvidia_tao_pytorch.multimodal.clip.model.preservation_loss import (
    build_preservation_loss,
)
from nvidia_tao_pytorch.config.video_clip.default_config import (
    VideoCLIPPEFTConfig,
    VideoCLIPRegularizationConfig,
)


# ---------------------------------------------------------------------------
# Mock adapter mirroring the real InternVideo2-CLIP leaf-module naming:
#   vision attention -> fused 'qkv' + 'proj'
#   text   attention -> fused 'qkv_proj' + 'out_proj'
# ---------------------------------------------------------------------------
class _VisionAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.proj(q + k + v)


class _TextAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        return self.out_proj(q + k + v)


class _Block(nn.Module):
    def __init__(self, dim, text=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = _TextAttn(dim) if text else _VisionAttn(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x + self.mlp(x)


class MockIV2Adapter(BaseCLIPAdapter):
    """Video adapter whose blocks match InternVideo2/MobileCLIP attention naming."""

    def __init__(self, dim=32, n_vision=6, n_text=4, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.vision_blocks = nn.ModuleList([_Block(dim) for _ in range(n_vision)])
        self.text_blocks = nn.ModuleList([_Block(dim, text=True) for _ in range(n_text)])
        self.vision_head = nn.Linear(dim, dim)

    def get_encoder_blocks(self, tower):
        if tower == 'vision':
            return list(self.vision_blocks)
        if tower == 'text':
            return list(self.text_blocks)
        raise ValueError(f"Unknown tower: {tower}")

    def encode_image(self, image, normalize=True):
        x = image
        for b in self.vision_blocks:
            x = b(x)
        x = self.vision_head(x.mean(dim=1))
        return F.normalize(x, dim=-1) if normalize else x

    def encode_text(self, text, normalize=True):
        x = text.float() if not isinstance(text, dict) else text['input_ids'].float()
        if x.dim() == 2:
            x = x.unsqueeze(1)
        for b in self.text_blocks:
            x = b(x)
        x = x.mean(dim=1)
        return F.normalize(x, dim=-1) if normalize else x

    def vision_named_parameters(self):
        yield from self.vision_blocks.named_parameters(prefix='vision_blocks')
        yield from self.vision_head.named_parameters(prefix='vision_head')

    def text_named_parameters(self):
        yield from self.text_blocks.named_parameters(prefix='text_blocks')


@pytest.mark.multimodal_unit
class TestVideoCLIPConfigDefaults:
    """The video_clip PEFT/regularization config defaults are correct."""

    def test_disabled_by_default(self):
        peft = VideoCLIPPEFTConfig()
        assert peft.enabled is False
        assert peft.vision.enabled is False and peft.text.enabled is False
        assert VideoCLIPRegularizationConfig().enabled is False

    def test_target_modules_match_internvideo2(self):
        peft = VideoCLIPPEFTConfig()
        assert peft.vision.target_modules == ["qkv", "proj"]
        assert peft.text.target_modules == ["qkv_proj", "out_proj"]


@pytest.mark.multimodal_unit
class TestGetEncoderBlocks:
    """Adapter get_encoder_blocks contract."""

    def test_base_raises_not_implemented(self):
        class _Bare(BaseCLIPAdapter):
            def encode_image(self, image, normalize=True):
                return image

            def encode_text(self, text, normalize=True):
                return text

            def vision_named_parameters(self):
                yield from ()

            def text_named_parameters(self):
                yield from ()

        with pytest.raises(NotImplementedError):
            _Bare().get_encoder_blocks('vision')

    def test_mock_returns_blocks(self):
        m = MockIV2Adapter(n_vision=6, n_text=4)
        assert len(m.get_encoder_blocks('vision')) == 6
        assert len(m.get_encoder_blocks('text')) == 4
        with pytest.raises(ValueError):
            m.get_encoder_blocks('audio')


@pytest.mark.multimodal_unit
class TestInjectLoRAVideoCLIP:
    """inject_lora driven by the real VideoCLIPPEFTConfig defaults."""

    def _peft(self, vision=True, text=True, num_last_blocks=3):
        peft = VideoCLIPPEFTConfig()
        peft.enabled = True
        peft.vision.enabled = vision
        peft.vision.num_last_blocks = num_last_blocks
        peft.text.enabled = text
        peft.text.num_last_blocks = num_last_blocks
        return peft

    def test_wraps_nonzero_modules_both_towers(self):
        m = MockIV2Adapter(n_vision=6, n_text=4)
        stats = inject_lora(m, self._peft(num_last_blocks=2))
        # 2 vision blocks x {qkv, proj} + 2 text blocks x {qkv_proj, out_proj} = 8
        assert len(stats['injected_modules']) == 8
        assert stats['lora_params'] > 0

    def test_only_lora_and_logit_trainable(self):
        m = MockIV2Adapter()
        stats = inject_lora(m, self._peft())
        assert m.logit_scale.requires_grad
        assert m.logit_bias.requires_grad
        assert stats['logit_trainable_params'] == 2
        leftover = [
            n for n, p in m.named_parameters()
            if p.requires_grad and 'lora_A' not in n and 'lora_B' not in n
            and 'logit' not in n
        ]
        assert leftover == []

    def test_disabled_tower_not_wrapped(self):
        m = MockIV2Adapter()
        inject_lora(m, self._peft(vision=True, text=False))
        text_lora = [
            mod for b in m.get_encoder_blocks('text')
            for mod in b.modules() if isinstance(mod, LoRALinear)
        ]
        assert text_lora == []

    def test_merge_removes_all_lora(self):
        m = MockIV2Adapter()
        inject_lora(m, self._peft())
        assert any(isinstance(mod, LoRALinear) for mod in m.modules())
        merge_lora(m)
        assert not any(isinstance(mod, LoRALinear) for mod in m.modules())


@pytest.mark.multimodal_unit
class TestPreservationLossVideoCLIP:
    """build_preservation_loss honors the regularization toggle."""

    def test_disabled_returns_none(self):
        reg = VideoCLIPRegularizationConfig()
        assert build_preservation_loss(MockIV2Adapter(), reg) is None

    def test_enabled_builds_frozen_teacher(self):
        reg = VideoCLIPRegularizationConfig()
        reg.enabled = True
        pres = build_preservation_loss(MockIV2Adapter(), reg)
        assert pres is not None
        assert all(not p.requires_grad for p in pres.teacher.parameters())


@pytest.mark.multimodal_unit
class TestConfigCopiesIdentical:
    """The tao-pytorch and tao-core video_clip config schemas must not drift."""

    def test_experiment_config_fields_match(self):
        from dataclasses import fields
        from nvidia_tao_pytorch.config.video_clip.default_config import (
            VideoCLIPExperimentConfig as PtCfg,
        )
        # tao-core only grows config.video_clip once NVIDIA-TAO/tao-core#30
        # lands. Until then the parity contract cannot be evaluated at all,
        # which is a different thing from the two schemas having drifted --
        # skip rather than report a missing package as a drift failure. Real
        # drift still fails, and this un-skips itself once tao-core ships it.
        core_cfg_mod = pytest.importorskip(
            "nvidia_tao_core.config.video_clip.default_config",
            reason="tao-core does not ship config.video_clip yet "
                   "(NVIDIA-TAO/tao-core#30); parity is unevaluable, not broken",
        )
        CoreCfg = core_cfg_mod.VideoCLIPExperimentConfig
        assert {f.name for f in fields(PtCfg)} == {f.name for f in fields(CoreCfg)}
        # PEFT plumbing present in both
        assert 'peft' in {f.name for f in fields(PtCfg)}
        assert 'regularization' in {f.name for f in fields(CoreCfg)}
