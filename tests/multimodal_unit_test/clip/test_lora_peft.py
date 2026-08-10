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

"""Unit tests for CLIP LoRA PEFT and preservation losses."""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.multimodal.clip.model.adapters.base import (
    BaseCLIPAdapter,
)
from nvidia_tao_pytorch.multimodal.clip.model.lora import (
    LoRALinear,
    inject_lora,
    merge_lora,
    _match_target,
)
from nvidia_tao_pytorch.multimodal.clip.model.preservation_loss import (
    PreservationLoss,
    build_preservation_loss,
)


# ---------------------------------------------------------------------------
# Fixtures: mock adapter with transformer-like blocks
# ---------------------------------------------------------------------------


class MockAttention(nn.Module):
    """Mock attention with separate q/k/v/out projections (SigLIP2-style)."""

    def __init__(self, dim=64):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.out_proj(q + k + v)


class MockFusedAttention(nn.Module):
    """Mock attention with fused qkv projection (RADIO-style)."""

    def __init__(self, dim=64):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        return self.proj(q + k + v)


class MockBlock(nn.Module):
    """Mock transformer block."""

    def __init__(self, dim=64, fused=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MockFusedAttention(dim) if fused else MockAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MockTransformerAdapter(BaseCLIPAdapter):
    """Mock CLIP adapter with transformer blocks for testing LoRA injection."""

    def __init__(self, dim=64, num_blocks=6, fused=False, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.vision_blocks = nn.ModuleList(
            [MockBlock(dim, fused=fused) for _ in range(num_blocks)]
        )
        self.text_blocks = nn.ModuleList(
            [MockBlock(dim, fused=False) for _ in range(num_blocks)]
        )
        self.vision_head = nn.Linear(dim, dim)
        self.text_head = nn.Linear(dim, dim)

    def get_encoder_blocks(self, tower):
        if tower == 'vision':
            return list(self.vision_blocks)
        elif tower == 'text':
            return list(self.text_blocks)
        raise ValueError(f"Unknown tower: {tower}")

    def encode_image(self, image, normalize=True):
        x = image
        for block in self.vision_blocks:
            x = block(x)
        x = self.vision_head(x.mean(dim=1))
        if normalize:
            x = F.normalize(x, dim=-1)
        return x

    def encode_text(self, text, normalize=True):
        if isinstance(text, dict):
            x = text['input_ids'].float()
        else:
            x = text.float()
        if x.dim() == 2:
            x = x.unsqueeze(1)
        for block in self.text_blocks:
            x = block(x)
        x = self.text_head(x.mean(dim=1))
        if normalize:
            x = F.normalize(x, dim=-1)
        return x

    def vision_named_parameters(self):
        for name, param in self.vision_blocks.named_parameters():
            yield f'vision_blocks.{name}', param
        for name, param in self.vision_head.named_parameters():
            yield f'vision_head.{name}', param

    def text_named_parameters(self):
        for name, param in self.text_blocks.named_parameters():
            yield f'text_blocks.{name}', param
        for name, param in self.text_head.named_parameters():
            yield f'text_head.{name}', param


class MockPEFTConfig:
    """Mock CLIPPEFTConfig."""

    def __init__(self, enabled=True, vision_enabled=True, text_enabled=False,
                 target_modules=None, num_last_blocks=3, rank=4, alpha=8,
                 dropout=0.0):
        self.enabled = enabled
        self.method = 'lora'
        self.vision = MockLoRATargetConfig(
            enabled=vision_enabled,
            target_modules=target_modules or ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
            num_last_blocks=num_last_blocks,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        self.text = MockLoRATargetConfig(
            enabled=text_enabled,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj'],
            num_last_blocks=2,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )


class MockLoRATargetConfig:
    """Mock CLIPLoRATargetConfig."""

    def __init__(self, enabled=True, target_modules=None, num_last_blocks=3,
                 rank=4, alpha=8, dropout=0.0):
        self.enabled = enabled
        self.target_modules = target_modules or ['q_proj', 'k_proj', 'v_proj', 'out_proj']
        self.num_last_blocks = num_last_blocks
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout


class MockRegConfig:
    """Mock CLIPRegularizationConfig."""

    def __init__(self, enabled=True, embedding_mse_weight=0.05,
                 cosine_weight=0.05, similarity_weight=0.10):
        self.enabled = enabled
        self.embedding_mse_weight = embedding_mse_weight
        self.cosine_weight = cosine_weight
        self.similarity_weight = similarity_weight


# ===========================================================================
# LoRALinear tests
# ===========================================================================

@pytest.mark.multimodal_unit
class TestLoRALinear:
    """Test LoRALinear module."""

    def test_output_shape_matches_original(self):
        """Test that LoRALinear output shape matches the original Linear."""
        original = nn.Linear(64, 128)
        lora = LoRALinear(original, rank=4, alpha=8, dropout=0.0)

        x = torch.randn(2, 10, 64)
        out = lora(x)
        assert out.shape == (2, 10, 128)

    def test_original_weights_frozen(self):
        """Test that original Linear weights are frozen after wrapping."""
        original = nn.Linear(64, 128)
        lora = LoRALinear(original, rank=4, alpha=8)

        assert not lora.original.weight.requires_grad
        assert not lora.original.bias.requires_grad

    def test_lora_params_trainable(self):
        """Test that LoRA A and B matrices are trainable."""
        original = nn.Linear(64, 128)
        lora = LoRALinear(original, rank=4, alpha=8)

        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_lora_a_shape(self):
        """Test LoRA A matrix shape: (rank, in_features)."""
        original = nn.Linear(64, 128)
        lora = LoRALinear(original, rank=4, alpha=8)

        assert lora.lora_A.shape == (4, 64)

    def test_lora_b_shape(self):
        """Test LoRA B matrix shape: (out_features, rank)."""
        original = nn.Linear(64, 128)
        lora = LoRALinear(original, rank=4, alpha=8)

        assert lora.lora_B.shape == (128, 4)

    def test_lora_b_initialized_zero(self):
        """Test that LoRA B is zero-initialized (so initial LoRA output is 0)."""
        original = nn.Linear(64, 128)
        lora = LoRALinear(original, rank=4, alpha=8)

        assert torch.all(lora.lora_B == 0)

    def test_initial_output_matches_original(self):
        """Test that initial LoRALinear output matches original (B=0)."""
        original = nn.Linear(64, 128)
        x = torch.randn(2, 10, 64)

        with torch.no_grad():
            expected = original(x)

        lora = LoRALinear(original, rank=4, alpha=8, dropout=0.0)
        with torch.no_grad():
            actual = lora(x)

        assert torch.allclose(actual, expected, atol=1e-6)

    def test_lora_residual_nonzero_after_training(self):
        """Test that LoRA produces a nonzero residual after B is non-zero."""
        original = nn.Linear(64, 128)
        lora = LoRALinear(original, rank=4, alpha=8, dropout=0.0)

        # Manually set B to non-zero
        with torch.no_grad():
            lora.lora_B.fill_(0.1)

        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            base_out = original(x)
            lora_out = lora(x)

        # Should differ
        assert not torch.allclose(lora_out, base_out, atol=1e-6)

    def test_scaling_factor(self):
        """Test that scaling = alpha / rank."""
        original = nn.Linear(64, 128)
        lora = LoRALinear(original, rank=4, alpha=16)
        assert lora.scaling == 4.0  # 16 / 4

    def test_merge_produces_plain_linear(self):
        """Test that merge returns the original nn.Linear with updated weights."""
        original = nn.Linear(64, 128)
        lora = LoRALinear(original, rank=4, alpha=8, dropout=0.0)

        # Set non-zero LoRA weights
        with torch.no_grad():
            lora.lora_A.normal_()
            lora.lora_B.normal_()

        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            pre_merge_out = lora(x)

        merged = lora.merge()
        assert isinstance(merged, nn.Linear)

        with torch.no_grad():
            post_merge_out = merged(x)

        assert torch.allclose(pre_merge_out, post_merge_out, atol=1e-5)

    def test_dropout_zero_is_identity(self):
        """Test that dropout=0 uses Identity (no-op)."""
        original = nn.Linear(64, 128)
        lora = LoRALinear(original, rank=4, alpha=8, dropout=0.0)
        assert isinstance(lora.lora_dropout, nn.Identity)

    def test_dropout_nonzero_is_dropout(self):
        """Test that dropout>0 uses Dropout."""
        original = nn.Linear(64, 128)
        lora = LoRALinear(original, rank=4, alpha=8, dropout=0.1)
        assert isinstance(lora.lora_dropout, nn.Dropout)

    def test_no_bias_original(self):
        """Test LoRALinear works with bias-free original Linear."""
        original = nn.Linear(64, 128, bias=False)
        lora = LoRALinear(original, rank=4, alpha=8)

        x = torch.randn(2, 10, 64)
        out = lora(x)
        assert out.shape == (2, 10, 128)

    def test_gradient_flows_through_lora(self):
        """Test that gradients flow through LoRA parameters."""
        original = nn.Linear(64, 128)
        lora = LoRALinear(original, rank=4, alpha=8, dropout=0.0)

        x = torch.randn(2, 10, 64)
        out = lora(x)
        loss = out.sum()
        loss.backward()

        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None
        # Original weights should not have gradients
        assert original.weight.grad is None


# ===========================================================================
# _match_target tests
# ===========================================================================

@pytest.mark.multimodal_unit
class TestMatchTarget:
    """Test _match_target helper."""

    def test_matches_leaf_name(self):
        """Test matching on the leaf component of a dotted path."""
        assert _match_target('self_attn.q_proj', ['q_proj', 'k_proj'])

    def test_no_match(self):
        """Test non-matching module name."""
        assert not _match_target('self_attn.q_proj', ['fc1', 'fc2'])

    def test_matches_simple_name(self):
        """Test matching on a simple (non-dotted) name."""
        assert _match_target('qkv', ['qkv', 'proj'])

    def test_no_partial_match(self):
        """Test that partial matches do not count (q_proj != proj)."""
        assert not _match_target('attn.q_proj', ['proj'])

    def test_empty_targets(self):
        """Test that empty target list never matches."""
        assert not _match_target('q_proj', [])


# ===========================================================================
# inject_lora tests
# ===========================================================================

@pytest.mark.multimodal_unit
class TestInjectLoRA:
    """Test inject_lora function."""

    @pytest.fixture
    def adapter(self):
        """Create mock adapter with 6 blocks, separate q/k/v/out projections."""
        return MockTransformerAdapter(dim=64, num_blocks=6, fused=False)

    @pytest.fixture
    def fused_adapter(self):
        """Create mock adapter with fused qkv projections."""
        return MockTransformerAdapter(dim=64, num_blocks=6, fused=True)

    def test_disabled_returns_none(self, adapter):
        """Test that disabled PEFT returns None."""
        cfg = MockPEFTConfig(enabled=False)
        result = inject_lora(adapter, cfg)
        assert result is None

    def test_freezes_all_backbone_params(self, adapter):
        """Test that all backbone parameters are frozen after injection."""
        cfg = MockPEFTConfig(enabled=True, vision_enabled=True)
        inject_lora(adapter, cfg)

        for name, param in adapter.named_parameters():
            if 'lora_' in name or 'logit_' in name:
                assert param.requires_grad, f"{name} should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_lora_params_are_trainable(self, adapter):
        """Test that LoRA A/B parameters are trainable."""
        cfg = MockPEFTConfig(enabled=True, vision_enabled=True)
        inject_lora(adapter, cfg)

        lora_params = [
            (n, p) for n, p in adapter.named_parameters()
            if 'lora_' in n
        ]
        assert len(lora_params) > 0
        for name, param in lora_params:
            assert param.requires_grad, f"{name} should be trainable"

    def test_logit_params_unfrozen(self, adapter):
        """Test that logit_scale and logit_bias remain trainable."""
        cfg = MockPEFTConfig(enabled=True, vision_enabled=True)
        inject_lora(adapter, cfg)

        assert adapter.logit_scale.requires_grad
        assert adapter.logit_bias.requires_grad

    def test_returns_stats_dict(self, adapter):
        """Test that inject_lora returns statistics dictionary."""
        cfg = MockPEFTConfig(enabled=True, vision_enabled=True)
        stats = inject_lora(adapter, cfg)

        assert 'total_params' in stats
        assert 'trainable_params' in stats
        assert 'lora_params' in stats
        assert 'injected_modules' in stats
        assert stats['lora_params'] > 0
        assert stats['trainable_params'] < stats['total_params']

    def test_vision_only_injection(self, adapter):
        """Test injecting LoRA only into vision tower."""
        cfg = MockPEFTConfig(
            enabled=True, vision_enabled=True, text_enabled=False,
        )
        stats = inject_lora(adapter, cfg)

        # All injected modules should be vision
        for tower, _ in stats['injected_modules']:
            assert tower == 'vision'

    def test_both_towers_injection(self, adapter):
        """Test injecting LoRA into both vision and text towers."""
        cfg = MockPEFTConfig(
            enabled=True, vision_enabled=True, text_enabled=True,
        )
        stats = inject_lora(adapter, cfg)

        towers = {tower for tower, _ in stats['injected_modules']}
        assert 'vision' in towers
        assert 'text' in towers

    def test_num_last_blocks_respected(self, adapter):
        """Test that only the last N blocks are adapted."""
        cfg = MockPEFTConfig(
            enabled=True, vision_enabled=True, num_last_blocks=2,
        )
        stats = inject_lora(adapter, cfg)

        # 6 blocks total, last 2 adapted, each has 4 projections
        block_indices = set()
        for _, mod_name in stats['injected_modules']:
            # Parse block index from "block[N].attn.q_proj"
            idx = int(mod_name.split('[')[1].split(']')[0])
            block_indices.add(idx)

        assert block_indices == {4, 5}  # last 2 of 6

    def test_num_last_blocks_zero_adapts_all(self, adapter):
        """Test that num_last_blocks=0 adapts all blocks."""
        cfg = MockPEFTConfig(
            enabled=True, vision_enabled=True, num_last_blocks=0,
        )
        stats = inject_lora(adapter, cfg)

        block_indices = set()
        for _, mod_name in stats['injected_modules']:
            idx = int(mod_name.split('[')[1].split(']')[0])
            block_indices.add(idx)

        assert block_indices == {0, 1, 2, 3, 4, 5}

    def test_target_modules_filtering(self, adapter):
        """Test that only matching modules are adapted."""
        cfg = MockPEFTConfig(
            enabled=True, vision_enabled=True,
            target_modules=['q_proj', 'out_proj'],  # only 2 of 4
            num_last_blocks=1,
        )
        stats = inject_lora(adapter, cfg)

        mod_names = [name for _, name in stats['injected_modules']]
        assert len(mod_names) == 2
        assert any('q_proj' in n for n in mod_names)
        assert any('out_proj' in n for n in mod_names)
        assert not any('k_proj' in n for n in mod_names)
        assert not any('v_proj' in n for n in mod_names)

    def test_fused_qkv_injection(self, fused_adapter):
        """Test LoRA injection into fused qkv Linear."""
        cfg = MockPEFTConfig(
            enabled=True, vision_enabled=True,
            target_modules=['qkv', 'proj'],
            num_last_blocks=2,
        )
        stats = inject_lora(fused_adapter, cfg)

        mod_names = [name for _, name in stats['injected_modules']]
        assert len(mod_names) == 4  # 2 blocks * (qkv + proj)
        assert any('qkv' in n for n in mod_names)
        assert any('proj' in n for n in mod_names)

    def test_forward_still_works_after_injection(self, adapter):
        """Test that the adapter forward pass works after LoRA injection."""
        cfg = MockPEFTConfig(enabled=True, vision_enabled=True)
        inject_lora(adapter, cfg)

        image = torch.randn(2, 10, 64)
        text = {'input_ids': torch.randn(2, 64)}

        result = adapter(image=image, text=text)
        assert len(result) == 4
        img_feat, txt_feat, scale, bias = result
        assert img_feat.shape == (2, 64)
        assert txt_feat.shape == (2, 64)

    def test_trainable_fraction_is_small(self, adapter):
        """Test that trainable params are a small fraction of total."""
        cfg = MockPEFTConfig(
            enabled=True, vision_enabled=True, rank=4, num_last_blocks=3,
        )
        stats = inject_lora(adapter, cfg)

        fraction = stats['trainable_params'] / stats['total_params']
        # With rank=4 on a small model, fraction should be well under 50%
        assert fraction < 0.5

    def test_rank_affects_param_count(self, adapter):
        """Test that higher rank means more LoRA parameters."""
        adapter_r4 = MockTransformerAdapter(dim=64, num_blocks=6)
        adapter_r16 = MockTransformerAdapter(dim=64, num_blocks=6)

        cfg_r4 = MockPEFTConfig(enabled=True, vision_enabled=True, rank=4)
        cfg_r16 = MockPEFTConfig(enabled=True, vision_enabled=True, rank=16)

        stats_r4 = inject_lora(adapter_r4, cfg_r4)
        stats_r16 = inject_lora(adapter_r16, cfg_r16)

        assert stats_r16['lora_params'] > stats_r4['lora_params']


# ===========================================================================
# merge_lora tests
# ===========================================================================

@pytest.mark.multimodal_unit
class TestMergeLoRA:
    """Test merge_lora function."""

    def test_merge_removes_lora_modules(self):
        """Test that merge replaces all LoRALinear with nn.Linear."""
        adapter = MockTransformerAdapter(dim=64, num_blocks=4)
        cfg = MockPEFTConfig(enabled=True, vision_enabled=True, num_last_blocks=2)
        inject_lora(adapter, cfg)

        # Verify LoRALinear modules exist
        lora_count_before = sum(
            1 for _, m in adapter.named_modules() if isinstance(m, LoRALinear)
        )
        assert lora_count_before > 0

        merged = merge_lora(adapter)
        assert merged == lora_count_before

        # Verify no LoRALinear modules remain
        lora_count_after = sum(
            1 for _, m in adapter.named_modules() if isinstance(m, LoRALinear)
        )
        assert lora_count_after == 0

    def test_merge_preserves_output(self):
        """Test that merged model produces same output as pre-merge."""
        adapter = MockTransformerAdapter(dim=64, num_blocks=4)
        cfg = MockPEFTConfig(
            enabled=True, vision_enabled=True, num_last_blocks=2,
            dropout=0.0,
        )
        inject_lora(adapter, cfg)

        # Set non-zero LoRA weights
        for m in adapter.modules():
            if isinstance(m, LoRALinear):
                with torch.no_grad():
                    m.lora_A.normal_(0, 0.01)
                    m.lora_B.normal_(0, 0.01)

        x = torch.randn(2, 10, 64)
        adapter.eval()
        with torch.no_grad():
            pre_merge = adapter.encode_image(x, normalize=False)

        merge_lora(adapter)

        with torch.no_grad():
            post_merge = adapter.encode_image(x, normalize=False)

        assert torch.allclose(pre_merge, post_merge, atol=1e-5)

    def test_merge_on_no_lora_model(self):
        """Test that merge_lora on a model with no LoRA returns 0."""
        adapter = MockTransformerAdapter(dim=64, num_blocks=4)
        merged = merge_lora(adapter)
        assert merged == 0


# ===========================================================================
# PreservationLoss tests
# ===========================================================================

@pytest.mark.multimodal_unit
class TestPreservationLoss:
    """Test PreservationLoss computation."""

    @pytest.fixture
    def teacher_and_student(self):
        """Create teacher (frozen copy) and student adapters."""
        student = MockTransformerAdapter(dim=64, num_blocks=4)
        teacher = copy.deepcopy(student)
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        return student, teacher

    @pytest.fixture
    def reg_config(self):
        """Create regularization config."""
        return MockRegConfig(
            enabled=True,
            embedding_mse_weight=0.05,
            cosine_weight=0.05,
            similarity_weight=0.10,
        )

    def test_returns_all_loss_keys(self, teacher_and_student, reg_config):
        """Test that all expected loss keys are returned."""
        student, teacher = teacher_and_student
        pres_loss = PreservationLoss(teacher, reg_config)

        image = torch.randn(4, 10, 64)
        text = {'input_ids': torch.randn(4, 64)}

        with torch.no_grad():
            s_img = student.encode_image(image)
            s_txt = student.encode_text(text)

        losses = pres_loss(s_img, s_txt, image, text)

        assert 'preservation_total' in losses
        assert 'embedding_mse' in losses
        assert 'cosine' in losses
        assert 'similarity' in losses

    def test_zero_loss_when_identical(self, teacher_and_student, reg_config):
        """Test that losses are ~0 when student equals teacher (no training)."""
        student, teacher = teacher_and_student
        pres_loss = PreservationLoss(teacher, reg_config)

        image = torch.randn(4, 10, 64)
        text = {'input_ids': torch.randn(4, 64)}

        student.eval()
        with torch.no_grad():
            s_img = student.encode_image(image)
            s_txt = student.encode_text(text)

        losses = pres_loss(s_img, s_txt, image, text)

        assert losses['embedding_mse'].item() < 1e-6
        assert losses['cosine'].item() < 1e-6
        assert losses['similarity'].item() < 1e-6
        assert losses['preservation_total'].item() < 1e-6

    def test_nonzero_loss_after_perturbation(self, teacher_and_student, reg_config):
        """Test that losses are nonzero when student diverges from teacher."""
        student, teacher = teacher_and_student
        pres_loss = PreservationLoss(teacher, reg_config)

        # Perturb student weights so outputs differ
        with torch.no_grad():
            for p in student.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        image = torch.randn(4, 10, 64)
        text = {'input_ids': torch.randn(4, 64)}

        student.eval()
        with torch.no_grad():
            s_img = student.encode_image(image)
            s_txt = student.encode_text(text)

        losses = pres_loss(s_img, s_txt, image, text)

        assert losses['embedding_mse'].item() > 1e-6
        assert losses['cosine'].item() > 1e-6
        assert losses['preservation_total'].item() > 0

    def test_weight_scaling(self):
        """Test that loss weights are correctly applied to the total."""
        student = MockTransformerAdapter(dim=64, num_blocks=4)
        teacher = copy.deepcopy(student)
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()

        # Perturb student
        with torch.no_grad():
            for p in student.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        # Use weight=1.0 for each to check total = sum of components
        cfg = MockRegConfig(
            enabled=True,
            embedding_mse_weight=1.0,
            cosine_weight=1.0,
            similarity_weight=1.0,
        )
        pres_loss = PreservationLoss(teacher, cfg)

        image = torch.randn(4, 10, 64)
        text = {'input_ids': torch.randn(4, 64)}

        student.eval()
        with torch.no_grad():
            s_img = student.encode_image(image)
            s_txt = student.encode_text(text)

        losses = pres_loss(s_img, s_txt, image, text)

        expected_total = (
            losses['embedding_mse'] + losses['cosine'] + losses['similarity']
        )
        assert torch.allclose(
            losses['preservation_total'], expected_total, atol=1e-6
        )

    def test_teacher_is_frozen(self, teacher_and_student, reg_config):
        """Test that teacher parameters have no gradients."""
        _, teacher = teacher_and_student
        pres_loss = PreservationLoss(teacher, reg_config)

        for param in pres_loss.teacher.parameters():
            assert not param.requires_grad


# ===========================================================================
# build_preservation_loss tests
# ===========================================================================

@pytest.mark.multimodal_unit
class TestBuildPreservationLoss:
    """Test build_preservation_loss factory function."""

    def test_disabled_returns_none(self):
        """Test that disabled config returns None."""
        model = MockTransformerAdapter(dim=64, num_blocks=4)
        cfg = MockRegConfig(enabled=False)
        result = build_preservation_loss(model, cfg)
        assert result is None

    def test_enabled_returns_preservation_loss(self):
        """Test that enabled config returns a PreservationLoss instance."""
        model = MockTransformerAdapter(dim=64, num_blocks=4)
        cfg = MockRegConfig(enabled=True)
        result = build_preservation_loss(model, cfg)
        assert isinstance(result, PreservationLoss)

    def test_teacher_is_independent_copy(self):
        """Test that teacher weights are independent of the original model."""
        model = MockTransformerAdapter(dim=64, num_blocks=4)
        cfg = MockRegConfig(enabled=True)
        pres_loss = build_preservation_loss(model, cfg)

        # Modify the original model
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)

        # Teacher should be unaffected
        for (_, p_teacher), (_, p_model) in zip(
            pres_loss.teacher.named_parameters(),
            model.named_parameters()
        ):
            assert not torch.equal(p_teacher, p_model)


# ===========================================================================
# Integration: inject_lora + merge_lora round-trip
# ===========================================================================

@pytest.mark.multimodal_unit
class TestLoRARoundTrip:
    """Test full LoRA lifecycle: inject -> train -> merge."""

    def test_inject_train_merge_roundtrip(self):
        """Test inject -> simulate training -> merge preserves correctness."""
        adapter = MockTransformerAdapter(dim=64, num_blocks=4, fused=False)
        cfg = MockPEFTConfig(
            enabled=True, vision_enabled=True,
            num_last_blocks=2, rank=4, alpha=8, dropout=0.0,
        )

        # Inject LoRA
        stats = inject_lora(adapter, cfg)
        assert stats['lora_params'] > 0

        # Simulate a training step (update LoRA params)
        x = torch.randn(2, 10, 64)
        adapter.train()
        out = adapter.encode_image(x, normalize=False)
        loss = out.sum()
        loss.backward()

        # Check gradients exist on LoRA params
        for n, p in adapter.named_parameters():
            if 'lora_' in n and p.requires_grad:
                assert p.grad is not None, f"No grad for {n}"

        # Record output before merge
        adapter.eval()
        with torch.no_grad():
            pre_merge = adapter.encode_image(x, normalize=False)

        # Merge
        merged_count = merge_lora(adapter)
        assert merged_count > 0

        # Output after merge should match
        with torch.no_grad():
            post_merge = adapter.encode_image(x, normalize=False)

        assert torch.allclose(pre_merge, post_merge, atol=1e-5)

    def test_fused_qkv_inject_train_merge(self):
        """Test full round-trip with fused QKV attention (RADIO-style)."""
        adapter = MockTransformerAdapter(dim=64, num_blocks=4, fused=True)
        cfg = MockPEFTConfig(
            enabled=True, vision_enabled=True,
            target_modules=['qkv', 'proj'],
            num_last_blocks=2, rank=4, alpha=8, dropout=0.0,
        )

        stats = inject_lora(adapter, cfg)
        assert stats['lora_params'] > 0

        # Forward + backward
        x = torch.randn(2, 10, 64)
        adapter.train()
        out = adapter.encode_image(x, normalize=False)
        out.sum().backward()

        # Merge and verify
        adapter.eval()
        with torch.no_grad():
            pre_merge = adapter.encode_image(x, normalize=False)

        merge_lora(adapter)

        with torch.no_grad():
            post_merge = adapter.encode_image(x, normalize=False)

        assert torch.allclose(pre_merge, post_merge, atol=1e-5)
