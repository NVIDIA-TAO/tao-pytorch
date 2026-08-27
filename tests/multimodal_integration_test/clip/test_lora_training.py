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

"""Integration tests for CLIP LoRA PEFT training and export flows."""

import copy
import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from nvidia_tao_pytorch.multimodal.clip.model.adapters.base import (
    BaseCLIPAdapter,
)
from nvidia_tao_pytorch.multimodal.clip.model.lora import (
    LoRALinear,
    inject_lora,
    merge_lora,
)
from nvidia_tao_pytorch.multimodal.clip.model.preservation_loss import (
    PreservationLoss,
    build_preservation_loss,
)
from nvidia_tao_pytorch.multimodal.clip.utils.utils import build_optimizer


# ---------------------------------------------------------------------------
# Mock model with transformer blocks for integration tests
# ---------------------------------------------------------------------------


class MockAttention(nn.Module):
    """Mock attention with separate q/k/v/out projections."""

    def __init__(self, dim=128):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.out_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class MockBlock(nn.Module):
    """Mock transformer block."""

    def __init__(self, dim=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MockAttention(dim)
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


class MockTransformerCLIP(BaseCLIPAdapter):
    """Mock CLIP adapter with transformer blocks for LoRA integration testing.

    Uses a simple patch embedding + transformer blocks architecture
    that supports get_encoder_blocks() for LoRA injection.
    """

    def __init__(self, dim=128, num_vision_blocks=4, num_text_blocks=4,
                 image_size=32, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.image_size = image_size

        # Vision: flatten image -> project -> transformer blocks -> head
        self.vision_proj = nn.Linear(3 * image_size * image_size, dim)
        self.vision_blocks = nn.ModuleList(
            [MockBlock(dim) for _ in range(num_vision_blocks)]
        )
        self.vision_head = nn.Linear(dim, dim)

        # Text: project tokens -> transformer blocks -> head
        self.text_proj = nn.Linear(64, dim)
        self.text_blocks = nn.ModuleList(
            [MockBlock(dim) for _ in range(num_text_blocks)]
        )
        self.text_head = nn.Linear(dim, dim)

    def get_encoder_blocks(self, tower):
        if tower == 'vision':
            return list(self.vision_blocks)
        elif tower == 'text':
            return list(self.text_blocks)
        raise ValueError(f"Unknown tower: {tower}")

    def encode_image(self, image, normalize=True):
        if isinstance(image, dict):
            image = image['pixel_values']
        b = image.shape[0]
        x = self.vision_proj(image.view(b, -1)).unsqueeze(1)  # (B, 1, D)
        for block in self.vision_blocks:
            x = block(x)
        x = self.vision_head(x.squeeze(1))
        if normalize:
            x = F.normalize(x, dim=-1)
        return x

    def encode_text(self, text, normalize=True):
        if isinstance(text, dict):
            text = text['input_ids']
        x = self.text_proj(text.float()).unsqueeze(1)  # (B, 1, D)
        for block in self.text_blocks:
            x = block(x)
        x = self.text_head(x.squeeze(1))
        if normalize:
            x = F.normalize(x, dim=-1)
        return x

    def vision_named_parameters(self):
        for name, param in self.vision_proj.named_parameters():
            yield f'vision_proj.{name}', param
        for name, param in self.vision_blocks.named_parameters():
            yield f'vision_blocks.{name}', param
        for name, param in self.vision_head.named_parameters():
            yield f'vision_head.{name}', param

    def text_named_parameters(self):
        for name, param in self.text_proj.named_parameters():
            yield f'text_proj.{name}', param
        for name, param in self.text_blocks.named_parameters():
            yield f'text_blocks.{name}', param
        for name, param in self.text_head.named_parameters():
            yield f'text_head.{name}', param

    def set_grad_checkpointing(self, enable=True):
        pass


# ---------------------------------------------------------------------------
# Mock configs
# ---------------------------------------------------------------------------


def _tower_mode(enabled):
    """Map a boolean tower selector onto its explicit tower mode."""
    return 'lora' if enabled else 'frozen'


class MockLoRATargetConfig:
    def __init__(self, mode='lora', target_modules=None,
                 num_last_blocks=2, rank=4, alpha=8, dropout=0.0):
        self.mode = mode
        self.target_modules = target_modules or [
            'q_proj', 'k_proj', 'v_proj', 'out_proj'
        ]
        self.num_last_blocks = num_last_blocks
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout


class MockPEFTConfig:
    def __init__(self, enabled=True, vision_enabled=True, text_enabled=False,
                 **kwargs):
        self.enabled = enabled
        self.method = 'lora'
        self.vision = MockLoRATargetConfig(
            mode=_tower_mode(vision_enabled), **kwargs)
        self.text = MockLoRATargetConfig(
            mode=_tower_mode(text_enabled), **kwargs)


class MockRegConfig:
    def __init__(self, enabled=True, embedding_mse_weight=0.05,
                 cosine_weight=0.05, similarity_weight=0.10):
        self.enabled = enabled
        self.embedding_mse_weight = embedding_mse_weight
        self.cosine_weight = cosine_weight
        self.similarity_weight = similarity_weight


class MockOptimConfig:
    optimizer_type = 'adamw'
    vision_lr = 1e-3
    text_lr = 1e-3
    weight_decay = 0.01
    betas = [0.9, 0.999]
    eps = 1e-8
    warmup_steps = 0
    scheduler = 'constant'


class MockTrainConfig:
    optim = MockOptimConfig()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_batch(batch_size=8, image_size=32, device=None):
    """Create a dummy (image, text) batch."""
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    texts = torch.randn(batch_size, 64, device=device)
    return images, texts


def run_training_steps(model, optimizer, num_steps=5, batch_size=8):
    """Run N training steps and return per-step losses."""
    device = next(model.parameters()).device
    model.train()
    losses = []
    for _ in range(num_steps):
        optimizer.zero_grad()
        images, texts = make_batch(batch_size, device=device)
        img_feat, txt_feat, scale, bias = model(image=images, text=texts)
        logits = scale * (img_feat @ txt_feat.T) + bias
        targets = torch.eye(batch_size, device=device)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


# ===========================================================================
# LoRA training loop integration tests
# ===========================================================================


@pytest.mark.multimodal_integration
class TestLoRATrainingLoop:
    """Test LoRA training loop end-to-end."""

    def test_lora_training_produces_finite_losses(self):
        """Test that LoRA training runs without NaN/Inf losses."""
        model = MockTransformerCLIP(dim=128, num_vision_blocks=4)
        peft_cfg = MockPEFTConfig(enabled=True, vision_enabled=True,
                                  num_last_blocks=2, rank=4)
        inject_lora(model, peft_cfg)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        losses = run_training_steps(model, optimizer, num_steps=10)

        assert all(torch.isfinite(torch.tensor(l)) for l in losses)

    def test_only_lora_params_change_during_training(self):
        """Test that only LoRA parameters are updated during training."""
        model = MockTransformerCLIP(dim=128, num_vision_blocks=4)
        peft_cfg = MockPEFTConfig(enabled=True, vision_enabled=True,
                                  num_last_blocks=2, rank=4)
        inject_lora(model, peft_cfg)

        # Snapshot all parameters
        frozen_snapshot = {}
        lora_snapshot = {}
        for name, param in model.named_parameters():
            if 'lora_' in name or 'logit_' in name:
                lora_snapshot[name] = param.clone().detach()
            else:
                frozen_snapshot[name] = param.clone().detach()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2
        )
        run_training_steps(model, optimizer, num_steps=5)

        # Frozen params must not change
        for name, param in model.named_parameters():
            if name in frozen_snapshot:
                assert torch.equal(param, frozen_snapshot[name]), \
                    f"Frozen param {name} changed during training"

        # LoRA params must change (at least some of them)
        any_lora_changed = False
        for name, param in model.named_parameters():
            if name in lora_snapshot:
                if not torch.equal(param, lora_snapshot[name]):
                    any_lora_changed = True
                    break
        assert any_lora_changed, "No LoRA parameter changed during training"

    def test_lora_loss_decreases_over_training(self):
        """Test that loss generally decreases over training steps."""
        torch.manual_seed(42)
        model = MockTransformerCLIP(dim=128, num_vision_blocks=4)
        peft_cfg = MockPEFTConfig(enabled=True, vision_enabled=True,
                                  num_last_blocks=2, rank=8)
        inject_lora(model, peft_cfg)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2
        )
        losses = run_training_steps(model, optimizer, num_steps=30,
                                    batch_size=16)

        # Average of last 5 should be lower than average of first 5
        avg_first = sum(losses[:5]) / 5
        avg_last = sum(losses[-5:]) / 5
        assert avg_last < avg_first, (
            f"Loss did not decrease: first 5 avg={avg_first:.4f}, "
            f"last 5 avg={avg_last:.4f}"
        )

    def test_dual_tower_lora_training(self):
        """Test LoRA on both vision and text towers simultaneously."""
        model = MockTransformerCLIP(dim=128, num_vision_blocks=4,
                                    num_text_blocks=4)
        peft_cfg = MockPEFTConfig(enabled=True, vision_enabled=True,
                                  text_enabled=True, num_last_blocks=2,
                                  rank=4)
        stats = inject_lora(model, peft_cfg)

        towers = {t for t, _ in stats['injected_modules']}
        assert 'vision' in towers
        assert 'text' in towers

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        losses = run_training_steps(model, optimizer, num_steps=5)
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)

    def test_lora_with_per_tower_optimizer(self):
        """Test LoRA training with the real per-tower optimizer builder."""
        model = MockTransformerCLIP(dim=128, num_vision_blocks=4)
        peft_cfg = MockPEFTConfig(enabled=True, vision_enabled=True,
                                  num_last_blocks=2, rank=4)
        inject_lora(model, peft_cfg)

        train_cfg = MockTrainConfig()
        optimizer = build_optimizer(model, train_cfg)

        # The optimizer should have param groups
        assert len(optimizer.param_groups) > 0

        # Run training
        model.train()
        images, texts = make_batch(8)
        optimizer.zero_grad()
        img_feat, txt_feat, scale, bias = model(image=images, text=texts)
        logits = scale * (img_feat @ txt_feat.T) + bias
        loss = F.binary_cross_entropy_with_logits(logits, torch.eye(8))
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)


# ===========================================================================
# Preservation loss training integration tests
# ===========================================================================


@pytest.mark.multimodal_integration
class TestPreservationLossTraining:
    """Test training with preservation losses enabled."""

    def test_preservation_loss_training_loop(self):
        """Test full training loop with contrastive + preservation losses."""
        model = MockTransformerCLIP(dim=128, num_vision_blocks=4)
        reg_cfg = MockRegConfig(enabled=True)

        # Build preservation loss BEFORE LoRA (teacher = original weights)
        pres_loss = build_preservation_loss(model, reg_cfg)

        # Inject LoRA
        peft_cfg = MockPEFTConfig(enabled=True, vision_enabled=True,
                                  num_last_blocks=2, rank=4)
        inject_lora(model, peft_cfg)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )

        model.train()
        all_losses = []
        for step in range(10):
            optimizer.zero_grad()
            images, texts = make_batch(8)

            # Forward
            img_feat, txt_feat, scale, bias = model(
                image=images, text=texts
            )

            # Contrastive loss
            logits = scale * (img_feat @ txt_feat.T) + bias
            contrastive_loss = F.binary_cross_entropy_with_logits(
                logits, torch.eye(8)
            )

            # Preservation losses
            pres_losses = pres_loss(img_feat, txt_feat, images, texts)
            total_loss = contrastive_loss + pres_losses['preservation_total']

            total_loss.backward()
            optimizer.step()

            all_losses.append({
                'total': total_loss.item(),
                'contrastive': contrastive_loss.item(),
                'embedding_mse': pres_losses['embedding_mse'].item(),
                'cosine': pres_losses['cosine'].item(),
                'similarity': pres_losses['similarity'].item(),
            })

        # All losses should be finite
        for step_losses in all_losses:
            for key, val in step_losses.items():
                assert torch.isfinite(torch.tensor(val)), \
                    f"Non-finite {key} at some step: {val}"

    def test_preservation_loss_constrains_drift(self):
        """Test that preservation losses keep embeddings closer to teacher."""
        torch.manual_seed(42)

        # Train model WITHOUT preservation
        model_no_reg = MockTransformerCLIP(dim=128, num_vision_blocks=4)
        teacher_snapshot = copy.deepcopy(model_no_reg)
        for p in teacher_snapshot.parameters():
            p.requires_grad = False
        teacher_snapshot.eval()

        peft_cfg = MockPEFTConfig(enabled=True, vision_enabled=True,
                                  num_last_blocks=2, rank=4)
        inject_lora(model_no_reg, peft_cfg)
        opt_no_reg = torch.optim.AdamW(
            [p for p in model_no_reg.parameters() if p.requires_grad],
            lr=1e-2,
        )
        run_training_steps(model_no_reg, opt_no_reg, num_steps=20,
                           batch_size=16)

        # Train model WITH preservation
        torch.manual_seed(42)
        model_reg = MockTransformerCLIP(dim=128, num_vision_blocks=4)
        reg_cfg = MockRegConfig(enabled=True, embedding_mse_weight=0.5,
                                cosine_weight=0.5, similarity_weight=0.5)
        pres_loss = build_preservation_loss(model_reg, reg_cfg)

        inject_lora(model_reg, peft_cfg)
        opt_reg = torch.optim.AdamW(
            [p for p in model_reg.parameters() if p.requires_grad],
            lr=1e-2,
        )

        torch.manual_seed(42)
        model_reg.train()
        for _ in range(20):
            opt_reg.zero_grad()
            images, texts = make_batch(16)
            img_feat, txt_feat, scale, bias = model_reg(
                image=images, text=texts
            )
            logits = scale * (img_feat @ txt_feat.T) + bias
            contrastive = F.binary_cross_entropy_with_logits(
                logits, torch.eye(16)
            )
            pres = pres_loss(img_feat, txt_feat, images, texts)
            total = contrastive + pres['preservation_total']
            total.backward()
            opt_reg.step()

        # Measure drift from teacher for both models
        test_images, test_texts = make_batch(16)
        model_no_reg.eval()
        model_reg.eval()
        teacher_snapshot.eval()

        with torch.no_grad():
            teacher_img = teacher_snapshot.encode_image(test_images)
            no_reg_img = model_no_reg.encode_image(test_images)
            reg_img = model_reg.encode_image(test_images)

        drift_no_reg = (
            1.0 - F.cosine_similarity(no_reg_img, teacher_img, dim=-1)
        ).mean().item()
        drift_reg = (
            1.0 - F.cosine_similarity(reg_img, teacher_img, dim=-1)
        ).mean().item()

        # Model with preservation should drift less
        assert drift_reg < drift_no_reg, (
            f"Preservation did not reduce drift: "
            f"with_reg={drift_reg:.4f}, no_reg={drift_no_reg:.4f}"
        )

    def test_preservation_losses_nonzero_after_training(self):
        """Test that individual preservation losses are non-zero after
        some training steps (student has diverged from teacher)."""
        model = MockTransformerCLIP(dim=128, num_vision_blocks=4)
        reg_cfg = MockRegConfig(enabled=True)
        pres_loss = build_preservation_loss(model, reg_cfg)

        peft_cfg = MockPEFTConfig(enabled=True, vision_enabled=True,
                                  num_last_blocks=2, rank=8)
        inject_lora(model, peft_cfg)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2
        )

        # Train a few steps to move away from teacher
        model.train()
        for _ in range(10):
            optimizer.zero_grad()
            images, texts = make_batch(8)
            img_feat, txt_feat, scale, bias = model(
                image=images, text=texts
            )
            logits = scale * (img_feat @ txt_feat.T) + bias
            loss = F.binary_cross_entropy_with_logits(logits, torch.eye(8))
            loss.backward()
            optimizer.step()

        # Now check that preservation losses are non-trivial
        model.eval()
        with torch.no_grad():
            images, texts = make_batch(16)
            img_feat = model.encode_image(images)
            txt_feat = model.encode_text(texts)

        losses = pres_loss(img_feat, txt_feat, images, texts)
        assert losses['embedding_mse'].item() > 1e-6
        assert losses['cosine'].item() > 1e-6


# ===========================================================================
# LoRA merge + export integration tests
# ===========================================================================


@pytest.mark.multimodal_integration
class TestLoRAMergeExport:
    """Test LoRA merge flow (simulating pre-export merge)."""

    def test_merge_after_training_preserves_output(self):
        """Test that merge after training preserves model output exactly."""
        model = MockTransformerCLIP(dim=128, num_vision_blocks=4)
        peft_cfg = MockPEFTConfig(enabled=True, vision_enabled=True,
                                  num_last_blocks=2, rank=4, dropout=0.0)
        inject_lora(model, peft_cfg)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2
        )
        run_training_steps(model, optimizer, num_steps=10)

        # Get output before merge
        model.eval()
        test_images, test_texts = make_batch(4)
        with torch.no_grad():
            pre_img = model.encode_image(test_images, normalize=False)
            pre_txt = model.encode_text(test_texts, normalize=False)

        # Merge
        merged_count = merge_lora(model)
        assert merged_count > 0

        # No LoRALinear modules should remain
        for _, m in model.named_modules():
            assert not isinstance(m, LoRALinear)

        # Get output after merge
        with torch.no_grad():
            post_img = model.encode_image(test_images, normalize=False)
            post_txt = model.encode_text(test_texts, normalize=False)

        assert torch.allclose(pre_img, post_img, atol=1e-5)
        assert torch.allclose(pre_txt, post_txt, atol=1e-5)

    def test_merged_model_still_trainable(self):
        """Test that a merged model can be further fine-tuned (full SFT)."""
        model = MockTransformerCLIP(dim=128, num_vision_blocks=4)
        peft_cfg = MockPEFTConfig(enabled=True, vision_enabled=True,
                                  num_last_blocks=2, rank=4, dropout=0.0)
        inject_lora(model, peft_cfg)
        run_training_steps(
            model,
            torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-3
            ),
            num_steps=5,
        )

        # Merge
        merge_lora(model)

        # Unfreeze all params for full fine-tuning after merge
        for param in model.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = run_training_steps(model, optimizer, num_steps=5)
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)

    def test_merge_dual_tower_lora(self):
        """Test merge with LoRA on both vision and text towers."""
        model = MockTransformerCLIP(dim=128, num_vision_blocks=4,
                                    num_text_blocks=4)
        peft_cfg = MockPEFTConfig(enabled=True, vision_enabled=True,
                                  text_enabled=True, num_last_blocks=2,
                                  rank=4, dropout=0.0)
        inject_lora(model, peft_cfg)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2
        )
        run_training_steps(model, optimizer, num_steps=5)

        model.eval()
        test_images, test_texts = make_batch(4)
        with torch.no_grad():
            pre_out = model(image=test_images, text=test_texts)

        merged = merge_lora(model)
        assert merged > 0

        with torch.no_grad():
            post_out = model(image=test_images, text=test_texts)

        # img features, txt features
        assert torch.allclose(pre_out[0], post_out[0], atol=1e-5)
        assert torch.allclose(pre_out[1], post_out[1], atol=1e-5)


# ===========================================================================
# Checkpoint save/load integration tests
# ===========================================================================


@pytest.mark.multimodal_integration
class TestLoRACheckpoint:
    """Test LoRA model checkpoint save and load."""

    def test_save_and_load_lora_state_dict(self):
        """Test that a LoRA-injected model can be saved and reloaded."""
        model = MockTransformerCLIP(dim=128, num_vision_blocks=4)
        peft_cfg = MockPEFTConfig(enabled=True, vision_enabled=True,
                                  num_last_blocks=2, rank=4, dropout=0.0)
        inject_lora(model, peft_cfg)

        # Train a few steps so LoRA weights are non-trivial
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2
        )
        run_training_steps(model, optimizer, num_steps=5)

        # Get reference output
        model.eval()
        test_images, test_texts = make_batch(4)
        with torch.no_grad():
            ref_img = model.encode_image(test_images, normalize=False)

        # Save state dict
        with tempfile.NamedTemporaryFile(suffix='.pth') as f:
            torch.save(model.state_dict(), f.name)

            # Build fresh model with same LoRA config
            model2 = MockTransformerCLIP(dim=128, num_vision_blocks=4)
            inject_lora(model2, peft_cfg)
            model2.load_state_dict(torch.load(f.name, weights_only=True))

        model2.eval()
        with torch.no_grad():
            loaded_img = model2.encode_image(test_images, normalize=False)

        assert torch.allclose(ref_img, loaded_img, atol=1e-6)

    def test_state_dict_contains_lora_keys(self):
        """Test that the state dict includes LoRA A/B weight keys."""
        model = MockTransformerCLIP(dim=128, num_vision_blocks=4)
        peft_cfg = MockPEFTConfig(enabled=True, vision_enabled=True,
                                  num_last_blocks=2, rank=4)
        inject_lora(model, peft_cfg)

        state = model.state_dict()
        lora_keys = [k for k in state.keys() if 'lora_' in k]
        assert len(lora_keys) > 0

        # Should have both A and B for each injected module
        a_keys = [k for k in lora_keys if 'lora_A' in k]
        b_keys = [k for k in lora_keys if 'lora_B' in k]
        assert len(a_keys) == len(b_keys)
        assert len(a_keys) > 0


# ===========================================================================
# Backward compatibility integration tests
# ===========================================================================


@pytest.mark.multimodal_integration
class TestBackwardCompatibility:
    """Test that PEFT-disabled mode is identical to standard training."""

    def test_disabled_peft_no_lora_modules(self):
        """Test that PEFT disabled leaves no LoRA modules in the model."""
        model = MockTransformerCLIP(dim=128, num_vision_blocks=4)
        peft_cfg = MockPEFTConfig(enabled=False)
        result = inject_lora(model, peft_cfg)

        assert result is None
        for _, m in model.named_modules():
            assert not isinstance(m, LoRALinear)

    def test_disabled_peft_all_params_trainable(self):
        """Test that PEFT disabled keeps all parameters trainable."""
        model = MockTransformerCLIP(dim=128, num_vision_blocks=4)
        peft_cfg = MockPEFTConfig(enabled=False)
        inject_lora(model, peft_cfg)

        for name, param in model.named_parameters():
            assert param.requires_grad, \
                f"Param {name} should still be trainable with PEFT disabled"

    def test_disabled_peft_training_identical(self):
        """Test that training with PEFT disabled is functionally identical
        to training without ever calling inject_lora."""
        torch.manual_seed(42)
        model_a = MockTransformerCLIP(dim=128, num_vision_blocks=4)
        opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)

        torch.manual_seed(42)
        model_b = MockTransformerCLIP(dim=128, num_vision_blocks=4)
        peft_cfg = MockPEFTConfig(enabled=False)
        inject_lora(model_b, peft_cfg)
        opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)

        # Same random batch
        torch.manual_seed(123)
        batch = make_batch(8)

        # Step A
        opt_a.zero_grad()
        out_a = model_a(image=batch[0], text=batch[1])
        logits_a = out_a[2] * (out_a[0] @ out_a[1].T) + out_a[3]
        loss_a = F.binary_cross_entropy_with_logits(logits_a, torch.eye(8))
        loss_a.backward()
        opt_a.step()

        # Step B
        opt_b.zero_grad()
        out_b = model_b(image=batch[0], text=batch[1])
        logits_b = out_b[2] * (out_b[0] @ out_b[1].T) + out_b[3]
        loss_b = F.binary_cross_entropy_with_logits(logits_b, torch.eye(8))
        loss_b.backward()
        opt_b.step()

        assert torch.allclose(
            torch.tensor(loss_a.item()),
            torch.tensor(loss_b.item()),
            atol=1e-6,
        )


# ===========================================================================
# Real dataloader + LoRA integration test
# ===========================================================================


@pytest.mark.multimodal_integration
class TestLoRAWithDataloader:
    """Test LoRA training with real data loading from disk."""

    @pytest.fixture
    def temp_dataset(self):
        """Create a temporary image-text dataset on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            image_dir = tmpdir / "images"
            label_dir = tmpdir / "labels"
            image_dir.mkdir()
            label_dir.mkdir()

            for i in range(16):
                img = Image.new(
                    'RGB', (32, 32),
                    color=(i * 15 % 256, i * 25 % 256, i * 35 % 256),
                )
                img.save(image_dir / f"img_{i}.jpg")
                (label_dir / f"img_{i}.txt").write_text(
                    f"A sample image number {i}"
                )

            image_list = tmpdir / "train.txt"
            image_list.write_text(
                "\n".join([f"img_{i}.jpg" for i in range(16)])
            )

            yield {
                'image_dir': str(image_dir),
                'caption_dir': str(label_dir),
                'image_list_file': str(image_list),
            }

    def test_lora_training_with_real_dataloader(self, temp_dataset):
        """Test LoRA training loop using images loaded from disk."""
        from nvidia_tao_pytorch.multimodal.clip.dataloader.custom_loader import (
            get_custom_dataloader,
        )

        def simple_transform(img):
            img = img.resize((32, 32))
            return (
                torch.tensor([list(img.getdata())])
                .reshape(3, 32, 32)
                .float() / 255.0
            )

        def simple_tokenizer(text):
            return [torch.randn(64)]

        train_loader = get_custom_dataloader(
            datasets=[temp_dataset],
            batch_size=4,
            transform=simple_transform,
            tokenizer=simple_tokenizer,
            num_workers=0,
            mode='train',
        )

        model = MockTransformerCLIP(dim=128, num_vision_blocks=4,
                                    image_size=32)
        peft_cfg = MockPEFTConfig(enabled=True, vision_enabled=True,
                                  num_last_blocks=2, rank=4)
        inject_lora(model, peft_cfg)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )

        model.train()
        losses = []
        for i, batch in enumerate(train_loader):
            if i >= 4:
                break
            images, texts = batch
            optimizer.zero_grad()
            img_feat, txt_feat, scale, bias = model(
                image=images, text=texts
            )
            logits = scale * (img_feat @ txt_feat.T) + bias
            targets = torch.eye(images.shape[0])
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert len(losses) > 0
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)


# ===========================================================================
# LoRA export integration tests
# ===========================================================================


class MockVisionExportWrapper(nn.Module):
    """Wrapper for ONNX-exporting the vision encoder only."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image, normalize=True)


class MockCombinedExportWrapper(nn.Module):
    """Wrapper for ONNX-exporting both encoders."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, text):
        img_feat, txt_feat, scale, bias = self.model(
            image=image, text=text
        )
        return img_feat, txt_feat, scale


@pytest.mark.multimodal_integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestLoRAExport:
    """Test LoRA merge + ONNX export flow on GPU."""

    @staticmethod
    def _train_lora_model(dim=64, num_vision_blocks=4, num_text_blocks=4,
                          image_size=32, vision_enabled=True,
                          text_enabled=False, num_steps=5):
        """Helper: build, inject LoRA, train on GPU, return model + config."""
        # Keep GPU accumulation-order tolerance checks reproducible regardless
        # of the order in which the wider CI suite runs this test class.
        torch.manual_seed(42)
        device = torch.device('cuda')
        model = MockTransformerCLIP(
            dim=dim, num_vision_blocks=num_vision_blocks,
            num_text_blocks=num_text_blocks, image_size=image_size,
        ).to(device)

        peft_cfg = MockPEFTConfig(
            enabled=True, vision_enabled=vision_enabled,
            text_enabled=text_enabled, num_last_blocks=2,
            rank=4, dropout=0.0,
        )
        inject_lora(model, peft_cfg)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2
        )
        model.train()
        for _ in range(num_steps):
            optimizer.zero_grad()
            images = torch.randn(8, 3, image_size, image_size, device=device)
            texts = torch.randn(8, 64, device=device)
            img_feat, txt_feat, scale, bias = model(image=images, text=texts)
            logits = scale * (img_feat @ txt_feat.T) + bias
            loss = F.binary_cross_entropy_with_logits(
                logits, torch.eye(8, device=device)
            )
            loss.backward()
            optimizer.step()

        return model, peft_cfg

    def test_lora_merge_then_onnx_export_vision(self):
        """Test LoRA train -> merge -> ONNX export (vision encoder)."""
        model, _ = self._train_lora_model()
        device = next(model.parameters()).device

        # Reference output before merge
        model.eval()
        test_images = torch.randn(2, 3, 32, 32, device=device)
        with torch.no_grad():
            ref_output = model.encode_image(test_images, normalize=True)

        # Merge
        merged = merge_lora(model)
        assert merged > 0
        assert not any(isinstance(m, LoRALinear) for m in model.modules())

        # Output should match post-merge (relaxed for GPU float accumulation —
        # merge folds B@A into W which changes FP operation order)
        with torch.no_grad():
            post_merge_output = model.encode_image(test_images, normalize=True)
        assert torch.allclose(ref_output, post_merge_output, atol=1e-3)

        # ONNX export
        export_wrapper = MockVisionExportWrapper(model)
        export_wrapper.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "vision_encoder.onnx")
            torch.onnx.export(
                export_wrapper,
                test_images,
                onnx_path,
                input_names=["image"],
                output_names=["image_embedding"],
                dynamic_axes={
                    "image": {0: "batch_size"},
                    "image_embedding": {0: "batch_size"},
                },
                opset_version=17,
            )
            assert os.path.exists(onnx_path)
            assert os.path.getsize(onnx_path) > 0

    def test_lora_merge_then_onnx_export_combined(self):
        """Test ONNX export of both encoders after dual-tower LoRA merge."""
        model, _ = self._train_lora_model(
            vision_enabled=True, text_enabled=True
        )
        device = next(model.parameters()).device

        merge_lora(model)
        model.eval()

        export_wrapper = MockCombinedExportWrapper(model)
        export_wrapper.eval()

        test_images = torch.randn(2, 3, 32, 32, device=device)
        test_texts = torch.randn(2, 64, device=device)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "combined.onnx")
            torch.onnx.export(
                export_wrapper,
                (test_images, test_texts),
                onnx_path,
                input_names=["image", "text"],
                output_names=[
                    "image_embedding", "text_embedding", "logit_scale"
                ],
                opset_version=17,
            )
            assert os.path.exists(onnx_path)
            assert os.path.getsize(onnx_path) > 0

    def test_sft_model_export_without_lora(self):
        """Test that a standard SFT model (no LoRA) exports normally."""
        device = torch.device('cuda')
        model = MockTransformerCLIP(dim=64, num_vision_blocks=4,
                                    image_size=32).to(device)

        # Merge is a no-op on SFT model
        merged = merge_lora(model)
        assert merged == 0
        assert not any(isinstance(m, LoRALinear) for m in model.modules())

        model.eval()
        export_wrapper = MockVisionExportWrapper(model)
        export_wrapper.eval()

        test_images = torch.randn(2, 3, 32, 32, device=device)
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "sft_vision.onnx")
            torch.onnx.export(
                export_wrapper,
                test_images,
                onnx_path,
                input_names=["image"],
                output_names=["image_embedding"],
                opset_version=17,
            )
            assert os.path.exists(onnx_path)
            assert os.path.getsize(onnx_path) > 0

    def test_onnx_output_matches_pytorch_after_merge(self):
        """Test that ONNX runtime output matches PyTorch on GPU."""
        import onnxruntime

        model, _ = self._train_lora_model(num_steps=5)
        device = next(model.parameters()).device

        merge_lora(model)
        model.eval()
        export_wrapper = MockVisionExportWrapper(model)
        export_wrapper.eval()

        test_images = torch.randn(1, 3, 32, 32, device=device)
        with torch.no_grad():
            pytorch_output = export_wrapper(test_images).cpu().numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")
            torch.onnx.export(
                export_wrapper,
                test_images,
                onnx_path,
                input_names=["image"],
                output_names=["image_embedding"],
                opset_version=17,
            )

            # Run with CUDA execution provider
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            sess = onnxruntime.InferenceSession(onnx_path, providers=providers)
            ort_output = sess.run(
                ["image_embedding"],
                {"image": test_images.cpu().numpy()},
            )[0]

        assert pytorch_output.shape == ort_output.shape
        max_diff = abs(pytorch_output - ort_output).max()
        assert max_diff < 1e-4, f"Max diff: {max_diff}"

    def test_export_preserves_normalization(self):
        """Test that merged+exported model still produces unit-norm embeddings."""
        model, _ = self._train_lora_model(num_steps=3)
        device = next(model.parameters()).device

        merge_lora(model)
        model.eval()

        test_images = torch.randn(4, 3, 32, 32, device=device)
        with torch.no_grad():
            embeddings = model.encode_image(test_images, normalize=True)

        norms = embeddings.norm(dim=-1)
        assert torch.allclose(
            norms, torch.ones_like(norms), atol=1e-5
        )

    def test_train_save_load_merge_export_pipeline(self):
        """Test full pipeline: train -> checkpoint -> reload -> merge ->
        ONNX export, all on GPU."""
        device = torch.device('cuda')
        model, peft_cfg = self._train_lora_model(num_steps=5)

        # Reference output
        model.eval()
        test_images = torch.randn(2, 3, 32, 32, device=device)
        with torch.no_grad():
            ref_output = model.encode_image(test_images, normalize=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            ckpt_path = os.path.join(tmpdir, "lora_model.pth")
            torch.save(model.state_dict(), ckpt_path)

            # Load into fresh model
            model2 = MockTransformerCLIP(dim=64, num_vision_blocks=4,
                                         image_size=32).to(device)
            inject_lora(model2, peft_cfg)
            model2.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=True)
            )
            model2.eval()

            with torch.no_grad():
                loaded_output = model2.encode_image(
                    test_images, normalize=False
                )
            assert torch.allclose(ref_output, loaded_output, atol=1e-6)

            # Merge (relaxed tolerance — merge changes FP operation order)
            merge_lora(model2)
            with torch.no_grad():
                merged_output = model2.encode_image(
                    test_images, normalize=False
                )
            assert torch.allclose(ref_output, merged_output, atol=1e-3)

            # ONNX export
            export_wrapper = MockVisionExportWrapper(model2)
            export_wrapper.eval()
            onnx_path = os.path.join(tmpdir, "exported.onnx")
            torch.onnx.export(
                export_wrapper,
                test_images,
                onnx_path,
                input_names=["image"],
                output_names=["image_embedding"],
                opset_version=17,
            )
            assert os.path.exists(onnx_path)
            assert os.path.getsize(onnx_path) > 0
