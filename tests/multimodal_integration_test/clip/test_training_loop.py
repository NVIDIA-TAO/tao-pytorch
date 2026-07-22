# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for CLIP training loop with mock models."""

import tempfile
from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MockCLIPAdapter(nn.Module):
    """Minimal mock CLIP adapter for integration testing."""

    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.vision_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, embed_dim),
        )
        self.text_encoder = nn.Linear(64, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))
        self.tokenizer = MagicMock()

    def encode_image(self, x, normalize=True):
        features = self.vision_encoder(x)
        if normalize:
            features = nn.functional.normalize(features, dim=-1)
        return features

    def encode_text(self, x, normalize=True):
        if isinstance(x, dict):
            x = x['input_ids']
        features = self.text_encoder(x.float())
        if normalize:
            features = nn.functional.normalize(features, dim=-1)
        return features

    def forward(self, image, text):
        img_features = self.encode_image(image)
        txt_features = self.encode_text(text)
        return img_features, txt_features, self.logit_scale.exp(), self.logit_bias

    def vision_named_parameters(self):
        for name, param in self.vision_encoder.named_parameters():
            yield f'vision_encoder.{name}', param

    def text_named_parameters(self):
        for name, param in self.text_encoder.named_parameters():
            yield f'text_encoder.{name}', param

    def other_named_parameters(self):
        yield 'logit_scale', self.logit_scale
        yield 'logit_bias', self.logit_bias

    def set_grad_checkpointing(self, enable=True):
        pass


@dataclass
class MockOptimConfig:
    optimizer_type: str = 'adamw'
    vision_lr: float = 1e-3
    text_lr: float = 1e-3
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    warmup_steps: int = 10
    scheduler: str = 'cosine'


@dataclass
class MockTrainConfig:
    num_epochs: int = 2
    loss_type: str = 'siglip'
    grad_checkpointing: bool = False
    optim: MockOptimConfig = field(default_factory=MockOptimConfig)


@dataclass
class MockModelConfig:
    type: str = 'test'
    image_size: int = 32
    init_logit_scale: float = None
    init_logit_bias: float = None


@dataclass
class MockExperimentConfig:
    train: MockTrainConfig = field(default_factory=MockTrainConfig)
    model: MockModelConfig = field(default_factory=MockModelConfig)


def create_dummy_dataloader(batch_size=8, num_samples=32):
    """Create a dummy dataloader for testing."""
    images = torch.randn(num_samples, 3, 32, 32)
    texts = torch.randint(0, 100, (num_samples, 64))
    dataset = TensorDataset(images, texts)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


@pytest.mark.multimodal_integration
class TestMockCLIPAdapterTraining:
    """Test training with mock CLIP adapter."""

    def test_single_training_step(self):
        """Test a single training step executes without error."""
        adapter = MockCLIPAdapter()
        batch = (torch.randn(4, 3, 32, 32), torch.randint(0, 100, (4, 64)))

        images, texts = batch
        img_features, txt_features, scale, bias = adapter(images, texts)

        # Compute loss manually
        logits = scale * (img_features @ txt_features.T) + bias
        targets = torch.eye(4)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss)

    def test_backward_pass(self):
        """Test backward pass computes gradients."""
        adapter = MockCLIPAdapter()
        batch = (torch.randn(4, 3, 32, 32), torch.randint(0, 100, (4, 64)))

        images, texts = batch
        img_features, txt_features, scale, bias = adapter(images, texts)
        logits = scale * (img_features @ txt_features.T) + bias
        targets = torch.eye(4)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()

        # Check gradients exist
        has_grad = False
        for param in adapter.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad

    def test_optimizer_step(self):
        """Test optimizer step updates parameters."""
        adapter = MockCLIPAdapter()
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)

        # Get initial param values
        initial_params = {
            name: param.clone()
            for name, param in adapter.named_parameters()
            if param.requires_grad
        }

        # Training step
        batch = (torch.randn(4, 3, 32, 32), torch.randint(0, 100, (4, 64)))
        images, texts = batch
        img_features, txt_features, scale, bias = adapter(images, texts)
        logits = scale * (img_features @ txt_features.T) + bias
        targets = torch.eye(4)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        optimizer.step()

        # Check params changed
        params_changed = False
        for name, param in adapter.named_parameters():
            if param.requires_grad and name in initial_params:
                if not torch.allclose(param, initial_params[name]):
                    params_changed = True
                    break
        assert params_changed

    def test_multiple_training_steps(self):
        """Test multiple training steps."""
        adapter = MockCLIPAdapter()
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)

        losses = []
        for i in range(5):
            batch = (torch.randn(4, 3, 32, 32), torch.randint(0, 100, (4, 64)))
            optimizer.zero_grad()
            images, texts = batch
            img_features, txt_features, scale, bias = adapter(images, texts)
            logits = scale * (img_features @ txt_features.T) + bias
            targets = torch.eye(4)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # All losses should be finite
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)


@pytest.mark.multimodal_integration
class TestLossTypes:
    """Test different loss types."""

    def test_siglip_loss_computation(self):
        """Test SigLIP-style loss computation."""
        adapter = MockCLIPAdapter()
        batch = (torch.randn(4, 3, 32, 32), torch.randint(0, 100, (4, 64)))

        images, texts = batch
        img_features, txt_features, scale, bias = adapter(images, texts)

        # SigLIP loss: sigmoid cross-entropy
        logits = scale * (img_features @ txt_features.T) + bias
        targets = 2 * torch.eye(4) - 1  # {-1, 1} targets for SigLIP
        loss = -nn.functional.logsigmoid(logits * targets).mean()

        assert torch.isfinite(loss)
        assert loss > 0

    def test_clip_loss_computation(self):
        """Test CLIP-style loss computation."""
        adapter = MockCLIPAdapter()
        batch = (torch.randn(4, 3, 32, 32), torch.randint(0, 100, (4, 64)))

        images, texts = batch
        img_features, txt_features, scale, bias = adapter(images, texts)

        # CLIP loss: cross-entropy
        logits = scale * (img_features @ txt_features.T)
        targets = torch.arange(4)
        loss = (nn.functional.cross_entropy(logits, targets) +
                nn.functional.cross_entropy(logits.T, targets)) / 2

        assert torch.isfinite(loss)
        assert loss > 0


@pytest.mark.multimodal_integration
class TestOptimizerConfigurations:
    """Test different optimizer configurations."""

    def test_adamw_optimizer(self):
        """Test AdamW optimizer configuration."""
        adapter = MockCLIPAdapter()
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)

        assert isinstance(optimizer, torch.optim.AdamW)

    def test_sgd_optimizer(self):
        """Test SGD optimizer configuration."""
        adapter = MockCLIPAdapter()
        optimizer = torch.optim.SGD(adapter.parameters(), lr=1e-3)

        assert isinstance(optimizer, torch.optim.SGD)

    def test_lr_scheduler_cosine(self):
        """Test cosine LR scheduler."""
        adapter = MockCLIPAdapter()
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        initial_lr = optimizer.param_groups[0]['lr']
        for _ in range(50):
            scheduler.step()

        assert optimizer.param_groups[0]['lr'] < initial_lr
