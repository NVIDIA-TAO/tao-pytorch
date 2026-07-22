# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BaseCLIPAdapter."""

import pytest
import torch
import torch.nn as nn

from nvidia_tao_pytorch.multimodal.clip.model.adapters import BaseCLIPAdapter


class ConcreteAdapter(BaseCLIPAdapter):
    """Concrete implementation for testing."""

    def __init__(self, embed_dim=512, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.image_encoder = nn.Linear(224 * 224 * 3, embed_dim)
        self.text_encoder = nn.Linear(64, embed_dim)

    def encode_image(self, image, normalize=True):
        """Encode image."""
        batch_size = image.shape[0]
        image_flat = image.view(batch_size, -1)
        features = self.image_encoder(image_flat)
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def encode_text(self, text, normalize=True):
        """Encode text."""
        if isinstance(text, dict):
            text = text['input_ids']
        features = self.text_encoder(text.float())
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def vision_named_parameters(self):
        """Return vision encoder named parameters."""
        for name, param in self.image_encoder.named_parameters():
            yield f'image_encoder.{name}', param

    def text_named_parameters(self):
        """Return text encoder named parameters."""
        for name, param in self.text_encoder.named_parameters():
            yield f'text_encoder.{name}', param


@pytest.mark.multimodal_unit
class TestBaseCLIPAdapterInit:
    """Test BaseCLIPAdapter initialization."""

    def test_default_logit_scale(self):
        """Test default logit scale initialization."""
        adapter = ConcreteAdapter()
        # Default: 2.3026 (ln(10))
        assert pytest.approx(adapter.logit_scale.item(), rel=1e-3) == 2.3026

    def test_default_logit_bias(self):
        """Test default logit bias initialization."""
        adapter = ConcreteAdapter()
        # Default: -10.0
        assert pytest.approx(adapter.logit_bias.item(), rel=1e-3) == -10.0

    def test_custom_logit_scale(self):
        """Test custom logit scale initialization."""
        adapter = ConcreteAdapter(logit_scale_init=1.0)
        assert pytest.approx(adapter.logit_scale.item(), rel=1e-3) == 1.0

    def test_custom_logit_bias(self):
        """Test custom logit bias initialization."""
        adapter = ConcreteAdapter(logit_bias_init=-5.0)
        assert pytest.approx(adapter.logit_bias.item(), rel=1e-3) == -5.0

    def test_tokenizer_none_by_default(self):
        """Test tokenizer is None by default."""
        adapter = ConcreteAdapter()
        assert adapter.tokenizer is None

    def test_logit_parameters_are_trainable(self):
        """Test that logit parameters are trainable."""
        adapter = ConcreteAdapter()
        assert adapter.logit_scale.requires_grad
        assert adapter.logit_bias.requires_grad


@pytest.mark.multimodal_unit
class TestBaseCLIPAdapterForward:
    """Test BaseCLIPAdapter forward pass."""

    @pytest.fixture
    def adapter(self):
        """Create adapter fixture."""
        return ConcreteAdapter(embed_dim=512)

    @pytest.fixture
    def sample_image(self):
        """Create sample image tensor."""
        return torch.randn(2, 3, 224, 224)

    @pytest.fixture
    def sample_text(self):
        """Create sample text tensor."""
        return {'input_ids': torch.randn(2, 64)}

    def test_forward_image_and_text(self, adapter, sample_image, sample_text):
        """Test forward with both image and text."""
        result = adapter(image=sample_image, text=sample_text)

        assert isinstance(result, tuple)
        assert len(result) == 4

        image_features, text_features, logit_scale, logit_bias = result

        assert image_features.shape == (2, 512)
        assert text_features.shape == (2, 512)
        assert logit_scale.dim() == 0  # scalar
        assert logit_bias.dim() == 0  # scalar

    def test_forward_image_only(self, adapter, sample_image):
        """Test forward with image only."""
        result = adapter(image=sample_image)

        assert isinstance(result, dict)
        assert 'image_features' in result
        assert result['image_features'].shape == (2, 512)

    def test_forward_text_only(self, adapter, sample_text):
        """Test forward with text only."""
        result = adapter(text=sample_text)

        assert isinstance(result, dict)
        assert 'text_features' in result
        assert result['text_features'].shape == (2, 512)

    def test_forward_neither_raises_error(self, adapter):
        """Test forward with neither image nor text raises error."""
        with pytest.raises(ValueError, match="Either image or text must be provided"):
            adapter(image=None, text=None)

    def test_forward_image_normalized(self, adapter, sample_image):
        """Test that image features are normalized."""
        result = adapter(image=sample_image)
        features = result['image_features']

        # Check unit norm
        norms = features.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_forward_text_normalized(self, adapter, sample_text):
        """Test that text features are normalized."""
        result = adapter(text=sample_text)
        features = result['text_features']

        # Check unit norm
        norms = features.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_logit_scale_exponentiated(self, adapter, sample_image, sample_text):
        """Test that logit_scale is returned exponentiated."""
        _, _, logit_scale, _ = adapter(image=sample_image, text=sample_text)

        # Default init is 2.3026, exp(2.3026) ≈ 10.0
        assert pytest.approx(logit_scale.item(), rel=1e-2) == 10.0


@pytest.mark.multimodal_unit
class TestBaseCLIPAdapterMethods:
    """Test BaseCLIPAdapter utility methods."""

    def test_set_grad_checkpointing(self):
        """Test set_grad_checkpointing method exists and is callable."""
        adapter = ConcreteAdapter()
        # Should not raise
        adapter.set_grad_checkpointing(True)
        adapter.set_grad_checkpointing(False)



@pytest.mark.multimodal_unit
class TestBaseCLIPAdapterAbstract:
    """Test that BaseCLIPAdapter is properly abstract."""

    def test_cannot_instantiate_base(self):
        """Test that BaseCLIPAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseCLIPAdapter()

    def test_must_implement_encode_image(self):
        """Test that encode_image must be implemented."""
        class IncompleteAdapter(BaseCLIPAdapter):
            def encode_text(self, text, normalize=True):
                return torch.zeros(1, 512)
            def vision_named_parameters(self):
                return iter([])
            def text_named_parameters(self):
                return iter([])

        with pytest.raises(TypeError):
            IncompleteAdapter()

    def test_must_implement_encode_text(self):
        """Test that encode_text must be implemented."""
        class IncompleteAdapter(BaseCLIPAdapter):
            def encode_image(self, image, normalize=True):
                return torch.zeros(1, 512)
            def vision_named_parameters(self):
                return iter([])
            def text_named_parameters(self):
                return iter([])

        with pytest.raises(TypeError):
            IncompleteAdapter()
