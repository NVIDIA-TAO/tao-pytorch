# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OpenCLIP adapter."""

import pytest
import torch
import torch.nn as nn

from nvidia_tao_pytorch.multimodal.clip.model.adapters.openclip import OpenCLIP


class MockOpenCLIPModel(nn.Module):
    """Mock OpenCLIP model for testing."""

    def __init__(self, embed_dim=512):
        super().__init__()
        self.visual = nn.Sequential(
            nn.Linear(3 * 32 * 32, embed_dim),
            nn.ReLU(),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True),
            num_layers=2
        )
        self.token_embedding = nn.Embedding(1000, embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(77, embed_dim))
        self.ln_final = nn.LayerNorm(embed_dim)
        self.text_projection = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.embed_dim = embed_dim

    def encode_image(self, x):
        b = x.shape[0]
        x = x.view(b, -1)
        return self.visual(x)

    def encode_text(self, text, normalize=False):
        x = self.token_embedding(text)
        x = x + self.positional_embedding[:x.shape[1]]
        x = self.transformer(x)
        x = self.ln_final(x[:, 0, :])
        x = x @ self.text_projection
        if normalize:
            x = x / x.norm(dim=-1, keepdim=True)
        return x


class MockBackbone(nn.Module):
    """Mock backbone wrapper for OpenCLIP model."""

    def __init__(self, embed_dim=512):
        super().__init__()
        self.model = MockOpenCLIPModel(embed_dim)
        self._model_name = "MockOpenCLIP"
        self.tokenizer = MockTokenizer()

    def forward_pre_logits(self, x):
        return self.model.encode_image(x)

    def encode_text(self, text, normalize=False):
        return self.model.encode_text(text, normalize)

    def set_grad_checkpointing(self, enable=True):
        pass


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __call__(self, text):
        if isinstance(text, str):
            return torch.randint(0, 1000, (1, 77))
        return torch.randint(0, 1000, (len(text), 77))


@pytest.mark.multimodal_unit
class TestOpenCLIPAdapter:
    """Test OpenCLIP adapter class."""

    @pytest.fixture
    def mock_backbone(self):
        """Create mock backbone."""
        return MockBackbone()

    def test_initialization(self, mock_backbone):
        """Test basic initialization."""
        adapter = OpenCLIP(mock_backbone)

        assert adapter.backbone is mock_backbone
        assert hasattr(adapter, 'logit_scale')
        assert hasattr(adapter, 'logit_bias')

    def test_logit_scale_init(self, mock_backbone):
        """Test logit scale initialization."""
        adapter = OpenCLIP(mock_backbone, logit_scale_init=2.3026)

        assert pytest.approx(adapter.logit_scale.item(), rel=1e-3) == 2.3026

    def test_logit_bias_init(self, mock_backbone):
        """Test logit bias initialization."""
        adapter = OpenCLIP(mock_backbone, logit_bias_init=-10.0)

        assert pytest.approx(adapter.logit_bias.item(), rel=1e-3) == -10.0

    def test_has_tokenizer(self, mock_backbone):
        """Test that adapter has tokenizer."""
        adapter = OpenCLIP(mock_backbone)

        assert adapter.tokenizer is not None

    def test_encode_image_output_shape(self, mock_backbone):
        """Test encode_image output shape."""
        adapter = OpenCLIP(mock_backbone)
        images = torch.randn(4, 3, 32, 32)

        features = adapter.encode_image(images)

        assert features.shape == (4, 512)

    def test_encode_image_normalized(self, mock_backbone):
        """Test that encode_image output is normalized by default."""
        adapter = OpenCLIP(mock_backbone)
        images = torch.randn(4, 3, 32, 32)

        features = adapter.encode_image(images, normalize=True)

        norms = features.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_encode_image_unnormalized(self, mock_backbone):
        """Test encode_image without normalization."""
        adapter = OpenCLIP(mock_backbone)
        images = torch.randn(4, 3, 32, 32)

        features = adapter.encode_image(images, normalize=False)

        # Output should exist but may not be unit norm
        assert features.shape == (4, 512)

    def test_encode_text_output_shape(self, mock_backbone):
        """Test encode_text output shape."""
        adapter = OpenCLIP(mock_backbone)
        text = torch.randint(0, 1000, (4, 77))

        features = adapter.encode_text(text)

        assert features.shape == (4, 512)

    def test_encode_text_with_dict_input(self, mock_backbone):
        """Test encode_text with dictionary input."""
        adapter = OpenCLIP(mock_backbone)
        text_dict = {'input_ids': torch.randint(0, 1000, (4, 77))}

        features = adapter.encode_text(text_dict)

        assert features.shape == (4, 512)

    def test_encode_text_normalized(self, mock_backbone):
        """Test that encode_text output is normalized by default."""
        adapter = OpenCLIP(mock_backbone)
        text = torch.randint(0, 1000, (4, 77))

        features = adapter.encode_text(text, normalize=True)

        norms = features.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_forward_returns_features(self, mock_backbone):
        """Test forward returns features and logit params."""
        adapter = OpenCLIP(mock_backbone)
        images = torch.randn(4, 3, 32, 32)
        text = torch.randint(0, 1000, (4, 77))

        result = adapter(images, text)

        # A biasless native model returns only its canonical calibration.
        assert isinstance(result, tuple)
        assert len(result) == 3
        image_features, text_features, logit_scale = result
        assert image_features.shape == (4, 512)
        assert text_features.shape == (4, 512)
        assert logit_scale.ndim == 0  # scalar

    def test_freeze_vision_encoder(self, mock_backbone):
        """Test freezing vision encoder."""
        adapter = OpenCLIP(mock_backbone, freeze_vision_encoder=True)

        for param in adapter.backbone.model.visual.parameters():
            assert not param.requires_grad

    def test_freeze_text_encoder(self, mock_backbone):
        """Test freezing text encoder."""
        adapter = OpenCLIP(mock_backbone, freeze_text_encoder=True)

        # Check transformer is frozen
        for param in adapter.backbone.model.transformer.parameters():
            assert not param.requires_grad

        # Check token_embedding is frozen
        for param in adapter.backbone.model.token_embedding.parameters():
            assert not param.requires_grad

    def test_freeze_both_encoders(self, mock_backbone):
        """Test freezing both encoders."""
        adapter = OpenCLIP(
            mock_backbone,
            freeze_vision_encoder=True,
            freeze_text_encoder=True
        )

        # Vision should be frozen
        for param in adapter.backbone.model.visual.parameters():
            assert not param.requires_grad

        # Text should be frozen
        for param in adapter.backbone.model.transformer.parameters():
            assert not param.requires_grad

    def test_vision_named_parameters(self, mock_backbone):
        """Test vision_named_parameters generator."""
        adapter = OpenCLIP(mock_backbone)

        params = list(adapter.vision_named_parameters())

        assert len(params) > 0
        for name, param in params:
            assert 'backbone.model.visual' in name
            assert isinstance(param, torch.nn.Parameter)

    def test_text_named_parameters(self, mock_backbone):
        """Test text_named_parameters generator."""
        adapter = OpenCLIP(mock_backbone)

        params = list(adapter.text_named_parameters())

        assert len(params) > 0
        # Should include transformer, token_embedding, etc.
        names = [n for n, _ in params]
        assert any('transformer' in n for n in names)
        assert any('token_embedding' in n for n in names)

    def test_other_named_parameters(self, mock_backbone):
        """Test the canonical calibration parameter iterator."""
        adapter = OpenCLIP(mock_backbone)

        params = list(adapter.other_named_parameters())

        assert len(params) == 1
        names = [n for n, _ in params]
        assert names == ['logit_scale']

    def test_set_grad_checkpointing(self, mock_backbone):
        """Test set_grad_checkpointing method."""
        adapter = OpenCLIP(mock_backbone)

        # Should not raise
        adapter.set_grad_checkpointing(True)
        adapter.set_grad_checkpointing(False)
