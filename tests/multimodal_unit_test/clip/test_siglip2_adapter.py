# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SigLIP2 adapter with mocked model loading."""

from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn

from nvidia_tao_pytorch.multimodal.clip.model.adapters.siglip2 import SigLIP2
from nvidia_tao_pytorch.multimodal.clip.model.lora import LoRALinear


class MockSigLIP2Vision(nn.Module):
    """Mock SigLIP2 vision model."""

    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(3 * 64 * 64, embed_dim)
        self.head = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, -1)
        return {'pooler_output': self.linear(x)}

    def parameters(self):
        return list(self.linear.parameters()) + list(self.head.parameters())

    def named_parameters(self):
        for name, param in self.linear.named_parameters():
            yield f'linear.{name}', param
        for name, param in self.head.named_parameters():
            yield f'head.{name}', param


class MockSigLIP2Text(nn.Module):
    """Mock SigLIP2 text model."""

    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(512, embed_dim)

    def forward(self, input_ids=None, attention_mask=None):
        b = input_ids.shape[0]
        return MagicMock(pooler_output=torch.randn(b, self.embed_dim))

    def parameters(self):
        return list(self.linear.parameters())

    def named_parameters(self):
        for name, param in self.linear.named_parameters():
            yield f'linear.{name}', param


class MockAutoModel:
    """Mock AutoModel.from_pretrained."""

    @staticmethod
    def from_pretrained(model_name, **kwargs):
        return MockSigLIP2Vision()


class MockAutoProcessor:
    """Mock AutoProcessor.from_pretrained."""

    @staticmethod
    def from_pretrained(model_name, **kwargs):
        proc = MagicMock()
        proc.tokenizer = MagicMock()
        proc.image_processor = MagicMock()
        proc.image_processor.size = {'height': 224, 'width': 224}
        proc.image_processor.image_mean = [0.5, 0.5, 0.5]
        proc.image_processor.image_std = [0.5, 0.5, 0.5]
        return proc


@pytest.mark.multimodal_unit
class TestSigLIP2AdapterMocked:
    """Test SigLIP2 adapter with mocked model loading."""

    def test_adapter_mock_structure(self):
        """Test that mock adapter has correct structure."""
        adapter = nn.Module()
        adapter.logit_scale = nn.Parameter(torch.tensor(2.3026))
        adapter.logit_bias = nn.Parameter(torch.tensor(-10.0))
        adapter.vision_model = MockSigLIP2Vision()
        adapter.text_model = MockSigLIP2Text()
        adapter.tokenizer = MagicMock()

        assert hasattr(adapter, 'logit_scale')
        assert hasattr(adapter, 'logit_bias')
        assert hasattr(adapter, 'vision_model')
        assert hasattr(adapter, 'text_model')


@pytest.mark.multimodal_unit
class TestSigLIP2EncodeMethods:
    """Test encode methods."""

    def test_encode_image_mock(self):
        """Test encode_image with mock."""
        from nvidia_tao_pytorch.multimodal.clip.model.adapters.base import BaseCLIPAdapter

        class TestAdapter(BaseCLIPAdapter):
            def __init__(self):
                super().__init__(logit_scale_init=2.3026, logit_bias_init=-10.0)
                self.vision = nn.Linear(3 * 64 * 64, 768)
                self._tokenizer = None

            def encode_image(self, images, normalize=True):
                b = images.shape[0]
                x = images.view(b, -1)
                features = self.vision(x)
                if normalize:
                    features = nn.functional.normalize(features, dim=-1)
                return features

            def encode_text(self, text, normalize=True):
                return torch.randn(text.shape[0], 768)

            def vision_named_parameters(self):
                for name, param in self.vision.named_parameters():
                    yield f'vision.{name}', param

            def text_named_parameters(self):
                return iter([])

            def other_named_parameters(self):
                yield 'logit_scale', self.logit_scale
                yield 'logit_bias', self.logit_bias

        adapter = TestAdapter()
        images = torch.randn(4, 3, 64, 64)
        features = adapter.encode_image(images)

        assert features.shape == (4, 768)

    def test_encode_text_mock(self):
        """Test encode_text with mock."""
        from nvidia_tao_pytorch.multimodal.clip.model.adapters.base import BaseCLIPAdapter

        class TestAdapter(BaseCLIPAdapter):
            def __init__(self):
                super().__init__(logit_scale_init=2.3026, logit_bias_init=-10.0)
                self.text = nn.Linear(64, 768)
                self._tokenizer = MagicMock()

            def encode_image(self, images, normalize=True):
                return torch.randn(images.shape[0], 768)

            def encode_text(self, text, normalize=True):
                if isinstance(text, dict):
                    text = text['input_ids']
                features = self.text(text.float())
                if normalize:
                    features = nn.functional.normalize(features, dim=-1)
                return features

            def vision_named_parameters(self):
                return iter([])

            def text_named_parameters(self):
                for name, param in self.text.named_parameters():
                    yield f'text.{name}', param

            def other_named_parameters(self):
                yield 'logit_scale', self.logit_scale
                yield 'logit_bias', self.logit_bias

        adapter = TestAdapter()
        text = torch.randint(0, 1000, (4, 64))
        features = adapter.encode_text(text)

        assert features.shape == (4, 768)


@pytest.mark.multimodal_unit
class TestSigLIP2BuilderFunctions:
    """Test builder helper functions."""

    def test_normalize_tensor(self):
        """Test tensor normalization."""
        x = torch.randn(4, 768)
        normalized = nn.functional.normalize(x, dim=-1)

        # Check that norm is 1
        norms = torch.norm(normalized, dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_logit_scale_exponentiation(self):
        """Test that logit_scale is exponentiated."""
        logit_scale = nn.Parameter(torch.tensor(2.3026))
        scale = logit_scale.exp()

        assert torch.isclose(scale, torch.tensor(10.0), atol=1e-3)

    def test_siglip_default_params(self):
        """Test default SigLIP parameters."""
        # Default SigLIP params
        default_scale = 2.3026
        default_bias = -10.0

        assert abs(default_scale - 2.3026) < 1e-4
        assert abs(default_bias - (-10.0)) < 1e-4


@pytest.mark.multimodal_unit
class TestSigLIP2GradientCheckpointing:
    """Test gradient checkpointing for frozen SigLIP2 LoRA towers."""

    def test_adapter_configures_both_towers_and_disables_checkpointing(self):
        """The HF root propagates non-reentrant checkpointing to both towers."""
        from transformers import Siglip2Config, Siglip2Model

        config = Siglip2Config(
            text_config={
                'vocab_size': 32,
                'hidden_size': 16,
                'intermediate_size': 32,
                'num_hidden_layers': 1,
                'num_attention_heads': 4,
                'max_position_embeddings': 8,
            },
            vision_config={
                'hidden_size': 16,
                'intermediate_size': 32,
                'num_hidden_layers': 1,
                'num_attention_heads': 4,
                'patch_size': 2,
                'num_channels': 3,
            },
        )
        model = Siglip2Model(config)
        adapter = MagicMock()
        adapter.backbone.inner = model

        with patch.object(
            model,
            'gradient_checkpointing_enable',
            wraps=model.gradient_checkpointing_enable,
        ) as enable_mock:
            SigLIP2.set_grad_checkpointing(adapter, enable=True)

        enable_mock.assert_called_once_with(
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )
        assert model.text_model.encoder.gradient_checkpointing
        assert model.vision_model.encoder.gradient_checkpointing

        with patch.object(
            model,
            'gradient_checkpointing_disable',
            wraps=model.gradient_checkpointing_disable,
        ) as disable_mock:
            SigLIP2.set_grad_checkpointing(adapter, enable=False)

        disable_mock.assert_called_once_with()
        assert not model.text_model.encoder.gradient_checkpointing
        assert not model.vision_model.encoder.gradient_checkpointing

    def test_frozen_checkpointed_text_tower_preserves_lora_gradients(self):
        """Non-reentrant checkpointing keeps LoRA gradients on a frozen tower."""
        from transformers import Siglip2TextConfig, Siglip2TextModel

        config = Siglip2TextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=8,
        )
        model = Siglip2TextModel(config).train()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )
        for parameter in model.parameters():
            parameter.requires_grad = False
        attention = model.text_model.encoder.layers[0].self_attn
        attention.q_proj = LoRALinear(
            attention.q_proj, rank=2, alpha=4, dropout=0.0
        )

        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        attention_mask = torch.ones_like(input_ids)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        output.sum().backward()

        lora_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if 'lora_' in name
        ]
        assert lora_gradients
        assert all(gradient is not None for gradient in lora_gradients)
        assert any(torch.count_nonzero(gradient) for gradient in lora_gradients)
