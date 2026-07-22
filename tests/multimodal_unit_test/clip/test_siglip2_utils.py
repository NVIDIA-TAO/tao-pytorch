# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SigLIP2 utilities (mock-based, no model downloads)."""

import pytest
import torch
import torch.nn as nn

from nvidia_tao_pytorch.multimodal.clip.model.transforms import SigLIP2ImageTransform


@pytest.mark.multimodal_unit
class TestSigLIP2ImageTransform:
    """Test SigLIP2ImageTransform class."""

    def test_initialization(self):
        """Test transform initialization."""
        class MockProcessor:
            def __call__(self, images, return_tensors):
                return {'pixel_values': torch.zeros(1, 3, 224, 224)}

        transform = SigLIP2ImageTransform(MockProcessor(), is_train=True)
        assert transform.is_train is True

        transform = SigLIP2ImageTransform(MockProcessor(), is_train=False)
        assert transform.is_train is False

    def test_call_squeezes_batch(self):
        """Test that __call__ squeezes batch dimension."""
        class MockProcessor:
            def __call__(self, images, return_tensors):
                return {
                    'pixel_values': torch.zeros(1, 3, 224, 224),
                    'attention_mask': torch.ones(1, 196),
                }

        transform = SigLIP2ImageTransform(MockProcessor())
        result = transform(None)  # Image not actually used by mock

        # Batch dimension should be squeezed
        assert result['pixel_values'].shape == (3, 224, 224)
        assert result['attention_mask'].shape == (196,)

    def test_returns_dict(self):
        """Test that transform returns dict with processor outputs."""
        class MockProcessor:
            def __call__(self, images, return_tensors):
                return {
                    'pixel_values': torch.zeros(1, 3, 224, 224),
                    'pixel_attention_mask': torch.ones(1, 196),
                    'spatial_shapes': torch.tensor([[14, 14]]),
                }

        transform = SigLIP2ImageTransform(MockProcessor())
        result = transform(None)

        assert isinstance(result, dict)
        assert 'pixel_values' in result
        assert 'pixel_attention_mask' in result
        assert 'spatial_shapes' in result


@pytest.mark.multimodal_unit
class TestSigLIP2Interface:
    """Test SigLIP2 interface without real model loading."""

    def test_adapter_inherits_from_base(self):
        """Test that SigLIP2 inherits from BaseCLIPAdapter."""
        from nvidia_tao_pytorch.multimodal.clip.model.adapters import SigLIP2
        from nvidia_tao_pytorch.multimodal.clip.model.adapters import BaseCLIPAdapter

        assert issubclass(SigLIP2, BaseCLIPAdapter)

    def test_build_function_validates_model_version(self):
        """Test that build_siglip2_model validates model version."""
        from nvidia_tao_pytorch.multimodal.clip.model.builders import build_siglip2_model

        with pytest.raises(ValueError, match="Unknown SigLIP2 model"):
            build_siglip2_model('invalid-model-version')


@pytest.mark.multimodal_unit
class TestBuildSigLIP2ModelValidation:
    """Test build_siglip2_model input validation."""

    def test_unknown_model_raises_error(self):
        """Test that unknown model version raises ValueError."""
        from nvidia_tao_pytorch.multimodal.clip.model.builders import build_siglip2_model

        with pytest.raises(ValueError) as exc_info:
            build_siglip2_model('nonexistent-model')

        assert "Unknown SigLIP2 model" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_error_includes_available_models(self):
        """Test that error message includes available model list."""
        from nvidia_tao_pytorch.multimodal.clip.model.builders import build_siglip2_model
        from nvidia_tao_pytorch.multimodal.clip.utils.model_configs import siglip2_model_configs

        with pytest.raises(ValueError) as exc_info:
            build_siglip2_model('bad-model')

        # Error should mention available models
        for model_name in siglip2_model_configs.keys():
            assert model_name in str(exc_info.value)
