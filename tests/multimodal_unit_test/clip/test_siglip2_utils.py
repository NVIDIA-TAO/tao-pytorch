# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SigLIP2 utilities (mock-based, no model downloads)."""

import pytest
import torch
import torch.nn as nn

from nvidia_tao_pytorch.multimodal.clip.model.transforms import SigLIP2ImageTransform


@pytest.mark.multimodal_unit
class TestSigLIP2BackboneVersions:
    """Test SigLIP2 backbone version resolution without model downloads."""

    def test_supported_hf_versions_are_mapped(self, monkeypatch):
        """Test that full SigLIP2 release names resolve to the right HF repos."""
        from nvidia_tao_pytorch.cv.backbone_v2 import siglip2 as siglip2_module

        seen_sources = []
        seen_wrapper_args = []

        def fake_from_pretrained(source, trust_remote_code):
            seen_sources.append((source, trust_remote_code))
            return object()

        class DummyProcessor:
            pass

        class DummyWrapper:
            def __init__(
                self, model, tokenizer, num_classes, is_dynamic, patch_size
            ):
                seen_wrapper_args.append((num_classes, is_dynamic, patch_size))

        monkeypatch.setattr(
            siglip2_module.AutoModel, "from_pretrained", fake_from_pretrained
        )
        monkeypatch.setattr(
            siglip2_module.AutoProcessor,
            "from_pretrained",
            lambda source, trust_remote_code: DummyProcessor(),
        )
        monkeypatch.setattr(siglip2_module, "SigLIP2Wrapper", DummyWrapper)

        siglip2_module.get_siglip2_model("siglip2-so400m-patch14-224")
        siglip2_module.get_siglip2_model("siglip2-so400m-patch16-naflex")

        assert seen_sources == [
            ("google/siglip2-so400m-patch14-224", True),
            ("google/siglip2-so400m-patch16-naflex", True),
        ]
        assert seen_wrapper_args == [
            (0, False, 14),
            (0, True, 16),
        ]


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


@pytest.mark.multimodal_unit
class TestBatchHardTripletLoss:
    """Test auxiliary batch-hard image-text triplet loss."""

    def test_single_sample_batch_returns_zero(self):
        """Single-sample batches have no negatives, so loss is zero."""
        from nvidia_tao_pytorch.multimodal.clip.model.pl_clip_model import (
            _batch_hard_image_text_triplet_loss,
        )

        image_features = torch.tensor([[1.0, 0.0]])
        text_features = torch.tensor([[1.0, 0.0]])

        loss = _batch_hard_image_text_triplet_loss(
            image_features, text_features, margin=0.2
        )

        assert loss.item() == 0.0

    def test_separated_pairs_have_zero_loss(self):
        """Perfectly aligned orthogonal pairs satisfy the margin."""
        from nvidia_tao_pytorch.multimodal.clip.model.pl_clip_model import (
            _batch_hard_image_text_triplet_loss,
        )

        image_features = torch.eye(3)
        text_features = torch.eye(3)

        loss = _batch_hard_image_text_triplet_loss(
            image_features, text_features, margin=0.2
        )

        assert torch.isclose(loss, torch.tensor(0.0))

    def test_swapped_pairs_have_positive_loss(self):
        """Swapped pairs make negatives more similar than positives."""
        from nvidia_tao_pytorch.multimodal.clip.model.pl_clip_model import (
            _batch_hard_image_text_triplet_loss,
        )

        image_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        text_features = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

        loss = _batch_hard_image_text_triplet_loss(
            image_features, text_features, margin=0.2
        )

        assert loss.item() > 0.0
