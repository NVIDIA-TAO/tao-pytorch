# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CLIP model builders."""

import pytest

from nvidia_tao_pytorch.multimodal.clip.model.builders import (
    _parse_aug_config,
    _build_image_transforms,
    _resolve_adaptor_name,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)


@pytest.mark.multimodal_unit
class TestNormalizationConstants:
    """Test normalization constants."""

    def test_openai_clip_mean(self):
        """Test OpenAI CLIP mean values."""
        assert len(OPENAI_CLIP_MEAN) == 3
        assert all(0 <= v <= 1 for v in OPENAI_CLIP_MEAN)

    def test_openai_clip_std(self):
        """Test OpenAI CLIP std values."""
        assert len(OPENAI_CLIP_STD) == 3
        assert all(0 < v <= 1 for v in OPENAI_CLIP_STD)


@pytest.mark.multimodal_unit
class TestParseAugConfig:
    """Test _parse_aug_config function."""

    def test_none_returns_none(self):
        """Test that None input returns None."""
        assert _parse_aug_config(None) is None

    def test_default_values(self):
        """Test default augmentation values."""
        result = _parse_aug_config({})

        assert result['scale'] == [0.4, 1.0]
        assert result['gray_scale_prob'] == 0.2
        assert result['color_jitter_prob'] == 0.8
        assert result['color_jitter'] == [0.32, 0.32, 0.32, 0.08]

    def test_custom_scale(self):
        """Test custom scale setting."""
        result = _parse_aug_config({'scale': [0.8, 1.0]})

        assert result['scale'] == [0.8, 1.0]

    def test_grayscale_setting(self):
        """Test grayscale probability setting."""
        result = _parse_aug_config({'grayscale': 0.5})

        assert result['gray_scale_prob'] == 0.5

    def test_legacy_grayscale_prob(self):
        """Test legacy grayscale_prob field."""
        result = _parse_aug_config({'grayscale_prob': 0.3})

        assert result['gray_scale_prob'] == 0.3

    def test_legacy_gray_scale_prob(self):
        """Test legacy gray_scale_prob field."""
        result = _parse_aug_config({'gray_scale_prob': 0.4})

        assert result['gray_scale_prob'] == 0.4

    def test_color_jitter_5_elements(self):
        """Test color_jitter with 5 elements [prob, b, c, s, h]."""
        result = _parse_aug_config({
            'color_jitter': [0.5, 0.1, 0.2, 0.3, 0.04]
        })

        assert result['color_jitter_prob'] == 0.5
        assert result['color_jitter'] == [0.1, 0.2, 0.3, 0.04]

    def test_color_jitter_4_elements_legacy(self):
        """Test legacy color_jitter with 4 elements [b, c, s, h]."""
        result = _parse_aug_config({
            'color_jitter': [0.1, 0.2, 0.3, 0.04],
            'color_jitter_prob': 0.7
        })

        assert result['color_jitter_prob'] == 0.7
        assert result['color_jitter'] == [0.1, 0.2, 0.3, 0.04]

    def test_color_jitter_empty_disables(self):
        """Test that empty color_jitter disables the augmentation."""
        result = _parse_aug_config({'color_jitter': []})

        assert result['color_jitter_prob'] == 0.0
        assert result['color_jitter'] is None

    def test_color_jitter_wrong_length_disables(self):
        """Test that color_jitter with wrong length disables augmentation."""
        result = _parse_aug_config({'color_jitter': [0.1, 0.2]})

        assert result['color_jitter_prob'] == 0.0
        assert result['color_jitter'] is None


@pytest.mark.multimodal_unit
class TestBuildImageTransforms:
    """Test _build_image_transforms function."""

    def test_creates_transforms(self):
        """Test that train and val transforms are created."""
        train_transform, val_transform = _build_image_transforms(
            image_size=224,
            aug_cfg=None,
            mean=OPENAI_CLIP_MEAN,
            std=OPENAI_CLIP_STD
        )

        assert train_transform is not None
        assert val_transform is not None

    def test_train_and_val_different(self):
        """Test that train and val transforms are different."""
        train_transform, val_transform = _build_image_transforms(
            image_size=224,
            aug_cfg={'scale': [0.4, 1.0]},
            mean=OPENAI_CLIP_MEAN,
            std=OPENAI_CLIP_STD
        )

        # They should be different objects (train has augmentation)
        assert train_transform is not val_transform

    def test_custom_image_size(self):
        """Test transforms with custom image size."""
        train_transform, val_transform = _build_image_transforms(
            image_size=512,
            aug_cfg=None,
            mean=OPENAI_CLIP_MEAN,
            std=OPENAI_CLIP_STD
        )

        # Transforms should be created successfully
        assert train_transform is not None
        assert val_transform is not None


@pytest.mark.multimodal_unit
class TestResolveAdaptorName:
    """Test _resolve_adaptor_name function."""

    def test_clip_passes_through(self):
        """Test that 'clip' passes through unchanged."""
        result = _resolve_adaptor_name('clip', 'c-radio_v3-l')
        assert result == 'clip'

    def test_legacy_dfn_clip_alias(self):
        """Test that 'dfn_clip' resolves to 'clip'."""
        result = _resolve_adaptor_name('dfn_clip', 'c-radio_v3-l')
        assert result == 'clip'

    def test_legacy_dfn_alias(self):
        """Test that 'dfn' resolves to 'clip'."""
        result = _resolve_adaptor_name('dfn', 'c-radio_v3-l')
        assert result == 'clip'

    def test_siglip_resolves_for_known_model(self):
        """Test that 'siglip' resolves to version-specific name."""
        result = _resolve_adaptor_name('siglip', 'c-radio_v3-l')
        # Should resolve to the default or model-specific siglip name
        assert result in ['siglip2', 'siglip2-g']

    def test_unknown_adaptor_warns(self):
        """Test that unknown adaptor name is returned with warning."""
        result = _resolve_adaptor_name('unknown_adaptor', 'c-radio_v3-l')
        assert result == 'unknown_adaptor'

    def test_internal_names_pass_through(self):
        """Test that internal adaptor names pass through."""
        assert _resolve_adaptor_name('siglip2', 'c-radio_v3-l') == 'siglip2'
        assert _resolve_adaptor_name('siglip2-g', 'c-radio_v3-l') == 'siglip2-g'
