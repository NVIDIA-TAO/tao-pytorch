# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RADIO adapter components."""

from unittest.mock import MagicMock
import pytest
import torch
import torch.nn as nn


class MockRADIOModel(nn.Module):
    """Mock RADIO model for testing."""

    def __init__(self, embed_dim=1280):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = 16
        self.max_resolution = 2048
        self.preferred_resolution = MagicMock()
        self.preferred_resolution.height = 512
        self.preferred_resolution.width = 512
        self.adaptor_names = ['siglip', 'clip']
        self._dummy_param = nn.Parameter(torch.zeros(1))
        self.input_conditioner = MagicMock()

    def make_preprocessor_external(self):
        pass

    def forward(self, x, feature_fmt='NCHW'):
        b = x.shape[0]
        summary = torch.randn(b, self.embed_dim)
        features = torch.randn(b, 256, self.embed_dim)
        return summary, features

    def parameters(self):
        return [self._dummy_param]

    def named_parameters(self):
        yield '_dummy_param', self._dummy_param


class MockAdaptorHead(nn.Module):
    """Mock adaptor head."""

    def __init__(self, embed_dim=768):
        super().__init__()
        self.output_dim = embed_dim
        self.linear = nn.Linear(1280, embed_dim)

    def forward(self, x):
        return self.linear(x)


class MockTextModel(nn.Module):
    """Mock text model."""

    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(512, embed_dim)

    def forward(self, x):
        return torch.randn(x.shape[0], self.embed_dim)


@pytest.mark.multimodal_unit
class TestRADIOAdapterMocked:
    """Test RADIO adapter mock components."""

    def test_radio_model_structure(self):
        """Test that mock RADIO model has correct structure."""
        mock_radio = MockRADIOModel()

        assert hasattr(mock_radio, 'preferred_resolution')
        assert hasattr(mock_radio, 'adaptor_names')
        assert hasattr(mock_radio, 'patch_size')
        assert mock_radio.embed_dim == 1280

    def test_make_preprocessor_external_called(self):
        """Test that CRADIO.__init__ calls make_preprocessor_external() to
        avoid double-normalizing images (external transforms + internal
        input_conditioner)."""
        from unittest.mock import patch, MagicMock

        mock_radio = MockRADIOModel()
        mock_radio.make_preprocessor_external = MagicMock()

        mock_adaptor = MagicMock()
        mock_adaptor.tokenizer = MagicMock()
        mock_adaptor.parameters.return_value = []
        mock_radio.adaptors = {'clip': mock_adaptor}

        with patch('torch.hub.load', return_value=mock_radio):
            from nvidia_tao_pytorch.multimodal.clip.model.adapters.radio import CRADIO
            model = CRADIO(
                model_version='c-radio_v3-h',
                adaptor_name='clip',
            )

        mock_radio.make_preprocessor_external.assert_called_once()

    def test_radio_model_forward(self):
        """Test mock RADIO model forward."""
        mock_radio = MockRADIOModel()

        images = torch.randn(4, 3, 512, 512)
        summary, features = mock_radio(images)

        assert summary.shape == (4, 1280)
        assert features.shape == (4, 256, 1280)

    def test_adaptor_head_forward(self):
        """Test adaptor head forward."""
        adaptor = MockAdaptorHead()

        features = torch.randn(4, 1280)
        output = adaptor(features)

        assert output.shape[0] == 4
        assert output.shape[1] == 768

    def test_text_model_forward(self):
        """Test text model forward."""
        text_model = MockTextModel()
        input_ids = torch.randint(0, 1000, (4, 64))
        output = text_model(input_ids)

        assert output.shape == (4, 768)


@pytest.mark.multimodal_unit
class TestRADIOAdaptorNameResolution:
    """Test adaptor name resolution logic."""

    def test_siglip_adaptor_name(self):
        """Test SigLIP adaptor name resolves to siglip2."""
        from nvidia_tao_pytorch.multimodal.clip.model.builders import _resolve_adaptor_name

        result = _resolve_adaptor_name('siglip', 'c-radio_v3-l')
        assert result == 'siglip2'

    def test_siglip2_adaptor_name(self):
        """Test SigLIP2 adaptor name stays siglip2."""
        from nvidia_tao_pytorch.multimodal.clip.model.builders import _resolve_adaptor_name

        result = _resolve_adaptor_name('siglip2', 'c-radio_v3-l')
        assert result == 'siglip2'

    def test_clip_adaptor_name(self):
        """Test CLIP adaptor name stays clip."""
        from nvidia_tao_pytorch.multimodal.clip.model.builders import _resolve_adaptor_name

        result = _resolve_adaptor_name('clip', 'c-radio_v3-l')
        assert result == 'clip'

    def test_dfn_adaptor_alias(self):
        """Test DFN adaptor alias resolves to clip."""
        from nvidia_tao_pytorch.multimodal.clip.model.builders import _resolve_adaptor_name

        result = _resolve_adaptor_name('dfn', 'c-radio_v3-l')
        assert result == 'clip'


@pytest.mark.multimodal_unit
class TestRADIOModelConfigs:
    """Test RADIO model configurations."""

    def test_radio_configs_exist(self):
        """Test RADIO configs are defined."""
        from nvidia_tao_pytorch.multimodal.clip.utils.model_configs import radio_model_configs

        assert isinstance(radio_model_configs, (list, tuple, set, dict))

    def test_siglip2_configs_exist(self):
        """Test SigLIP2 configs are defined."""
        from nvidia_tao_pytorch.multimodal.clip.utils.model_configs import siglip2_model_configs

        assert isinstance(siglip2_model_configs, (list, tuple, set, dict))

    def test_openclip_configs_exist(self):
        """Test OpenCLIP configs are defined."""
        from nvidia_tao_pytorch.multimodal.clip.utils.model_configs import openclip_model_configs

        assert isinstance(openclip_model_configs, (list, tuple, set, dict))


@pytest.mark.multimodal_unit
class TestRADIOBuildHelpers:
    """Test RADIO build helper functions."""

    def test_parse_aug_config_none(self):
        """Test parse_aug_config with None."""
        from nvidia_tao_pytorch.multimodal.clip.model.builders import _parse_aug_config

        result = _parse_aug_config(None)
        assert result is None

    def test_parse_aug_config_empty(self):
        """Test parse_aug_config with empty dict."""
        from nvidia_tao_pytorch.multimodal.clip.model.builders import _parse_aug_config

        result = _parse_aug_config({})
        assert result is not None
        assert 'gray_scale_prob' in result
        assert 'color_jitter' in result

    def test_parse_aug_config_with_values(self):
        """Test parse_aug_config with values."""
        from nvidia_tao_pytorch.multimodal.clip.model.builders import _parse_aug_config

        cfg = {
            'grayscale_prob': 0.5,
            'color_jitter': [0.1, 0.1, 0.1, 0.05]
        }
        result = _parse_aug_config(cfg)

        assert result['gray_scale_prob'] == 0.5
        assert result['color_jitter'] == [0.1, 0.1, 0.1, 0.05]
