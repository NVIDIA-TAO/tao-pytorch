# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RADIO adapter components."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from nvidia_tao_pytorch.multimodal.clip.model.logit_calibration import (
    DEFAULT_MAX_LOGIT_SCALE,
    register_logit_checkpoint_guard,
)


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


class MockOpenCLIPTextModel(nn.Module):
    """Production-faithful RADIO OpenCLIP text model with calibration."""

    def __init__(self, has_bias=True, with_norm=True):
        super().__init__()
        self.transformer = nn.Linear(2, 2, bias=False)
        if with_norm:
            self.ln_final = nn.LayerNorm(2)
        self.logit_scale = nn.Parameter(torch.tensor([3.75]))
        if has_bias:
            self.logit_bias = nn.Parameter(torch.tensor([-4.5]))


class MockClipAdaptor(nn.Module):
    """RADIO clip adaptor retaining its native OpenCLIP model."""

    def __init__(self, has_bias=True, with_norm=True):
        super().__init__()
        self.oc_model = MockOpenCLIPTextModel(
            has_bias=has_bias,
            with_norm=with_norm,
        )
        self.tokenizer = lambda texts: torch.zeros(
            len(texts), 2, dtype=torch.long
        )


class MockSigLIPAdaptor(nn.Module):
    """RADIO SigLIP2 adaptor retaining only the text tower."""

    def __init__(self):
        super().__init__()
        self.text_model = nn.Linear(2, 2, bias=False)
        self.tokenizer = MagicMock()
        self.tokenizer._proc = MagicMock()


class MockLoadedRADIO(nn.Module):
    """Minimal loaded RADIO model with registered adaptors."""

    def __init__(self, name, adaptor):
        super().__init__()
        self.model = nn.Linear(2, 2, bias=False)
        self.adaptors = nn.ModuleDict({name: adaptor})
        self.make_preprocessor_external = MagicMock()


def _build_radio(name='clip', adaptor=None, **kwargs):
    """Build CRADIO around a production-shaped adaptor mock."""
    from nvidia_tao_pytorch.multimodal.clip.model.adapters.radio import CRADIO

    if adaptor is None:
        adaptor = MockClipAdaptor()
    radio = MockLoadedRADIO(name, adaptor)
    with patch('torch.hub.load', return_value=radio):
        model = CRADIO(
            model_version='c-radio_v3-h',
            adaptor_name=name,
            **kwargs,
        )
    return model, radio, adaptor


def _train_config():
    """Return a small optimizer configuration."""
    return SimpleNamespace(
        optim=SimpleNamespace(
            optimizer_type='adamw',
            vision_lr=1e-3,
            text_lr=2e-3,
            weight_decay=0.01,
            betas=[0.9, 0.999],
            eps=1e-8,
            warmup_steps=0,
            scheduler='constant',
        )
    )


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
        """CRADIO externalizes preprocessing exactly once."""
        _, radio, _ = _build_radio(loss_type='clip')

        radio.make_preprocessor_external.assert_called_once()

    def test_clip_reuses_one_native_pair_in_state_and_optimizer(self):
        """RADIO OpenCLIP calibration is canonical and optimized once."""
        from nvidia_tao_pytorch.multimodal.clip.utils.utils import (
            build_optimizer,
        )

        model, _, adaptor = _build_radio(loss_type='clip')
        source = adaptor.oc_model
        state = model.state_dict()
        optimizer = build_optimizer(model, _train_config())
        optimizer_params = [
            param for group in optimizer.param_groups
            for param in group['params']
        ]

        assert model.get_logit_scale_parameter() is source.logit_scale
        assert model.get_logit_bias_parameter() is source.logit_bias
        assert source.logit_scale.shape == torch.Size([])
        assert source.logit_bias.shape == torch.Size([])
        assert [key for key in state if key.endswith('logit_scale')] == [
            'radio_model.adaptors.clip.oc_model.logit_scale'
        ]
        assert [key for key in state if key.endswith('logit_bias')] == [
            'radio_model.adaptors.clip.oc_model.logit_bias'
        ]
        assert not any(key.startswith('adaptor.') for key in state)
        assert sum(param is source.logit_scale for param in optimizer_params) == 1
        assert sum(param is source.logit_bias for param in optimizer_params) == 1
        register_logit_checkpoint_guard(model)
        result = model.load_state_dict(state, strict=True)
        assert not result.missing_keys
        assert not result.unexpected_keys

    @pytest.mark.parametrize('loss_type', ['clip', 'siglip'])
    def test_clip_preserves_native_bias_absence(self, loss_type):
        """RADIO preserves a native OpenCLIP model without logit bias."""
        from nvidia_tao_pytorch.multimodal.clip.utils.utils import (
            build_optimizer,
        )

        adaptor = MockClipAdaptor(has_bias=False)
        model, _, _ = _build_radio(
            adaptor=adaptor,
            loss_type=loss_type,
        )
        source = adaptor.oc_model
        state = model.state_dict()
        optimizer = build_optimizer(model, _train_config())
        optimizer_params = [
            param for group in optimizer.param_groups
            for param in group['params']
        ]

        assert model.get_logit_scale_parameter() is source.logit_scale
        assert model.get_logit_bias_parameter() is None
        assert not any(key.endswith('logit_bias') for key in state)
        assert sum(
            param is source.logit_scale for param in optimizer_params
        ) == 1

    def test_clip_calibration_stays_trainable_when_text_is_frozen(self):
        """Freezing RADIO text excludes its native calibration parameters."""
        model, _, adaptor = _build_radio(
            loss_type='clip', freeze_text_encoder=True
        )
        source = adaptor.oc_model

        assert source.logit_scale.requires_grad
        assert source.logit_bias.requires_grad
        assert not source.transformer.weight.requires_grad
        text_params = dict(model.text_named_parameters()).values()
        assert all(param is not source.logit_scale for param in text_params)
        assert all(param is not source.logit_bias for param in text_params)

    def test_siglip2_adaptor_uses_one_local_fallback_pair(self):
        """RADIO SigLIP2 text-only adaptors use adapter-owned fallback."""
        model, _, _ = _build_radio(
            name='siglip2',
            adaptor=MockSigLIPAdaptor(),
            loss_type='siglip',
        )
        state = model.state_dict()

        assert [key for key in state if key.endswith('logit_scale')] == [
            'logit_scale'
        ]
        assert [key for key in state if key.endswith('logit_bias')] == [
            'logit_bias'
        ]
        assert model.get_logit_scale_parameter().item() == pytest.approx(2.3026)
        assert model.get_logit_bias_parameter().item() == pytest.approx(-10.0)

    def test_local_override_does_not_raise_fallback_ceiling(self):
        """RADIO local calibration keeps the historical safety cap."""
        model, _, _ = _build_radio(
            name='siglip2',
            adaptor=MockSigLIPAdaptor(),
            loss_type='siglip',
            logit_scale_init=8.0,
        )

        assert model.get_logit_scale_parameter().item() == pytest.approx(8.0)
        assert model.logit_scale_max == pytest.approx(
            DEFAULT_MAX_LOGIT_SCALE
        )
        model.clamp_logit_scale_()
        assert model.get_logit_scale_parameter().item() == pytest.approx(
            DEFAULT_MAX_LOGIT_SCALE
        )

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
