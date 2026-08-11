# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for canonical CLIP logit calibration ownership."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from open_clip.loss import SigLipLoss

from nvidia_tao_pytorch.cv.backbone_v2.siglip2 import SigLIP2Wrapper
from nvidia_tao_pytorch.multimodal.clip.loss.masked_siglip_loss import (
    MetadataMaskedSigLipLoss,
)
from nvidia_tao_pytorch.multimodal.clip.model import clip as clip_module
from nvidia_tao_pytorch.multimodal.clip.model.logit_calibration import (
    DEFAULT_MAX_LOGIT_SCALE,
    clamp_logit_scale_,
    configure_source_logit_calibration,
    named_logit_parameters,
    register_logit_checkpoint_guard,
)
from nvidia_tao_pytorch.multimodal.clip.model.adapters.openclip import OpenCLIP
from nvidia_tao_pytorch.multimodal.clip.model.adapters.siglip2 import SigLIP2
from nvidia_tao_pytorch.multimodal.clip.model.pl_clip_model import CLIPPlModel
from nvidia_tao_pytorch.multimodal.clip.utils.utils import build_optimizer
from nvidia_tao_pytorch.multimodal.clip.scripts.export import (
    CLIPCombinedEncoder,
    CLIPVisionEncoder,
)


_MISSING = object()


class _TinyTower(nn.Module):
    """Small projection tower used by adapter tests."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False)


class _TinySigLIP2Source(nn.Module):
    """Minimal Hugging Face-like SigLIP2 source model."""

    def __init__(self, scale=4.6994, bias=-15.9324):
        super().__init__()
        self.vision_model = _TinyTower()
        self.text_model = _TinyTower()
        if scale is not _MISSING:
            self.logit_scale = nn.Parameter(torch.tensor([scale]))
        if bias is not _MISSING:
            self.logit_bias = nn.Parameter(torch.tensor([bias]))


class _TinySigLIP2Backbone(nn.Module):
    """Minimal TAO SigLIP2 backbone wrapper."""

    def __init__(self, source):
        super().__init__()
        self.inner = source

    def forward(self, values, return_features=False, return_logits=True):
        del return_features, return_logits
        return self.inner.vision_model.projection(values.float())

    def encode_text(self, text, normalize=False):
        del normalize
        return self.inner.text_model.projection(text['input_ids'].float())


class _TinyOpenCLIPSource(nn.Module):
    """Minimal native OpenCLIP model."""

    def __init__(self, scale=3.75, bias=-4.5):
        super().__init__()
        self.visual = nn.Linear(2, 2, bias=False)
        self.transformer = nn.Linear(2, 2, bias=False)
        if scale is not _MISSING:
            self.logit_scale = nn.Parameter(torch.tensor(scale))
        if bias is not _MISSING:
            self.logit_bias = nn.Parameter(torch.tensor(bias))

    def encode_image(self, values):
        return self.visual(values.float())

    def encode_text(self, values, normalize=False):
        del normalize
        return self.transformer(values.float())


class _TinyOpenCLIPBackbone(nn.Module):
    """Minimal TAO OpenCLIP backbone wrapper."""

    def __init__(self, source):
        super().__init__()
        self.model = source
        self._model_name = 'tiny-openclip'
        self.tokenizer = lambda texts: torch.zeros(len(texts), 2, dtype=torch.long)

    def forward_pre_logits(self, values):
        return self.model.encode_image(values)

    def encode_text(self, values, normalize=False):
        return self.model.encode_text(values, normalize=normalize)

    def set_grad_checkpointing(self, enable=True):
        del enable


def _siglip2_adapter(
    scale=4.6994,
    bias=-15.9324,
    scale_override=None,
    bias_override=None,
):
    source = _TinySigLIP2Source(scale=scale, bias=bias)
    backbone = _TinySigLIP2Backbone(source)
    adapter = SigLIP2(
        backbone,
        processor=MagicMock(),
        logit_scale_init=scale_override,
        logit_bias_init=bias_override,
    )
    return adapter, source


def _openclip_adapter(
    scale=3.75,
    bias=-4.5,
    scale_override=None,
    bias_override=None,
    loss_type='siglip',
):
    source = _TinyOpenCLIPSource(scale=scale, bias=bias)
    backbone = _TinyOpenCLIPBackbone(source)
    adapter = OpenCLIP(
        backbone,
        logit_scale_init=scale_override,
        logit_bias_init=bias_override,
        loss_type=loss_type,
    )
    return adapter, source


def _train_config():
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


def _optimizer_parameters(optimizer):
    return [
        parameter
        for group in optimizer.param_groups
        for parameter in group['params']
    ]


def _forward_inputs():
    values = torch.tensor([[1.0, 0.5], [0.25, 1.0]])
    return values, {'input_ids': values.clone()}


@pytest.mark.multimodal_unit
class TestSourceCalibrationConfiguration:
    """Test source-preserving configuration rules."""

    def test_none_preserves_pretrained_parameter_objects_and_values(self):
        """None must retain both learned source parameters exactly."""
        source = _TinySigLIP2Source()
        scale = source.logit_scale
        bias = source.logit_bias

        configured = configure_source_logit_calibration(source)

        assert configured == (scale, bias)
        assert source.logit_scale is scale
        assert source.logit_bias is bias
        assert source.logit_scale.item() == pytest.approx(4.6994)
        assert source.logit_bias.item() == pytest.approx(-15.9324)
        assert source.logit_scale.shape == torch.Size([])
        assert source.logit_bias.shape == torch.Size([])

    @pytest.mark.parametrize(
        ('scale_override', 'bias_override', 'expected_scale', 'expected_bias'),
        [
            (1.25, None, 1.25, -15.9324),
            (None, -2.5, 4.6994, -2.5),
            (1.25, -2.5, 1.25, -2.5),
        ],
    )
    def test_overrides_update_each_canonical_parameter_in_place(
        self,
        scale_override,
        bias_override,
        expected_scale,
        expected_bias,
    ):
        """Per-field overrides must not replace native Parameter objects."""
        source = _TinySigLIP2Source()
        scale = source.logit_scale
        bias = source.logit_bias

        configure_source_logit_calibration(
            source,
            logit_scale_init=scale_override,
            logit_bias_init=bias_override,
        )

        assert source.logit_scale is scale
        assert source.logit_bias is bias
        assert source.logit_scale.item() == pytest.approx(expected_scale)
        assert source.logit_bias.item() == pytest.approx(expected_bias)

    def test_missing_required_values_use_loss_family_fallback_on_source(self):
        """A source without calibration owns the required fallback pair."""
        source = _TinySigLIP2Source(
            scale=_MISSING, bias=_MISSING
        ).double()

        configure_source_logit_calibration(source, loss_type='siglip')

        assert isinstance(source.logit_scale, nn.Parameter)
        assert isinstance(source.logit_bias, nn.Parameter)
        assert source.logit_scale.item() == pytest.approx(2.3026)
        assert source.logit_bias.item() == pytest.approx(-10.0)
        assert source.logit_scale.dtype == torch.float64
        assert source.logit_bias.dtype == torch.float64

    @pytest.mark.parametrize('missing_name', ['logit_scale', 'logit_bias'])
    def test_each_missing_required_parameter_gets_only_its_fallback(
        self, missing_name
    ):
        """A missing field must not replace the learned sibling field."""
        source = _TinySigLIP2Source()
        preserved_name = (
            'logit_bias' if missing_name == 'logit_scale' else 'logit_scale'
        )
        preserved = getattr(source, preserved_name)
        delattr(source, missing_name)

        configure_source_logit_calibration(source, loss_type='siglip')

        assert getattr(source, preserved_name) is preserved
        if missing_name == 'logit_scale':
            assert source.logit_scale.item() == pytest.approx(2.3026)
            assert source.logit_bias.item() == pytest.approx(-15.9324)
        else:
            assert source.logit_scale.item() == pytest.approx(4.6994)
            assert source.logit_bias.item() == pytest.approx(-10.0)

    def test_clip_source_without_bias_stays_biasless(self):
        """Bias-optional CLIP models must not gain a generic bias."""
        source = _TinyOpenCLIPSource(bias=_MISSING)

        _, bias = configure_source_logit_calibration(
            source,
            loss_type='clip',
            bias_required=False,
        )

        assert bias is None
        assert not hasattr(source, 'logit_bias')

    def test_explicit_bias_override_creates_optional_source_bias(self):
        """An explicit override creates bias on an otherwise biasless source."""
        source = _TinyOpenCLIPSource(bias=_MISSING)
        scale = source.logit_scale

        _, bias = configure_source_logit_calibration(
            source,
            logit_bias_init=-2.5,
            loss_type='clip',
            bias_required=False,
        )

        assert source.logit_scale is scale
        assert bias is source.logit_bias
        assert isinstance(source.logit_bias, nn.Parameter)
        assert source.logit_bias.item() == pytest.approx(-2.5)

    def test_invalid_source_parameters_fail_clearly(self):
        """Non-Parameter and non-scalar source values are rejected."""
        non_parameter = _TinySigLIP2Source()
        del non_parameter.logit_scale
        non_parameter.logit_scale = torch.tensor(1.0)
        with pytest.raises(TypeError, match='logit_scale must be an nn.Parameter'):
            configure_source_logit_calibration(non_parameter)

        non_scalar = _TinySigLIP2Source()
        non_scalar.logit_scale = nn.Parameter(torch.ones(2))
        with pytest.raises(ValueError, match='logit_scale must be scalar'):
            configure_source_logit_calibration(non_scalar)

    def test_explicit_scale_override_does_not_raise_native_ceiling(self):
        """Only the source value, not a config override, may raise the cap."""
        source = _TinySigLIP2Source(scale=3.0)

        configure_source_logit_calibration(
            source,
            logit_scale_init=8.0,
        )

        assert source.logit_scale.item() == pytest.approx(8.0)
        assert source.logit_scale_max == pytest.approx(
            DEFAULT_MAX_LOGIT_SCALE
        )
        clamp_logit_scale_(source)
        assert source.logit_scale.item() == pytest.approx(
            DEFAULT_MAX_LOGIT_SCALE
        )

    def test_missing_source_scale_override_uses_default_ceiling(self):
        """A newly created config value is not treated as pretrained state."""
        source = _TinySigLIP2Source(scale=_MISSING)

        configure_source_logit_calibration(
            source,
            logit_scale_init=8.0,
        )

        assert source.logit_scale_max == pytest.approx(
            DEFAULT_MAX_LOGIT_SCALE
        )


@pytest.mark.multimodal_unit
class TestSigLIP2CanonicalCalibration:
    """Test canonical Hugging Face SigLIP2 ownership."""

    def test_fresh_model_has_one_native_pair_used_by_forward(self):
        """Forward and state dict must use only the learned HF pair."""
        adapter, source = _siglip2_adapter()
        image, text = _forward_inputs()

        outputs = adapter(image=image, text=text)
        named = dict(adapter.named_logit_parameters())
        state = adapter.state_dict()

        assert adapter.logit_scale is source.logit_scale
        assert adapter.logit_bias is source.logit_bias
        assert named['logit_scale'] is source.logit_scale
        assert named['logit_bias'] is source.logit_bias
        assert source.logit_scale.shape == torch.Size([])
        assert source.logit_bias.shape == torch.Size([])
        assert outputs[2].shape == torch.Size([])
        assert outputs[3].shape == torch.Size([])
        assert outputs[2].item() == pytest.approx(math.exp(4.6994))
        assert outputs[3].item() == pytest.approx(-15.9324)
        assert [key for key in state if key.endswith('logit_scale')] == [
            'backbone.inner.logit_scale'
        ]
        assert [key for key in state if key.endswith('logit_bias')] == [
            'backbone.inner.logit_bias'
        ]

    def test_optimizer_updates_native_pair_once_and_all_paths_agree(self):
        """Training, zero-shot, and optimizer access share the HF objects."""
        adapter, source = _siglip2_adapter()
        optimizer = build_optimizer(adapter, _train_config())
        parameters = _optimizer_parameters(optimizer)
        image, text = _forward_inputs()

        outputs = adapter(image=image, text=text)
        similarity = outputs[0] @ outputs[1].T
        zero_shot = SigLIP2Wrapper.zero_shot_postproc(
            adapter.backbone, similarity
        )
        expected = similarity * outputs[2] + outputs[3]

        assert sum(param is source.logit_scale for param in parameters) == 1
        assert sum(param is source.logit_bias for param in parameters) == 1
        torch.testing.assert_close(zero_shot, expected)

        old_scale = source.logit_scale.detach().clone()
        old_bias = source.logit_bias.detach().clone()
        (outputs[2] + outputs[3]).backward()
        optimizer.step()

        assert adapter.logit_scale is source.logit_scale
        assert adapter.logit_bias is source.logit_bias
        assert not torch.equal(source.logit_scale, old_scale)
        assert not torch.equal(source.logit_bias, old_bias)

    def test_siglip_scale_ceiling_preserves_pretrained_value(self):
        """The ceiling preserves pretrained scale while remaining bounded."""
        adapter, source = _siglip2_adapter(scale=5.5)

        assert adapter.logit_scale_max == pytest.approx(5.5)
        source.logit_scale.data.fill_(6.0)
        clamp_logit_scale_(adapter)
        assert source.logit_scale.item() == pytest.approx(5.5)

        source.logit_scale.data.fill_(-1.0)
        clamp_logit_scale_(adapter)
        assert source.logit_scale.item() == pytest.approx(0.0)


@pytest.mark.multimodal_unit
class TestOpenCLIPCanonicalCalibration:
    """Test native OpenCLIP ownership and optional bias."""

    def test_native_pair_is_reused_without_duplicate_state_keys(self):
        """Wrapped OpenCLIP calibration remains registered only natively."""
        adapter, source = _openclip_adapter()
        state = adapter.state_dict()

        assert adapter.logit_scale is source.logit_scale
        assert adapter.logit_bias is source.logit_bias
        assert [key for key in state if key.endswith('logit_scale')] == [
            'backbone.model.logit_scale'
        ]
        assert [key for key in state if key.endswith('logit_bias')] == [
            'backbone.model.logit_bias'
        ]

    @pytest.mark.parametrize('loss_type', ['clip', 'siglip'])
    def test_bias_optional_clip_forward_returns_three_values(
        self, loss_type
    ):
        """Native CLIP without bias remains supported by the common forward."""
        adapter, source = _openclip_adapter(
            bias=_MISSING,
            loss_type=loss_type,
        )
        image, text = _forward_inputs()

        outputs = adapter(image=image, text=text)

        assert len(outputs) == 3
        assert adapter.logit_bias is None
        assert not hasattr(source, 'logit_bias')
        assert list(dict(named_logit_parameters(adapter))) == ['logit_scale']

    def test_clip_scale_uses_log_100_ceiling(self):
        """The CLIP-family policy retains its historical log(100) maximum."""
        adapter, source = _openclip_adapter(
            scale=3.0,
            bias=_MISSING,
            loss_type='clip',
        )
        source.logit_scale.data.fill_(6.0)

        clamp_logit_scale_(adapter)

        assert source.logit_scale.item() == pytest.approx(
            DEFAULT_MAX_LOGIT_SCALE
        )

    def test_raw_openclip_optimizer_contains_native_calibration_once(self):
        """The unwrapped OpenCLIP fallback follows the same optimizer contract."""
        model = _TinyOpenCLIPSource(scale=3.5, bias=_MISSING)
        configure_source_logit_calibration(
            model,
            loss_type='clip',
            bias_required=False,
        )

        optimizer = build_optimizer(model, _train_config())
        parameters = _optimizer_parameters(optimizer)

        assert sum(param is model.logit_scale for param in parameters) == 1
        assert all(name != 'logit_bias' for name, _ in model.named_parameters())
        model.logit_scale.data.fill_(-1.0)
        clamp_logit_scale_(model)
        assert model.logit_scale.item() == pytest.approx(0.0)
        model.logit_scale.data.fill_(6.0)
        clamp_logit_scale_(model)
        assert model.logit_scale.item() == pytest.approx(
            DEFAULT_MAX_LOGIT_SCALE
        )


@pytest.mark.multimodal_unit
class TestLightningCalibrationPolicy:
    """Test calibration bounds at Lightning lifecycle integration points."""

    @staticmethod
    def _module(model):
        module = CLIPPlModel.__new__(CLIPPlModel)
        from pytorch_lightning import LightningModule

        LightningModule.__init__(module)
        module.model = model
        return module

    def test_optimizer_step_clamps_after_parameter_update(self):
        """The stored post-step scale must satisfy the model ceiling."""
        model = _TinyOpenCLIPSource(scale=3.0, bias=_MISSING)
        configure_source_logit_calibration(
            model, loss_type='clip', bias_required=False
        )
        module = self._module(model)
        optimizer = torch.optim.SGD([model.logit_scale], lr=0.1)
        model.logit_scale.grad = torch.tensor(-100.0)

        module.optimizer_step(0, 0, optimizer, lambda: None)

        assert model.logit_scale.item() == pytest.approx(
            DEFAULT_MAX_LOGIT_SCALE
        )

    def test_biasless_siglip_loss_receives_explicit_none_bias(self):
        """SigLIP loss supports a source model without a bias parameter."""
        image_features = torch.eye(2)
        text_features = torch.eye(2)
        logit_scale = torch.tensor(2.0)
        module = SimpleNamespace(loss=SigLipLoss())

        loss, _, _, _ = CLIPPlModel._backward(
            module,
            (image_features, text_features, logit_scale),
        )

        assert torch.isfinite(loss)

    def test_biasless_masked_siglip_loss_accepts_three_value_output(self):
        """Metadata masking supports native biasless OpenCLIP output."""
        image_features = torch.eye(2)
        text_features = torch.eye(2)
        logit_scale = torch.tensor(2.0)
        metadata = {
            'image_attr_values': torch.tensor([[1], [2]]),
            'text_attr_values': torch.tensor([[1], [2]]),
        }
        module = SimpleNamespace(loss=MetadataMaskedSigLipLoss())

        loss, _, _, _ = CLIPPlModel._backward(
            module,
            (image_features, text_features, logit_scale),
            (None, None, metadata),
        )

        assert torch.isfinite(loss)

    def test_train_start_refreshes_raw_model_resume_ceiling(self):
        """A restored raw model is not clamped by a stale constructor bound."""
        model = _TinyOpenCLIPSource(scale=5.5, bias=_MISSING)
        configure_source_logit_calibration(
            model, loss_type='clip', bias_required=False
        )
        model.logit_scale_max = DEFAULT_MAX_LOGIT_SCALE
        module = self._module(model)
        datamodule = SimpleNamespace(resume_step=None)
        module._trainer = SimpleNamespace(
            datamodule=datamodule,
            global_step=7,
        )

        module.on_load_checkpoint({})
        module.on_train_start()

        assert model.logit_scale_max == pytest.approx(5.5)
        assert datamodule.resume_step == 7

    def test_fresh_train_start_does_not_trust_unrestored_scale(self):
        """Fresh explicit state cannot enlarge the clamp at train start."""
        model = _TinyOpenCLIPSource(scale=8.0, bias=_MISSING)
        model.logit_scale_max = DEFAULT_MAX_LOGIT_SCALE
        module = self._module(model)
        datamodule = SimpleNamespace(resume_step=None)
        module._trainer = SimpleNamespace(
            datamodule=datamodule,
            global_step=0,
        )

        module.on_train_start()

        assert model.logit_scale_max == pytest.approx(
            DEFAULT_MAX_LOGIT_SCALE
        )


@pytest.mark.multimodal_unit
class TestCheckpointLayoutGuard:
    """Test explicit rejection of obsolete calibration checkpoints."""

    @pytest.mark.parametrize('layout', ['siglip2', 'openclip', 'local'])
    def test_current_canonical_state_loads_strictly(self, layout):
        """The guard accepts each current single-owner checkpoint layout."""
        if layout == 'siglip2':
            model, _ = _siglip2_adapter()
        elif layout == 'openclip':
            model, _ = _openclip_adapter()
        else:
            model = nn.Module()
            model.logit_scale = nn.Parameter(torch.tensor(2.3026))
            model.logit_bias = nn.Parameter(torch.tensor(-10.0))
        register_logit_checkpoint_guard(model)
        state = model.state_dict()

        result = model.load_state_dict(state, strict=True)

        assert not result.missing_keys
        assert not result.unexpected_keys

    def test_legacy_radio_adaptor_alias_fails_with_targeted_message(self):
        """The removed C-RADIO module alias gets the compatibility error."""
        adapter, _ = _siglip2_adapter()
        register_logit_checkpoint_guard(adapter)
        state = adapter.state_dict()
        state['adaptor.legacy.weight'] = torch.ones(1)

        with pytest.raises(
            RuntimeError,
            match='predates CLIP logit calibration ownership unification',
        ):
            adapter.load_state_dict(state, strict=True)

    def test_legacy_outer_pair_fails_with_targeted_message(self):
        """A duplicate outer pair is rejected instead of guessed at."""
        adapter, _ = _siglip2_adapter()
        register_logit_checkpoint_guard(adapter)
        wrapper = nn.Module()
        wrapper.model = adapter
        state = wrapper.state_dict()
        state['model.logit_scale'] = torch.tensor(2.3026)
        state['model.logit_bias'] = torch.tensor(-10.0)

        with pytest.raises(
            RuntimeError,
            match='predates CLIP logit calibration ownership unification',
        ):
            wrapper.load_state_dict(state, strict=True)

    def test_legacy_scalar_shape_fails_with_targeted_message(self):
        """Old one-element calibration tensors get the same clear error."""
        adapter, _ = _siglip2_adapter()
        register_logit_checkpoint_guard(adapter)
        state = adapter.state_dict()
        state['backbone.inner.logit_scale'] = state[
            'backbone.inner.logit_scale'
        ].reshape(1)

        with pytest.raises(
            RuntimeError,
            match='predates CLIP logit calibration ownership unification',
        ):
            adapter.load_state_dict(state, strict=True)


@pytest.mark.multimodal_unit
class TestCalibrationDispatch:
    """Test top-level model dispatch preserves the public None semantics."""

    @staticmethod
    def _config(
        model_type,
        loss_type='siglip',
        scale=None,
        bias=None,
        freeze_vision=False,
        freeze_text=False,
    ):
        return SimpleNamespace(
            model=SimpleNamespace(
                type=model_type,
                image_size=256,
                init_logit_scale=scale,
                init_logit_bias=bias,
                freeze_vision_encoder=freeze_vision,
                freeze_text_encoder=freeze_text,
                canonicalize_text=False,
                adaptor_name=None,
            ),
            train=SimpleNamespace(loss_type=loss_type),
            dataset=SimpleNamespace(augmentation={}),
        )

    @pytest.mark.parametrize(
        ('model_type', 'builder_name'),
        [
            ('siglip2-so400m-patch16-256', 'build_siglip2_model'),
            ('ViT-L-14-SigLIP-CLIPA-224', 'build_openclip_model'),
            ('c-radio_v3-l', 'build_radio_model'),
        ],
    )
    def test_none_and_loss_type_reach_each_adapter_builder(
        self, model_type, builder_name
    ):
        """Dispatch must not replace None before ownership is known."""
        model = MagicMock()
        with (
            patch.object(
                clip_module,
                builder_name,
                return_value=(model, 'train', 'val', 'tokenizer'),
            ) as builder,
            patch.object(
                clip_module, 'register_logit_checkpoint_guard'
            ) as register_guard,
        ):
            clip_module.build_model(self._config(model_type))

        assert builder.call_args.kwargs['logit_scale_init'] is None
        assert builder.call_args.kwargs['logit_bias_init'] is None
        assert builder.call_args.kwargs['loss_type'] == 'siglip'
        register_guard.assert_called_once_with(model)

    @pytest.mark.parametrize(
        ('model_type', 'builder_name'),
        [
            ('siglip2-so400m-patch16-256', 'build_siglip2_model'),
            ('ViT-L-14-SigLIP-CLIPA-224', 'build_openclip_model'),
            ('c-radio_v3-l', 'build_radio_model'),
        ],
    )
    @pytest.mark.parametrize(
        ('scale', 'bias'),
        [(1.25, None), (None, -2.5), (1.25, -2.5)],
    )
    def test_per_field_overrides_reach_each_builder(
        self, model_type, builder_name, scale, bias
    ):
        """Dispatch preserves every explicit per-field override."""
        with patch.object(
            clip_module,
            builder_name,
            return_value=(MagicMock(), 'train', 'val', 'tokenizer'),
        ) as builder:
            clip_module.build_model(
                self._config(model_type, scale=scale, bias=bias)
            )

        assert builder.call_args.kwargs['logit_scale_init'] == scale
        assert builder.call_args.kwargs['logit_bias_init'] == bias

    @pytest.mark.parametrize(
        ('freeze_vision', 'freeze_text'),
        [(True, False), (False, True)],
    )
    def test_raw_openclip_applies_each_encoder_freeze_flag(
        self, freeze_vision, freeze_text
    ):
        """Raw OpenCLIP freezes the selected tower but not calibration."""
        model = _TinyOpenCLIPSource()
        with (
            patch.object(
                clip_module.open_clip,
                'create_model_and_transforms',
                return_value=(model, 'train', 'val'),
            ),
            patch.object(
                clip_module.open_clip,
                'get_tokenizer',
                return_value='tokenizer',
            ),
        ):
            clip_module.build_model(self._config(
                'unit-test-raw-openclip',
                freeze_vision=freeze_vision,
                freeze_text=freeze_text,
            ))

        optimizer = build_optimizer(model, _train_config())
        parameters = _optimizer_parameters(optimizer)
        assert model.visual.weight.requires_grad is (not freeze_vision)
        assert model.transformer.weight.requires_grad is (not freeze_text)
        assert model.logit_scale.requires_grad
        assert model.logit_bias.requires_grad
        assert any(param is model.logit_scale for param in parameters)
        assert any(param is model.logit_bias for param in parameters)
        assert any(
            param is model.visual.weight for param in parameters
        ) is (not freeze_vision)
        assert any(
            param is model.transformer.weight for param in parameters
        ) is (not freeze_text)

    def test_source_owned_registry_defaults_are_none(self):
        """Registry metadata must not advertise generic source overrides."""
        from nvidia_tao_pytorch.multimodal.clip.utils.model_configs import (
            openclip_model_configs,
            siglip2_model_configs,
        )

        for configs in (siglip2_model_configs, openclip_model_configs):
            for config in configs.values():
                assert config['init_logit_scale'] is None
                assert config['init_logit_bias'] is None

    def test_raw_openclip_none_preserves_native_and_override_is_in_place(self):
        """Unwrapped OpenCLIP receives the same source-preserving rules."""
        model = _TinyOpenCLIPSource(scale=4.1, bias=_MISSING)
        scale = model.logit_scale
        with (
            patch.object(
                clip_module.open_clip,
                'create_model_and_transforms',
                return_value=(model, 'train', 'val'),
            ),
            patch.object(
                clip_module.open_clip,
                'get_tokenizer',
                return_value='tokenizer',
            ),
        ):
            clip_module.build_model(
                self._config('unit-test-raw-openclip', loss_type='clip')
            )
            assert model.logit_scale is scale
            assert model.logit_scale.item() == pytest.approx(4.1)
            assert not hasattr(model, 'logit_bias')

            clip_module.build_model(
                self._config(
                    'unit-test-raw-openclip',
                    loss_type='clip',
                    scale=1.5,
                )
            )

        assert model.logit_scale is scale
        assert model.logit_scale.item() == pytest.approx(1.5)


@pytest.mark.multimodal_unit
class TestCalibrationExport:
    """Test deployment export uses the canonical calibration interface."""

    def test_nested_native_parameters_are_exported(self):
        """Export must not require adapter-owned root parameters."""
        class NestedCalibrationModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.source = nn.Module()
                self.source.logit_scale = nn.Parameter(
                    torch.tensor(4.6994)
                )
                self.source.logit_bias = nn.Parameter(
                    torch.tensor(-15.9324)
                )

            def get_logit_scale_parameter(self):
                return self.source.logit_scale

            def get_logit_bias_parameter(self):
                return self.source.logit_bias

            def forward(self, image=None, text=None):
                del text
                return {'image_features': image}

        model = NestedCalibrationModel()
        encoder = CLIPVisionEncoder(model)

        _, scale, bias = encoder(torch.ones(2, 2))

        assert not hasattr(model, 'logit_scale')
        assert scale.item() == pytest.approx(math.exp(4.6994))
        assert bias.shape == torch.Size([])
        assert bias.item() == pytest.approx(-15.9324)

    def test_biasless_clip_exports_zero_bias(self):
        """The fixed deployment interface emits zero for missing bias."""
        class BiaslessModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.logit_scale = nn.Parameter(torch.tensor(2.0))

            def forward(self, image=None, text=None):
                del text
                return {'image_features': image}

        _, _, bias = CLIPVisionEncoder(BiaslessModel())(torch.ones(2, 2))

        assert bias.item() == pytest.approx(0.0)

    def test_combined_export_ignores_stale_tuple_calibration(self):
        """Combined export resolves calibration from the shared accessor."""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.logit_scale = nn.Parameter(torch.tensor(2.0))
                self.logit_bias = nn.Parameter(torch.tensor(-3.0))

            def forward(self, image=None, text=None):
                batch_size = text['input_ids'].shape[0]
                return (
                    image,
                    torch.ones(batch_size, image.shape[-1]),
                    torch.tensor(999.0),
                    torch.tensor(999.0),
                )

        encoder = CLIPCombinedEncoder(MockModel())
        image = torch.ones(2, 2)
        input_ids = torch.ones(2, 2, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        _, _, scale, bias = encoder(image, input_ids, attention_mask)

        assert scale.item() == pytest.approx(math.exp(2.0))
        assert bias.item() == pytest.approx(-3.0)

    def test_scalar_calibration_outputs_survive_onnx_export(self, tmp_path):
        """The deployment graph keeps scale and bias as scalar outputs."""
        import onnx

        class ExportModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.logit_scale = nn.Parameter(torch.tensor(2.0))
                self.logit_bias = nn.Parameter(torch.tensor(-3.0))

            def forward(self, image=None, text=None):
                del text
                return {'image_features': image}

        export_path = tmp_path / 'calibration.onnx'
        torch.onnx.export(
            CLIPVisionEncoder(ExportModel()),
            torch.ones(2, 2),
            export_path,
            opset_version=17,
            input_names=['image'],
            output_names=['image_embedding', 'logit_scale', 'logit_bias'],
        )

        graph = onnx.load(export_path).graph
        for output in graph.output[-2:]:
            assert len(output.type.tensor_type.shape.dim) == 0
