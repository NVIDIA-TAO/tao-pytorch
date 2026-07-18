# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RADIO utilities (mock-based, no model downloads)."""

import pytest
import torch
import torch.nn as nn


@pytest.mark.multimodal_unit
class TestCRADIOInterface:
    """Test CRADIO interface without real model loading."""

    def test_adapter_inherits_from_base(self):
        """Test that CRADIO inherits from BaseCLIPAdapter."""
        from nvidia_tao_pytorch.multimodal.clip.model.adapters import CRADIO
        from nvidia_tao_pytorch.multimodal.clip.model.adapters import BaseCLIPAdapter

        assert issubclass(CRADIO, BaseCLIPAdapter)

    def test_adapter_has_required_methods(self):
        """Test that CRADIO has required abstract method implementations."""
        from nvidia_tao_pytorch.multimodal.clip.model.adapters import CRADIO

        # Check that abstract methods are implemented (not raising NotImplementedError)
        assert hasattr(CRADIO, 'encode_image')
        assert hasattr(CRADIO, 'encode_text')
        assert hasattr(CRADIO, 'forward')
        assert hasattr(CRADIO, 'set_grad_checkpointing')

    def test_adapter_has_tokenizer_utilities(self):
        """Test that the adapter uses shared tokenizer utilities."""
        from nvidia_tao_pytorch.multimodal.clip.model.adapters.radio import CRADIO
        import inspect

        source = inspect.getsource(CRADIO)
        assert 'CLIPCompatibleTokenizer' in source
        assert 'OpenCLIPWrappedTokenizer' in source


@pytest.mark.multimodal_unit
class TestBuildRadioModelInterface:
    """Test build_radio_model function interface."""

    def test_build_function_exists(self):
        """Test that build_radio_model function exists and is callable."""
        from nvidia_tao_pytorch.multimodal.clip.model.builders import build_radio_model

        assert callable(build_radio_model)

    def test_build_function_signature(self):
        """Test that build_radio_model has expected parameters."""
        from nvidia_tao_pytorch.multimodal.clip.model.builders import build_radio_model
        import inspect

        sig = inspect.signature(build_radio_model)
        params = list(sig.parameters.keys())

        assert 'model_version' in params
        assert 'aug_cfg' in params

    def test_default_model_version(self):
        """Test that build_radio_model has correct default model version."""
        from nvidia_tao_pytorch.multimodal.clip.model.builders import build_radio_model
        import inspect

        sig = inspect.signature(build_radio_model)
        default = sig.parameters['model_version'].default

        assert default == 'c-radio_v3-l'


@pytest.mark.multimodal_unit
class TestModuleExports:
    """Test that module exports are correct."""

    def test_adapters_exports(self):
        """Test that adapters package exports expected symbols."""
        from nvidia_tao_pytorch.multimodal.clip.model import adapters

        assert hasattr(adapters, 'BaseCLIPAdapter')
        assert hasattr(adapters, 'CRADIO')
        assert hasattr(adapters, 'SigLIP2')

    def test_package_init_exports(self):
        """Test that package __init__ exports all utilities."""
        from nvidia_tao_pytorch.multimodal.clip.model import (
            CRADIO,
            SigLIP2,
            build_radio_model,
            build_siglip2_model,
        )

        assert CRADIO is not None
        assert SigLIP2 is not None
        assert build_radio_model is not None
        assert build_siglip2_model is not None

    def test_package_exports_base_adapter(self):
        """Test that package exports BaseCLIPAdapter."""
        from nvidia_tao_pytorch.multimodal.clip.model import BaseCLIPAdapter

        assert BaseCLIPAdapter is not None

    def test_package_exports_tokenizer_utils(self):
        """Test that package exports tokenizer utilities."""
        from nvidia_tao_pytorch.multimodal.clip.model import (
            canonicalize_text,
            SigLIP2WrappedTokenizer,
            CLIPCompatibleTokenizer,
        )

        assert canonicalize_text is not None
        assert SigLIP2WrappedTokenizer is not None
        assert CLIPCompatibleTokenizer is not None

    def test_package_exports_transforms(self):
        """Test that package exports transform utilities."""
        from nvidia_tao_pytorch.multimodal.clip.model import SigLIP2ImageTransform

        assert SigLIP2ImageTransform is not None
