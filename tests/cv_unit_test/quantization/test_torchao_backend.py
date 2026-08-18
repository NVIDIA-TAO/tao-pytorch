# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TorchAO backend integration."""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import os
import tempfile
import pytest
import torch.nn as nn

from nvidia_tao_pytorch.core.quantization import (
    get_registry_manager,
    get_backend_class,
)
from nvidia_tao_pytorch.core.quantization.utils import (
    build_model_quant_config_from_omegaconf,
)


def _create_torchao_mocks():
    """Create mock objects for torchao.quantization symbols.

    Creates light-weight stand-ins for ``Float8WeightOnlyConfig``, ``Int8WeightOnlyConfig``,
    ``AOPerModuleConfig`` and ``quantize_``. The dummy ``quantize_`` returns the model passed in
    to keep the test simple and focused.
    """

    class _DummyCfg:
        def __init__(self, kind: str):
            self.kind = kind

    class _DummyAOPerModuleConfig:
        def __init__(self, module_fqn_to_config):  # noqa: D401 - emulate torchao signature
            self.module_fqn_to_config = module_fqn_to_config

    def _dummy_quantize_(model, cfg, **kwargs):  # noqa: D401 - emulate torchao signature
        # In-place no-op; return the model
        return model

    return {
        "Float8WeightOnlyConfig": lambda: _DummyCfg("fp8"),
        "Int8WeightOnlyConfig": lambda: _DummyCfg("int8"),
        "AOPerModuleConfig": _DummyAOPerModuleConfig,
        "quantize_": _dummy_quantize_,
    }


def _ensure_torchao_registered():
    """Ensure torchao backend module executes decorator to register backend.

    Uses proper module patching and manual registration instead of reloading.
    """
    # Mock the torchao imports before importing the backend
    with patch.multiple(
        "torchao.quantization",
        Float8WeightOnlyConfig=lambda: MagicMock(),
        Int8WeightOnlyConfig=lambda: MagicMock(),
        AOPerModuleConfig=MagicMock,
        quantize_=MagicMock(return_value=MagicMock()),
    ):
        from nvidia_tao_pytorch.core.quantization.backends.torchao.torchao import TorchAOBackend
        from nvidia_tao_pytorch.core.quantization.registry import register_backend

        # Manually register the backend to avoid module reloading
        register_backend("torchao")(TorchAOBackend)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 10)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.unit
def test_torchao_backend_prepare_and_quantize_int8():
    # Ensure clean registry across tests
    get_registry_manager().clear_all()

    torchao_mocks = _create_torchao_mocks()
    with patch.multiple(
        "torchao.quantization",
        **torchao_mocks
    ):
        _ensure_torchao_registered()
        backend_cls = get_backend_class("torchao")
        q = backend_cls()

        model = ToyModel()
        cfg = build_model_quant_config_from_omegaconf(
            {
                "backend": "torchao",
                "mode": "weight_only_ptq",
                "layers": [
                    {
                        "module_name": "Linear",
                        "weights": {"dtype": "int8"},
                    }
                ],
            }
        )

        prepared = q.prepare(model, cfg)
        assert prepared is model, "prepare should be a no-op for the TorchAO backend"

        quantized = q.quantize(prepared, cfg)
        assert isinstance(quantized, nn.Module), "quantize should return a torch.nn.Module"

    # Clean up registry after test
    get_registry_manager().clear_all()


@pytest.mark.unit
def test_torchao_backend_quantize_fp8_and_skip():
    # Ensure clean registry across tests
    get_registry_manager().clear_all()

    torchao_mocks = _create_torchao_mocks()
    with patch.multiple(
        "torchao.quantization",
        **torchao_mocks
    ):
        _ensure_torchao_registered()
        backend_cls = get_backend_class("torchao")
        q = backend_cls()

        model = ToyModel()
        cfg = build_model_quant_config_from_omegaconf(
            {
                "backend": "torchao",
                "mode": "weight_only_ptq",
                "layers": [
                    {
                        "module_name": "Linear",
                        "weights": {"dtype": "fp8_e4m3fn"},
                    }
                ],
                "skip_names": ["*linear*"],
            }
        )

        prepared = q.prepare(model, cfg)
        # With skip covering the only layer, quantize should still succeed
        quantized = q.quantize(prepared, cfg)
        assert isinstance(quantized, nn.Module)

    # Clean up registry after test
    get_registry_manager().clear_all()


@pytest.mark.unit
def test_torchao_backend_weights_native_disables_quantization():
    # Ensure clean registry across tests
    get_registry_manager().clear_all()

    torchao_mocks = _create_torchao_mocks()
    with patch.multiple(
        "torchao.quantization",
        **torchao_mocks
    ):
        _ensure_torchao_registered()
        backend_cls = get_backend_class("torchao")
        q = backend_cls()

        model = ToyModel()
        cfg = build_model_quant_config_from_omegaconf(
            {
                "backend": "torchao",
                "mode": "weight_only_ptq",
                "layers": [
                    {
                        "module_name": "Linear",
                        "weights": {"dtype": "native"},
                    }
                ],
            }
        )

        prepared = q.prepare(model, cfg)
        # With 'native', mapping should be empty and quantize becomes a no-op
        quantized = q.quantize(prepared, cfg)
        assert isinstance(quantized, nn.Module)

    # Clean up registry after test
    get_registry_manager().clear_all()


@pytest.mark.unit
def test_torchao_backend_save_model(tmp_path=None):
    # Ensure clean registry across tests
    get_registry_manager().clear_all()

    torchao_mocks = _create_torchao_mocks()
    with patch.multiple(
        "torchao.quantization",
        **torchao_mocks
    ):
        _ensure_torchao_registered()
        backend_cls = get_backend_class("torchao")
        q = backend_cls()

        model = ToyModel()
        cfg = build_model_quant_config_from_omegaconf(
            {
                "backend": "torchao",
                "mode": "weight_only_ptq",
                "layers": [
                    {
                        "module_name": "Linear",
                        "weights": {"dtype": "int8"},
                    }
                ],
            }
        )

        prepared = q.prepare(model, cfg)
        quantized = q.quantize(prepared, cfg)

        # Use tmp directory from pytest or create a temp dir
        save_dir = tmp_path if tmp_path is not None else tempfile.mkdtemp()
        q.save_model(quantized, str(save_dir))
        expected_path = os.path.join(str(save_dir), "quantized_model_torchao.pth")
        assert os.path.exists(expected_path), "Expected saved file was not created"

    # Clean up registry after test
    get_registry_manager().clear_all()
