# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ModelOpt backend integration."""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest
import torch
import torch.nn as nn

from nvidia_tao_pytorch.core.quantization import (  # noqa: E402
    get_registry_manager,
    get_backend_class,
)
from nvidia_tao_pytorch.core.quantization.registry import (  # noqa: E402
    register_backend,
)
from nvidia_tao_pytorch.core.quantization.utils import (  # noqa: E402
    build_model_quant_config_from_omegaconf,
)
from nvidia_tao_pytorch.core.quantization.backends.modelopt_pytorch.utils import (  # noqa: E402
    convert_tao_to_modelopt_config,
)


class _DummyMtqModule:
    def __init__(self):
        self._last = None

    def quantize(self, model, cfg, forward_loop):
        self._last = (model, cfg)
        # Emulate quantizer insertion by returning the same model
        return model


def _create_modelopt_mocks():
    """Create mock objects for modelopt modules."""
    dummy = _DummyMtqModule()
    return {
        "mtq": MagicMock(quantize=dummy.quantize),
        "mto": MagicMock(save=MagicMock()),
    }


def _patch_modelopt_imports():
    """Targeted patch for modelopt imports - FAST"""
    modelopt_mocks = _create_modelopt_mocks()
    return patch.multiple(
        "nvidia_tao_pytorch.core.quantization.backends.modelopt_pytorch.modelopt_pytorch",
        **modelopt_mocks
    )


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 10)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.unit
def test_modelopt_backend_prepare():
    # Ensure clean registry across tests
    get_registry_manager().clear_all()
    from nvidia_tao_pytorch.core.quantization.backends.modelopt_pytorch.modelopt_pytorch import (
        ModelOptBackend,
    )

    register_backend("modelopt.pytorch")(ModelOptBackend)

    backend_cls = get_backend_class("modelopt.pytorch")
    q = backend_cls()

    model = ToyModel()
    cfg = build_model_quant_config_from_omegaconf(
        {
            "backend": "modelopt",
            "mode": "static_ptq",
            "algorithm": "max",
            "layers": [
                {
                    "module_name": "Linear",
                    "weights": {"dtype": "int8"},
                    "activations": {"dtype": "int8"},
                }
            ],
        }
    )

    prepared = q.prepare(model, cfg)
    assert prepared is model, "prepare should be a no-op for the ModelOpt backend"


@pytest.mark.unit
def test_modelopt_backend_calibrate():
    # Ensure clean registry across tests
    get_registry_manager().clear_all()
    from nvidia_tao_pytorch.core.quantization.backends.modelopt_pytorch.modelopt_pytorch import (
        ModelOptBackend,
    )

    register_backend("modelopt.pytorch")(ModelOptBackend)

    backend_cls = get_backend_class("modelopt.pytorch")
    q = backend_cls()

    model = ToyModel()
    cfg = build_model_quant_config_from_omegaconf(
        {
            "backend": "modelopt",
            "mode": "static_ptq",
            "algorithm": "max",
            "layers": [
                {
                    "module_name": "Linear",
                    "weights": {"dtype": "int8"},
                    "activations": {"dtype": "int8"},
                }
            ],
        }
    )

    prepared = q.prepare(model, cfg)

    # Build a small calibration loader
    data = torch.randn(4, 16)
    labels = torch.zeros(4, dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, labels), batch_size=2
    )

    q.calibrate(prepared, loader)
    assert (
        q._forward_loop is not None
    ), "calibrate should set an internal forward loop for later use"


@pytest.mark.unit
def test_modelopt_backend_calibrate_then_quantize():
    # Ensure clean registry across tests
    get_registry_manager().clear_all()
    from nvidia_tao_pytorch.core.quantization.backends.modelopt_pytorch.modelopt_pytorch import (
        ModelOptBackend,
    )

    register_backend("modelopt.pytorch")(ModelOptBackend)

    backend_cls = get_backend_class("modelopt.pytorch")
    q = backend_cls()

    model = ToyModel()
    cfg = build_model_quant_config_from_omegaconf(
        {
            "backend": "modelopt",
            "mode": "static_ptq",
            "algorithm": "max",
            "layers": [
                {
                    "module_name": "Linear",
                    "weights": {"dtype": "int8"},
                    "activations": {"dtype": "int8"},
                }
            ],
        }
    )

    prepared = q.prepare(model, cfg)

    # Build a small calibration loader
    data = torch.randn(4, 16)
    labels = torch.zeros(4, dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, labels), batch_size=2
    )

    q.calibrate(prepared, loader)

    with _patch_modelopt_imports():
        out = q.quantize(prepared, cfg)
        assert isinstance(out, nn.Module), "quantize should return a torch.nn.Module"


@pytest.mark.unit
def test_convert_tao_to_modelopt_config_emits_expected_shape():
    model = ToyModel()
    cfg = build_model_quant_config_from_omegaconf(
        {
            "backend": "modelopt",
            "mode": "static_ptq",
            "algorithm": "max",
            "layers": [
                {
                    "module_name": "Linear",
                    "weights": {"dtype": "int8"},
                    "activations": {"dtype": "int8"},
                }
            ],
        }
    )
    mo_cfg = convert_tao_to_modelopt_config(cfg, model)
    assert "quant_cfg" in mo_cfg, "converter should emit a 'quant_cfg' section"
    assert "algorithm" in mo_cfg, "converter should emit an 'algorithm' key"
    # Either class rules or expanded module rules must exist
    has_linear = any(
        k == "nn.Linear" or k.endswith("linear.weight_quantizer")
        for k in mo_cfg["quant_cfg"].keys()
    )
    assert (
        has_linear
    ), "Expected rules targeting Linear layers (class key or expanded module rules)"


@pytest.mark.unit
def test_modelopt_backend_device_parameter_cpu():
    """Test that CPU device is properly used in ModelOpt PyTorch backend."""
    get_registry_manager().clear_all()
    from nvidia_tao_pytorch.core.quantization.backends.modelopt_pytorch.modelopt_pytorch import ModelOptBackend

    register_backend("modelopt.pytorch")(ModelOptBackend)

    backend_cls = get_backend_class("modelopt.pytorch")
    q = backend_cls()

    model = ToyModel()
    cfg = build_model_quant_config_from_omegaconf(
        {
            "backend": "modelopt.pytorch",
            "mode": "static_ptq",
            "algorithm": "max",
            "device": "cpu",  # Explicitly set CPU
            "layers": [
                {
                    "module_name": "Linear",
                    "weights": {"dtype": "int8"},
                    "activations": {"dtype": "int8"},
                }
            ],
        }
    )

    # Prepare and store config
    prepared = q.prepare(model, cfg)

    # Verify config is stored
    assert q._config is not None
    assert q._config.device == "cpu"

    # Build a small calibration loader
    data = torch.randn(4, 16)
    labels = torch.zeros(4, dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, labels), batch_size=2
    )

    # Calibrate
    q.calibrate(prepared, loader)

    # Verify forward loop was created
    assert q._forward_loop is not None


@pytest.mark.unit
def test_modelopt_backend_device_parameter_cuda():
    """Test that CUDA device is properly handled in ModelOpt PyTorch backend."""
    get_registry_manager().clear_all()
    from nvidia_tao_pytorch.core.quantization.backends.modelopt_pytorch.modelopt_pytorch import ModelOptBackend

    register_backend("modelopt.pytorch")(ModelOptBackend)

    backend_cls = get_backend_class("modelopt.pytorch")
    q = backend_cls()

    model = ToyModel()
    cfg = build_model_quant_config_from_omegaconf(
        {
            "backend": "modelopt.pytorch",
            "mode": "static_ptq",
            "algorithm": "max",
            "device": "cuda",  # Request CUDA (will fall back to CPU if not available)
            "layers": [
                {
                    "module_name": "Linear",
                    "weights": {"dtype": "int8"},
                    "activations": {"dtype": "int8"},
                }
            ],
        }
    )

    # Prepare and store config
    prepared = q.prepare(model, cfg)

    # Verify config is stored
    assert q._config is not None
    assert q._config.device == "cuda"

    # Build a small calibration loader
    data = torch.randn(4, 16)
    labels = torch.zeros(4, dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, labels), batch_size=2
    )

    # Calibrate (should handle device gracefully - falling back to CPU if CUDA unavailable)
    q.calibrate(prepared, loader)

    # Verify forward loop was created
    assert q._forward_loop is not None


@pytest.mark.unit
def test_modelopt_backend_device_parameter_trt_fallback():
    """Test that TRT device falls back to CUDA in ModelOpt PyTorch backend."""
    get_registry_manager().clear_all()
    from nvidia_tao_pytorch.core.quantization.backends.modelopt_pytorch.modelopt_pytorch import ModelOptBackend

    register_backend("modelopt.pytorch")(ModelOptBackend)

    backend_cls = get_backend_class("modelopt.pytorch")
    q = backend_cls()

    model = ToyModel()
    cfg = build_model_quant_config_from_omegaconf(
        {
            "backend": "modelopt.pytorch",
            "mode": "static_ptq",
            "algorithm": "max",
            "device": "trt",  # Should fall back to CUDA
            "layers": [
                {
                    "module_name": "Linear",
                    "weights": {"dtype": "int8"},
                    "activations": {"dtype": "int8"},
                }
            ],
        }
    )

    # Prepare and store config
    prepared = q.prepare(model, cfg)

    # Verify config is stored
    assert q._config is not None
    assert q._config.device == "trt"

    # Build a small calibration loader
    data = torch.randn(4, 16)
    labels = torch.zeros(4, dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, labels), batch_size=2
    )

    # Calibrate (should handle TRT -> CUDA fallback)
    q.calibrate(prepared, loader)

    # Verify forward loop was created
    assert q._forward_loop is not None
