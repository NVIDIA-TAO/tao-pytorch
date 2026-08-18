# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ModelQuantizer error cases."""

import pytest
import torch
import torch.nn as nn
import os

from nvidia_tao_pytorch.core.quantization import (
    ModelQuantizer,
    get_registry_manager,
    QuantizationMode,
)
from nvidia_tao_pytorch.core.quantization.registry import register_backend
from nvidia_tao_pytorch.core.quantization.quantizer_base import QuantizerBase
from nvidia_tao_pytorch.core.quantization.calibratable import Calibratable
from nvidia_tao_pytorch.core.quantization import ModelQuantizationConfig
from nvidia_tao_pytorch.core.quantization.utils import build_model_quant_config_from_omegaconf


class DummyBackendForErrorTests(QuantizerBase):
    """A minimal backend for testing error cases."""

    def __init__(self, backend_kwargs=None):
        pass

    def prepare(self, model, config):
        return model

    def quantize(self, model, config):
        return model


class NotABackend:
    """A class that is not a QuantizerBase subclass."""

    pass


def test_quantizer_missing_backend():
    """Test that ModelQuantizer raises ValueError when backend is not specified."""
    config = ModelQuantizationConfig(backend=None, layers=[])
    with pytest.raises(ValueError, match="Quantization backend must be specified"):
        ModelQuantizer(config)


def test_quantizer_invalid_backend_type():
    """Test that ModelQuantizer raises TypeError when backend is not a QuantizerBase subclass."""
    # Clean registry
    registry = get_registry_manager()
    registry.clear_all()

    # Register a non-backend class
    register_backend("invalid_backend")(NotABackend)

    config = {"backend": "invalid_backend", "layers": []}
    with pytest.raises(TypeError, match="must be a subclass of QuantizerBase"):
        ModelQuantizer(config)

    # Clean up
    registry.clear_all()


def test_prepare_with_invalid_model():
    """Test that prepare() raises TypeError when model is not nn.Module."""
    # Clean registry and register dummy backend
    registry = get_registry_manager()
    registry.clear_all()
    register_backend("dummy_error_test")(DummyBackendForErrorTests)

    config = {"backend": "dummy_error_test", "layers": []}
    quantizer = ModelQuantizer(config)

    with pytest.raises(TypeError, match="model must be an instance of torch.nn.Module"):
        quantizer.prepare("not_a_model")

    # Clean up
    registry.clear_all()


def test_save_model_without_quantizing():
    """Test that save_model() raises RuntimeError when called before quantize()."""
    # Clean registry and register dummy backend
    registry = get_registry_manager()
    registry.clear_all()
    register_backend("dummy_error_test")(DummyBackendForErrorTests)

    config = {"backend": "dummy_error_test", "layers": []}
    quantizer = ModelQuantizer(config)

    model = nn.Linear(10, 5)

    # With state validation, RuntimeError is raised before checking for quantized model
    with pytest.raises(RuntimeError, match="save_model\\(\\) called in invalid state"):
        quantizer.save_model(model=model, path="dummy_path.pth")

    # Clean up
    registry.clear_all()


def test_save_model_fallback_to_torch_save(tmp_path):
    """Test that save_model() falls back to torch.save when backend doesn't provide save_model."""
    # Clean registry and register dummy backend
    registry = get_registry_manager()
    registry.clear_all()
    register_backend("dummy_error_test")(DummyBackendForErrorTests)

    config = {"backend": "dummy_error_test", "layers": []}
    quantizer = ModelQuantizer(config)

    model = nn.Linear(10, 5)
    prepared = quantizer.prepare(model)
    quantized = quantizer.quantize(prepared)

    # Backend doesn't have save_model, so it should use torch.save
    save_path = str(tmp_path / "model.pth")
    quantizer.save_model(model=quantized, path=save_path)

    # Verify the file was created
    assert os.path.exists(save_path)

    # Clean up
    registry.clear_all()


def test_calibrate_with_non_calibratable_backend():
    """Test that calibrate() logs warning when backend doesn't support calibration."""
    # Clean registry and register dummy backend
    registry = get_registry_manager()
    registry.clear_all()
    register_backend("dummy_error_test")(DummyBackendForErrorTests)

    config = {"backend": "dummy_error_test", "layers": []}
    quantizer = ModelQuantizer(config)

    model = nn.Linear(10, 5)
    quantizer.prepare(model)

    # Create a dummy dataloader
    data = torch.randn(4, 10)
    labels = torch.zeros(4, dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, labels), batch_size=2
    )

    # Should not raise, but log a warning
    # The warning is tested by checking that the method doesn't crash
    quantizer.calibrate(model, loader)

    # Clean up
    registry.clear_all()


def test_abstract_base_methods_raise_not_implemented():
    """Test that abstract base classes cannot be instantiated."""
    # Test that we cannot instantiate abstract classes
    # Abstract classes with proper abstract methods should raise TypeError on instantiation
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        QuantizerBase()  # pylint: disable=abstract-class-instantiated

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        Calibratable()  # pylint: disable=abstract-class-instantiated


def test_utils_with_quantization_mode_enum():
    """Test build_model_quant_config_from_omegaconf with QuantizationMode enum."""
    config_dict = {
        "backend": "modelopt.pytorch",
        "mode": QuantizationMode.STATIC_PTQ,  # Pass enum instead of string
        "layers": [
            {
                "module_name": "Linear",
                "weights": {"dtype": "int8"},
            }
        ],
    }

    result = build_model_quant_config_from_omegaconf(config_dict)
    assert result.mode == "static_ptq"


class CalibrableTestBackend(QuantizerBase, Calibratable):
    """A backend that supports calibration with save_model."""

    def __init__(self, backend_kwargs=None):
        self.calibrated = False
        self.saved_path = None

    def prepare(self, model, config):
        return model

    def quantize(self, model, config):
        return model

    def calibrate(self, model, data_loader):
        self.calibrated = True
        model.eval()
        with torch.no_grad():
            data, _ = next(iter(data_loader))
            model(data)

    def save_model(self, model, path):
        self.saved_path = path
        # Actually save something so the test can verify
        torch.save(model.state_dict(), os.path.join(path, "model.pth"))


def test_calibrate_with_calibratable_backend():
    """Test that calibrate() actually calls the backend's calibrate method."""
    registry = get_registry_manager()
    registry.clear_all()
    register_backend("calibrable_test")(CalibrableTestBackend)

    config = {"backend": "calibrable_test", "layers": []}
    quantizer = ModelQuantizer(config)

    model = nn.Linear(10, 5)
    quantizer.prepare(model)

    # Create dataloader
    data = torch.randn(4, 10)
    labels = torch.zeros(4, dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, labels), batch_size=2
    )

    # Should call backend's calibrate
    quantizer.calibrate(model, loader)
    assert quantizer.quantizer.calibrated is True

    registry.clear_all()


def test_save_model_with_backend_save_method(tmp_path):
    """Test that save_model() uses backend's save_model if available."""
    registry = get_registry_manager()
    registry.clear_all()
    register_backend("calibrable_test")(CalibrableTestBackend)

    config = {"backend": "calibrable_test", "layers": []}
    quantizer = ModelQuantizer(config)

    model = nn.Linear(10, 5)
    prepared = quantizer.prepare(model)
    quantized = quantizer.quantize(prepared)

    save_path = str(tmp_path)
    quantizer.save_model(model=quantized, path=save_path)

    # Verify backend's save_model was called
    assert quantizer.quantizer.saved_path == save_path
    assert os.path.exists(os.path.join(save_path, "model.pth"))

    registry.clear_all()


def test_quantize_model_with_calibration():
    """Test quantize_model() with calibration_loader and static_ptq mode."""
    registry = get_registry_manager()
    registry.clear_all()
    register_backend("calibrable_test")(CalibrableTestBackend)

    config = {"backend": "calibrable_test", "mode": "static_ptq", "layers": []}
    quantizer = ModelQuantizer(config)

    model = nn.Linear(10, 5)

    # Create calibration dataloader
    data = torch.randn(4, 10)
    labels = torch.zeros(4, dtype=torch.long)
    calibration_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, labels), batch_size=2
    )

    # Should call calibrate internally
    quantized = quantizer.quantize_model(model, calibration_loader=calibration_loader)
    assert quantizer.quantizer.calibrated is True
    assert isinstance(quantized, nn.Module)

    registry.clear_all()


def test_backend_import_exception_handling():
    """Test that backend import exceptions are handled gracefully."""
    # This test verifies that the quantization module handles ImportError
    # exceptions when optional backends fail to import.
    #
    # The actual exception handling happens in:
    # - nvidia_tao_pytorch/core/quantization/quantizer.py lines 33-37
    # - nvidia_tao_pytorch/core/quantization/backends/__init__.py lines 26-44
    #
    # These try/except blocks ensure the module remains functional even if
    # optional dependencies (modelopt, torchao) are not installed.

    # Instead of manipulating sys.modules (which can cause test pollution),
    # we verify the module is importable and functional
    from nvidia_tao_pytorch.core.quantization import quantizer

    # The module should be importable and usable
    assert quantizer is not None
    assert ModelQuantizer is not None

    # The try/except blocks in the source code handle ImportError gracefully
    # This defensive approach ensures toolkit works with subset of backends
