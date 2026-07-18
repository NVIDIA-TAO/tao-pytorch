# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ModelOpt ONNX backend."""

from __future__ import annotations

import tempfile
import os
from unittest.mock import patch
import pytest
import torch
import numpy as np

from nvidia_tao_pytorch.core.quantization import (
    get_registry_manager,
    get_backend_class,
    register_backend,
)
from nvidia_tao_pytorch.core.quantization.utils import (
    build_model_quant_config_from_omegaconf,
)


@pytest.fixture
def clean_registry():
    """Ensure clean registry for each test."""
    get_registry_manager().clear_all()
    yield
    get_registry_manager().clear_all()


@pytest.fixture
def mock_onnx_file():
    """Create a temporary ONNX file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
        tmp_file.write(b"dummy onnx content")
        onnx_path = tmp_file.name

    yield onnx_path

    if os.path.exists(onnx_path):
        os.unlink(onnx_path)


@pytest.fixture
def backend_class(clean_registry):
    """Register and return the backend class."""
    from nvidia_tao_pytorch.core.quantization.backends.modelopt_onnx.modelopt_onnx import (
        ModelOptONNXBackend,
    )
    register_backend("modelopt.onnx")(ModelOptONNXBackend)
    return get_backend_class("modelopt.onnx")


@pytest.fixture
def quant_config(mock_onnx_file):
    """Create a standard quantization config with ONNX path."""
    return build_model_quant_config_from_omegaconf({
        "backend": "modelopt.onnx",
        "model_path": mock_onnx_file,
        "mode": "static_ptq",
        "algorithm": "max",
        "layers": [{
            "module_name": "Linear",
            "weights": {"dtype": "int8"},
            "activations": {"dtype": "int8"},
        }],
    })


@pytest.fixture
def mock_dataloader():
    """Create a simple mock dataloader."""
    data = torch.randn(4, 16)
    labels = torch.zeros(4, dtype=torch.long)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, labels),
        batch_size=2
    )


class TestModelOptONNXBackend:
    """Unit tests for ModelOptONNXBackend."""

    # === Prepare Tests ===

    def test_prepare_success(self, backend_class, quant_config):
        """Test successful prepare operation."""
        backend = backend_class()
        result = backend.prepare(None, quant_config)
        assert result is None
        assert backend._onnx_path is not None

    def test_prepare_with_model_raises_error(self, backend_class, quant_config):
        """Test that passing model to prepare raises ValueError."""
        backend = backend_class()
        import torch.nn as nn
        model = nn.Linear(10, 5)
        with pytest.raises(ValueError, match="ONNX backend requires model=None"):
            backend.prepare(model, quant_config)

    def test_prepare_missing_onnx_path_in_config(self, backend_class):
        """Test prepare with missing onnx_path in config."""
        backend = backend_class()
        config_without_onnx = build_model_quant_config_from_omegaconf({
            "backend": "modelopt.onnx",
            "mode": "static_ptq",
            "algorithm": "max",
            "layers": [{"module_name": "Linear", "weights": {"dtype": "int8"}}],
        })

        with pytest.raises(ValueError, match="ONNX file path must be specified in config.model_path"):
            backend.prepare(None, config_without_onnx)

    def test_prepare_with_nonexistent_onnx_file(self, backend_class):
        """Test prepare with non-existent ONNX file in config."""
        backend = backend_class()
        config_with_bad_path = build_model_quant_config_from_omegaconf({
            "backend": "modelopt.onnx",
            "model_path": "/nonexistent/file.onnx",
            "mode": "static_ptq",
            "algorithm": "max",
            "layers": [{"module_name": "Linear", "weights": {"dtype": "int8"}}],
        })

        with pytest.raises(FileNotFoundError, match="ONNX model file not found"):
            backend.prepare(None, config_with_bad_path)

    def test_prepare_invalid_config_type(self, backend_class):
        """Test prepare with invalid config type."""
        backend = backend_class()
        with pytest.raises(TypeError, match="config must be an instance of ModelQuantizationConfig"):
            backend.prepare(None, "invalid_config")

    def test_prepare_invalid_extension(self, backend_class):
        """Test prepare with invalid file extension."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(b"dummy content")
            invalid_path = tmp_file.name

        try:
            backend = backend_class()
            config_with_bad_extension = build_model_quant_config_from_omegaconf({
                "backend": "modelopt.onnx",
                "model_path": invalid_path,
                "mode": "static_ptq",
                "algorithm": "max",
                "layers": [{"module_name": "Linear", "weights": {"dtype": "int8"}}],
            })

            with pytest.raises(ValueError, match="Invalid file extension"):
                backend.prepare(None, config_with_bad_extension)
        finally:
            os.unlink(invalid_path)

    def test_prepare_unsupported_mode(self, backend_class):
        """Test prepare with unsupported quantization mode."""
        backend = backend_class()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            unsupported_config = build_model_quant_config_from_omegaconf({
                "backend": "modelopt.onnx",
                "model_path": tmp_path,
                "mode": "weight_only_ptq",  # Unsupported
                "algorithm": "max",
                "layers": [{"module_name": "Linear", "weights": {"dtype": "int8"}}],
            })

            with pytest.raises(ValueError, match="Unsupported mode"):
                backend.prepare(None, unsupported_config)
        finally:
            os.unlink(tmp_path)

    # === Calibrate Tests ===

    def test_set_calibration_data(self, backend_class):
        """Test setting calibration data directly."""
        backend = backend_class()
        calibration_data = np.random.randn(2, 3, 32, 32).astype(np.float32)
        backend.set_calibration_data(calibration_data)
        assert backend._calibration_data is not None
        assert backend._calibration_data.shape == (2, 3, 32, 32)

    def test_calibrate_with_none_dataloader(self, backend_class):
        """Test calibrate with None dataloader."""
        backend = backend_class()
        backend.calibrate(None, None)
        assert backend._calibration_data is None

    # === Quantize Tests ===

    @patch('nvidia_tao_pytorch.core.quantization.backends.modelopt_onnx.modelopt_onnx.modelopt_onnx_quantize')
    @patch('os.path.getsize')
    @patch('os.path.exists')
    def test_quantize_warns_when_calibration_data_missing(self, mock_exists, mock_getsize, mock_quantize, backend_class, quant_config, caplog):
        """Test that quantize() logs warning when calibration data is missing."""
        backend = backend_class()
        backend.prepare(None, quant_config)

        # Don't set calibration data - should warn but not error
        mock_exists.return_value = True
        mock_getsize.return_value = 1024 * 1024

        result = backend.quantize(None, quant_config)
        assert result is not None
        assert isinstance(result, str)

        # Check that warning was logged
        assert "No calibration data provided" in caplog.text
        assert "ModelOpt will generate dummy data" in caplog.text

    @patch('nvidia_tao_pytorch.core.quantization.backends.modelopt_onnx.modelopt_onnx.modelopt_onnx_quantize')
    @patch('os.path.getsize')
    @patch('os.makedirs')
    @patch('os.path.exists')
    def test_quantize_creates_output_directory(self, mock_exists, mock_makedirs, mock_getsize, mock_quantize, backend_class, mock_onnx_file):
        """Test that quantize() creates output directory if it doesn't exist."""
        # Create config with a non-existent results_dir
        config = build_model_quant_config_from_omegaconf({
            "backend": "modelopt.onnx",
            "model_path": mock_onnx_file,
            "mode": "static_ptq",
            "algorithm": "max",
            "layers": [{
                "module_name": "Linear",
                "weights": {"dtype": "int8"},
                "activations": {"dtype": "int8"},
            }],
            "results_dir": "/tmp/nonexistent_dir/quantized",
        })

        backend = backend_class()
        backend.prepare(None, config)
        backend.set_calibration_data(np.random.randn(10, 3, 32, 32).astype(np.float32))

        # Mock file existence checks
        def exists_side_effect(path):
            # Input ONNX file exists
            if path == mock_onnx_file:
                return True
            # Output directory doesn't exist
            if path == "/tmp/nonexistent_dir/quantized":
                return False
            # Output file exists after quantization
            if path.endswith("quantized_model.onnx"):
                return True
            return False

        mock_exists.side_effect = exists_side_effect
        mock_getsize.return_value = 1024 * 1024  # 1 MB

        result = backend.quantize(None, config)
        assert result is not None
        assert isinstance(result, str)
        # Verify makedirs was called for the output directory
        mock_makedirs.assert_called_once_with("/tmp/nonexistent_dir/quantized", exist_ok=True)

    @patch('nvidia_tao_pytorch.core.quantization.backends.modelopt_onnx.modelopt_onnx.modelopt_onnx_quantize')
    @patch('os.path.exists')
    def test_quantize_verifies_output_file_exists(self, mock_exists, mock_quantize, backend_class, quant_config, mock_dataloader):
        """Test that quantize() verifies output file was created."""
        backend = backend_class()
        backend.prepare(None, quant_config)
        backend.calibrate(None, mock_dataloader)

        # Mock that output file doesn't exist after quantization
        def exists_side_effect(path):
            if path == backend._onnx_path:
                return True
            # Output path doesn't exist (simulating ModelOpt failure)
            return False
        mock_exists.side_effect = exists_side_effect

        with pytest.raises(FileNotFoundError, match="Quantized model was not created at expected path"):
            backend.quantize(None, quant_config)

    @patch('nvidia_tao_pytorch.core.quantization.backends.modelopt_onnx.modelopt_onnx.modelopt_onnx_quantize')
    def test_quantize_wraps_modelopt_errors(self, mock_quantize, backend_class, quant_config, mock_dataloader):
        """Test that quantize() wraps ModelOpt errors with context."""
        backend = backend_class()
        backend.prepare(None, quant_config)
        backend.calibrate(None, mock_dataloader)

        # Make ModelOpt raise an exception
        mock_quantize.side_effect = RuntimeError("ModelOpt internal error")

        with pytest.raises(RuntimeError, match="ModelOpt ONNX quantization failed"):
            backend.quantize(None, quant_config)

    def test_quantize_with_model_raises_error(self, backend_class, quant_config, mock_dataloader):
        """Test that passing model to quantize raises ValueError."""
        backend = backend_class()
        backend.prepare(None, quant_config)
        backend.calibrate(None, mock_dataloader)

        import torch.nn as nn
        model = nn.Linear(10, 5)
        with pytest.raises(ValueError, match="ONNX backend requires model=None"):
            backend.quantize(model, quant_config)

    # === Save Model Tests ===

    def test_save_model_with_string_path(self, backend_class):
        """Test that save_model() handles string path (from quantize return) gracefully."""
        backend = backend_class()
        # Should not raise error, just log
        backend.save_model("/path/to/quantized_model.onnx", "/some/path")

    def test_save_model_with_none(self, backend_class):
        """Test that save_model() handles None gracefully."""
        backend = backend_class()
        # Should not raise error, just log
        backend.save_model(None, "/some/path")

    def test_save_model_with_pytorch_model_raises_error(self, backend_class):
        """Test that save_model() raises error for PyTorch models."""
        backend = backend_class()
        import torch.nn as nn
        model = nn.Linear(10, 5)
        with pytest.raises(ValueError, match="ONNX backend does not support saving PyTorch models"):
            backend.save_model(model, "/some/path")

    # === Full Workflow Test ===

    @patch('nvidia_tao_pytorch.core.quantization.backends.modelopt_onnx.modelopt_onnx.modelopt_onnx_quantize')
    @patch('os.path.getsize')
    @patch('os.path.exists')
    def test_full_workflow(self, mock_exists, mock_getsize, mock_quantize, backend_class, quant_config, mock_dataloader):
        """Test complete workflow: prepare -> calibrate -> quantize -> save."""
        backend = backend_class()

        # Prepare
        result = backend.prepare(None, quant_config)
        assert result is None
        assert backend._onnx_path is not None

        # Calibrate
        backend.calibrate(None, mock_dataloader)
        assert backend._calibration_data is not None
        assert isinstance(backend._calibration_data, np.ndarray)
        assert backend._calibration_data.shape[0] == 4  # Total samples

        # Mock file operations for quantize
        def exists_side_effect(path):
            return True  # All paths exist
        mock_exists.side_effect = exists_side_effect
        mock_getsize.return_value = 1024 * 1024  # 1 MB

        # Quantize
        quantized_model = backend.quantize(None, quant_config)
        assert quantized_model is not None
        assert isinstance(quantized_model, str)
        assert quantized_model.endswith(".onnx")
        mock_quantize.assert_called_once()

        # Verify parameters passed to ModelOpt
        call_args = mock_quantize.call_args
        params = call_args.kwargs
        assert "onnx_path" in params
        assert "calibration_data" in params
        assert "quantize_mode" in params
        assert params["quantize_mode"] == "int8"

        # Save (should handle the string path gracefully)
        backend.save_model(quantized_model, quant_config.results_dir)  # Should not raise

    # === Device Parameter Tests ===

    def test_execution_providers_cpu(self):
        """Test that CPU device maps to CPUExecutionProvider."""
        from nvidia_tao_pytorch.core.quantization.backends.modelopt_onnx.utils import _determine_execution_providers
        from nvidia_tao_pytorch.core.tlt_logging import logger

        providers = _determine_execution_providers("cpu", logger)
        assert "CPUExecutionProvider" in providers

    def test_execution_providers_cuda(self):
        """Test that CUDA device maps to CUDA/CPU providers."""
        from nvidia_tao_pytorch.core.quantization.backends.modelopt_onnx.utils import _determine_execution_providers
        from nvidia_tao_pytorch.core.tlt_logging import logger

        providers = _determine_execution_providers("cuda", logger)
        # Always includes CPU as fallback
        assert "CPUExecutionProvider" in providers

    def test_execution_providers_trt(self):
        """Test that TRT device attempts TensorRT provider."""
        from nvidia_tao_pytorch.core.quantization.backends.modelopt_onnx.utils import _determine_execution_providers
        from nvidia_tao_pytorch.core.tlt_logging import logger

        providers = _determine_execution_providers("trt", logger)
        # Always includes CPU as fallback
        assert "CPUExecutionProvider" in providers

    @patch('nvidia_tao_pytorch.core.quantization.backends.modelopt_onnx.modelopt_onnx.modelopt_onnx_quantize')
    @patch('os.path.getsize')
    @patch('os.path.exists')
    def test_device_parameter_used_in_quantization(self, mock_exists, mock_getsize, mock_quantize, backend_class, mock_onnx_file, mock_dataloader):
        """Test that device parameter is properly passed to execution providers."""
        config = build_model_quant_config_from_omegaconf({
            "backend": "modelopt.onnx",
            "model_path": mock_onnx_file,
            "mode": "static_ptq",
            "algorithm": "max",
            "device": "cuda",  # Explicitly set device
            "layers": [{
                "module_name": "Linear",
                "weights": {"dtype": "int8"},
                "activations": {"dtype": "int8"},
            }],
        })

        backend = backend_class()
        backend.prepare(None, config)
        backend.calibrate(None, mock_dataloader)

        # Mock file operations
        mock_exists.return_value = True
        mock_getsize.return_value = 1024 * 1024

        # Quantize
        result = backend.quantize(None, config)
        assert result is not None

        # Verify that execution_providers was passed
        call_args = mock_quantize.call_args
        params = call_args.kwargs
        assert "execution_providers" in params
        assert isinstance(params["execution_providers"], list)
        # Should always have CPU as fallback
        assert "CPUExecutionProvider" in params["execution_providers"]
