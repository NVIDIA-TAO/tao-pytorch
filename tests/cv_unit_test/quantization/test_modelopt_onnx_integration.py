# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the ModelOpt ONNX backend."""

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
def quant_config(mock_onnx_file, tmp_path):
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
        "results_dir": str(tmp_path / "results"),
    })


@pytest.fixture
def dict_dataloader():
    """Create dataloader with dict format (common in CV tasks)."""
    data = torch.randn(50, 3, 32, 32)
    labels = torch.zeros(50, dtype=torch.long)
    dataset = [({"images": data[i], "labels": labels[i]}) for i in range(50)]
    return torch.utils.data.DataLoader(dataset, batch_size=10)


@pytest.fixture
def large_dataloader():
    """Create a larger dataloader for testing."""
    data = torch.randn(200, 10)
    labels = torch.zeros(200, dtype=torch.long)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data, labels),
        batch_size=20
    )


class TestModelOptONNXIntegration:
    """Integration tests for ModelOpt ONNX backend."""

    @patch('nvidia_tao_pytorch.core.quantization.backends.modelopt_onnx.modelopt_onnx.modelopt_onnx_quantize')
    @patch('os.path.getsize')
    @patch('os.path.exists')
    def test_full_workflow_with_dict_dataloader(self, mock_exists, mock_getsize, mock_quantize, backend_class, quant_config, dict_dataloader):
        """Test complete workflow with dict-based dataloader (common in CV tasks)."""
        backend = backend_class()

        # Prepare
        backend.prepare(None, quant_config)
        assert backend._onnx_path is not None

        # Calibrate with dict dataloader
        backend.calibrate(None, dict_dataloader)
        assert backend._calibration_data is not None
        assert isinstance(backend._calibration_data, np.ndarray)
        assert backend._calibration_data.shape == (50, 3, 32, 32)

        # Mock file operations for quantize
        mock_exists.return_value = True
        mock_getsize.return_value = 1024 * 1024  # 1 MB

        # Quantize
        result = backend.quantize(None, quant_config)
        assert result is not None
        assert isinstance(result, str)
        assert result.endswith(".onnx")
        mock_quantize.assert_called_once()

    def test_calibration_data_extraction_large_dataset(self, backend_class, quant_config, large_dataloader):
        """Test calibration data extraction from larger dataset."""
        backend = backend_class()
        backend.prepare(None, quant_config)

        # Calibrate with large dataloader
        backend.calibrate(None, large_dataloader)

        # Verify calibration data
        assert backend._calibration_data is not None
        assert isinstance(backend._calibration_data, np.ndarray)
        assert backend._calibration_data.shape[0] == 200  # All samples extracted
        assert backend._calibration_data.shape[1] == 10  # Feature dimension

    @patch('nvidia_tao_pytorch.core.quantization.backends.modelopt_onnx.modelopt_onnx.modelopt_onnx_quantize')
    @patch('os.path.getsize')
    @patch('os.path.exists')
    def test_quantization_with_backend_kwargs(self, mock_exists, mock_getsize, mock_quantize, backend_class, mock_onnx_file, tmp_path):
        """Test quantization with additional backend_kwargs parameters."""
        # Create backend with extra ModelOpt parameters
        backend_kwargs = {
            "per_channel": True,
            "reduce_range": True,
            "use_external_data_format": False,
        }
        backend = backend_class(backend_kwargs=backend_kwargs)

        # Config without backend_kwargs
        config = build_model_quant_config_from_omegaconf({
            "backend": "modelopt.onnx",
            "model_path": mock_onnx_file,
            "mode": "static_ptq",
            "algorithm": "max",
            "layers": [{"module_name": "Linear", "weights": {"dtype": "int8"}}],
            "results_dir": str(tmp_path / "results"),
        })

        backend.prepare(None, config)
        backend.set_calibration_data(np.random.randn(10, 3, 32, 32).astype(np.float32))

        # Mock file operations for quantize
        mock_exists.return_value = True
        mock_getsize.return_value = 1024 * 1024  # 1 MB

        # Quantize
        result = backend.quantize(None, config)
        assert result is not None
        assert isinstance(result, str)

        # Verify backend_kwargs were passed to ModelOpt
        call_args = mock_quantize.call_args
        params = call_args.kwargs
        assert params.get("per_channel") is True
        assert params.get("reduce_range") is True
        assert params.get("use_external_data_format") is False
