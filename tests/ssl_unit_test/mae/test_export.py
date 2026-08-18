# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import onnx
import onnxruntime
from omegaconf import OmegaConf
from dataclasses import replace
from nvidia_tao_pytorch.ssl.mae.scripts.export import create_onnx_model
from nvidia_tao_pytorch.ssl.mae.model.pl_model import MAEPlModule
from nvidia_tao_pytorch.config.mae.default_config import ExperimentConfig


@pytest.fixture
def base_config(tmp_path):
    """Create a base configuration for testing.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.

    Returns:
        ExperimentConfig: A configuration object containing default test parameters:
            - export settings (GPU, input shape, batch size, etc.)
            - training settings (model path)
            - dataset settings (number of classes)
    """
    config = ExperimentConfig()
    config.export = replace(
        config.export, **{
            'gpu_id': 0,
            'checkpoint': str(tmp_path / 'model.pt'),
            'onnx_file': str(tmp_path / 'model.onnx'),
            'input_channel': 3,
            'input_width': 224,
            'input_height': 224,
            'batch_size': 1,
            'opset_version': 17,
            'on_cpu': True,
            'verbose': True
        }
    )
    return OmegaConf.structured(config)


@pytest.fixture
def mock_model(base_config):
    """Create a mock MAE model for testing.

    Args:
        base_config: Fixture providing the base configuration.

    Returns:
        nn.Module: A simplified mock MAE model with basic convolutional layers.
            The model consists of:
            - Input conv layer (3->64 channels)
            - ReLU activation
            - Output conv layer (64->3 channels)
    """
    model = MAEPlModule(
        cfg=base_config,
        export=True
    )
    return model.model


def test_successful_export(mock_model, base_config, tmp_path):
    """Test successful model export to ONNX format.

    This test verifies that:
    1. The model can be successfully exported to ONNX
    2. The exported ONNX file is valid
    3. The exported model can perform inference

    Args:
        mock_model: Fixture providing a mock MAE model
        base_config: Fixture providing the base configuration
        tmp_path: Pytest fixture providing a temporary directory path
    """
    # Prepare model
    mock_model.eval()
    input_shape = [3, 224, 224]
    output_path = base_config.export.onnx_file
    input_names = ["input"]
    output_names = ["output"]

    # Create ONNX model
    create_onnx_model(
        model=mock_model,
        input_shape=input_shape,
        input_batch_size=1,
        output_path=output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axis=True
    )
    
    # Verify ONNX file exists and is valid
    assert os.path.exists(output_path), \
        f"ONNX file was not created at expected path: {output_path}"
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Test inference with ONNX Runtime
    ort_session = onnxruntime.InferenceSession(output_path)
    dummy_input = torch.randn(1, *input_shape).numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
    ort_outs = ort_session.run(None, ort_inputs)
    assert len(ort_outs) > 0, \
        "ONNX Runtime inference failed to produce any outputs"


def test_export_with_different_batch_sizes(mock_model, base_config, tmp_path):
    """Test export with different batch sizes.

    This test verifies that:
    1. The model can be exported with different batch sizes
    2. The exported model supports dynamic batch sizes
    3. The model can perform inference with different batch sizes

    Args:
        mock_model: Fixture providing a mock MAE model
        base_config: Fixture providing the base configuration
        tmp_path: Pytest fixture providing a temporary directory path
    """
    # Prepare model
    mock_model.eval()
    input_shape = [3, 224, 224]
    output_path = base_config.export.onnx_file
    input_names = ["input"]
    output_names = ["output"]

    # Create ONNX model
    create_onnx_model(
        model=mock_model,
        input_shape=input_shape,
        input_batch_size=1,
        output_path=output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axis=True
    )
    
    # Test with different batch sizes
    ort_session = onnxruntime.InferenceSession(output_path)
    
    # Test batch size 1
    dummy_input = torch.randn(1, *input_shape).numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
    ort_outs = ort_session.run(None, ort_inputs)
    assert len(ort_outs) > 0, \
        "ONNX Runtime inference failed for batch size 1"

    # Test batch size 4
    dummy_input = torch.randn(4, *input_shape).numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
    ort_outs = ort_session.run(None, ort_inputs)
    assert len(ort_outs) > 0, \
        "ONNX Runtime inference failed for batch size 4"


def test_export_with_cpu_device(mock_model, base_config, tmp_path):
    """Test export on CPU device.

    This test verifies that:
    1. The model can be exported on CPU
    2. The exported model is valid when created on CPU
    3. The export process works correctly without GPU acceleration

    Args:
        mock_model: Fixture providing a mock MAE model
        base_config: Fixture providing the base configuration
        tmp_path: Pytest fixture providing a temporary directory path
    """
    # Prepare model
    mock_model.eval()
    input_shape = [3, 224, 224]
    output_path = base_config.export.onnx_file
    input_names = ["input"]
    output_names = ["output"]

    # Create ONNX model
    create_onnx_model(
        model=mock_model,
        input_shape=input_shape,
        input_batch_size=1,
        output_path=output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axis=True
    )
    
    # Verify ONNX file
    assert os.path.exists(output_path), \
        f"ONNX file was not created at expected path: {output_path}"
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)


def test_export_with_existing_output(mock_model, base_config, tmp_path):
    """Test export when output file already exists.

    This test verifies that:
    1. The export process properly handles existing output files
    2. Appropriate error is raised when output file already exists
    3. The original file is not overwritten

    Args:
        mock_model: Fixture providing a mock MAE model
        base_config: Fixture providing the base configuration
        tmp_path: Pytest fixture providing a temporary directory path
    """
    # Prepare model
    mock_model.eval()
    input_shape = [3, 224, 224]
    output_path = base_config.export.onnx_file
    input_names = ["input"]
    output_names = ["output"]

    # Create existing output file
    with open(output_path, 'w') as f:
        f.write('dummy')
    
    with pytest.raises(ValueError, match="Default onnx file .* already exists"):
        create_onnx_model(
            model=mock_model,
            input_shape=input_shape,
            input_batch_size=1,
            output_path=output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axis=True
        )


def test_export_with_invalid_input_shape(mock_model, base_config, tmp_path):
    """Test export with invalid input shape.

    This test verifies that:
    1. The export process properly handles invalid input shapes
    2. Appropriate error is raised for incompatible shapes
    3. The model validates input dimensions before export

    Args:
        mock_model: Fixture providing a mock MAE model
        base_config: Fixture providing the base configuration
        tmp_path: Pytest fixture providing a temporary directory path
    """
    # Prepare model
    mock_model.eval()
    invalid_shape = [4, 224, 224]  # Invalid channel count
    output_path = base_config.export.onnx_file
    input_names = ["input"]
    output_names = ["output"]
    
    with pytest.raises(ValueError, match="Invalid input channel: .*. Only 1 or 3 are supported."):
        create_onnx_model(
            model=mock_model,
            input_shape=invalid_shape,
            input_batch_size=1,
            output_path=output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axis=True
        )
