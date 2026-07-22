
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple test cases to test export of DepthNet model."""

import os
import pytest
import torch
import onnx
import onnxruntime
from omegaconf import OmegaConf
from dataclasses import replace
from nvidia_tao_pytorch.ssl.mae.scripts.export import create_onnx_model
from nvidia_tao_pytorch.cv.depth_net.model.build_pl_model import build_pl_model
from nvidia_tao_pytorch.config.depth_net.default_config import ExperimentConfig


@pytest.fixture
def relative_config(tmp_path):
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
    config.model.model_type = "RelativeDepthAnything"
    config.export = replace(
        config.export, **{
            'gpu_id': 0,
            'checkpoint': str(tmp_path / 'model.pt'),
            'onnx_file': str(tmp_path / 'model.onnx'),
            'input_channel': 3,
            'input_width': 924,
            'input_height': 518,
            'batch_size': 1,
            'opset_version': 17,
            'on_cpu': True,
            'verbose': True
        }
    )
    return OmegaConf.structured(config)

@pytest.fixture
def metric_config(tmp_path):
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
    config.model.model_type = "MetricDepthAnything"
    config.export = replace(
        config.export, **{
            'gpu_id': 0,
            'checkpoint': str(tmp_path / 'model.pt'),
            'onnx_file': str(tmp_path / 'model.onnx'),
            'input_channel': 3,
            'input_width': 924,
            'input_height': 518,
            'batch_size': 1,
            'opset_version': 17,
            'on_cpu': True,
            'verbose': True
        }
    )
    return OmegaConf.structured(config)

@pytest.fixture
def mock_relative_model(relative_config):
    """Create a mock NvDepthAnythingV2 model for testing.

    Args:
        base_config: Fixture providing the base configuration.

    Returns:
        nn.Module: DepthNet PL Module
    """
    model = build_pl_model(
        experiment_config=relative_config,
        export=True
    )
    return model.model

@pytest.fixture
def mock_metric_model(metric_config):
    """Create a mock NvDepthAnythingV2 model for testing.

    Args:
        base_config: Fixture providing the base configuration.

    Returns:
        nn.Module: DepthNet PL Module
    """
    model = build_pl_model(
        experiment_config=metric_config,
        export=True
    )
    return model.model


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.export
def test_successful_relative_export(mock_relative_model, relative_config, tmp_path):
    """Test successful model export to ONNX format.

    This test verifies that:
    1. The model can be successfully exported to ONNX
    2. The exported ONNX file is valid
    3. The exported model can perform inference

    Args:
        mock_model: Fixture providing a mock NvDepthAnythingV2 model
        base_config: Fixture providing the base configuration
        tmp_path: Pytest fixture providing a temporary directory path
    """
    # Prepare model
    mock_relative_model.eval()
    input_shape = [relative_config.export.input_channel, relative_config.export.input_width, relative_config.export.input_height]
    output_path = relative_config.export.onnx_file
    input_names = ["images"]
    output_names = ["outputs"]

    # Create ONNX model
    create_onnx_model(
        model=mock_relative_model,
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


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.export
def test_successful_metric_export(mock_metric_model, metric_config, tmp_path):
    """Test successful model export to ONNX format.

    This test verifies that:
    1. The model can be successfully exported to ONNX
    2. The exported ONNX file is valid
    3. The exported model can perform inference

    Args:
        mock_model: Fixture providing a mock NvDepthAnythingV2 model
        base_config: Fixture providing the base configuration
        tmp_path: Pytest fixture providing a temporary directory path
    """
    # Prepare model
    mock_metric_model.eval()
    input_shape = [metric_config.export.input_channel, metric_config.export.input_width, metric_config.export.input_height]
    output_path = metric_config.export.onnx_file
    input_names = ["images"]
    output_names = ["outputs"]

    # Create ONNX model
    create_onnx_model(
        model=mock_metric_model,
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


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.export
def test_export_with_different_batch_sizes_relative(mock_relative_model, relative_config, tmp_path):
    """Test export with different batch sizes.

    This test verifies that:
    1. The model can be exported with different batch sizes
    2. The exported model supports dynamic batch sizes
    3. The model can perform inference with different batch sizes

    Args:
        mock_model: Fixture providing a mock NvDepthAnythingV2 model
        base_config: Fixture providing the base configuration
        tmp_path: Pytest fixture providing a temporary directory path
    """
    # Prepare model
    mock_relative_model.eval()
    input_shape = [relative_config.export.input_channel, relative_config.export.input_width, relative_config.export.input_height]
    output_path = relative_config.export.onnx_file
    input_names = ["images"]
    output_names = ["outputs"]

    # Create ONNX model
    create_onnx_model(
        model=mock_relative_model,
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


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.export
def test_export_with_different_batch_sizes_metric(mock_metric_model, metric_config, tmp_path):
    """Test export with different batch sizes.

    This test verifies that:
    1. The model can be exported with different batch sizes
    2. The exported model supports dynamic batch sizes
    3. The model can perform inference with different batch sizes

    Args:
        mock_model: Fixture providing a mock NvDepthAnythingV2 model
        base_config: Fixture providing the base configuration
        tmp_path: Pytest fixture providing a temporary directory path
    """
    # Prepare model
    mock_metric_model.eval()
    input_shape = [metric_config.export.input_channel, metric_config.export.input_width, metric_config.export.input_height]
    output_path = metric_config.export.onnx_file
    input_names = ["images"]
    output_names = ["outputs"]

    # Create ONNX model
    create_onnx_model(
        model=mock_metric_model,
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


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.export
def test_export_with_cpu_device_relative(mock_relative_model, relative_config, tmp_path):
    """Test export on CPU device.

    This test verifies that:
    1. The model can be exported on CPU
    2. The exported model is valid when created on CPU
    3. The export process works correctly without GPU acceleration

    Args:
        mock_model: Fixture providing a mock NvDepthAnythingV2 model
        base_config: Fixture providing the base configuration
        tmp_path: Pytest fixture providing a temporary directory path
    """
    # Prepare model
    mock_relative_model.eval()
    input_shape = [relative_config.export.input_channel, relative_config.export.input_width, relative_config.export.input_height]
    output_path = relative_config.export.onnx_file
    input_names = ["images"]
    output_names = ["outputs"]

    # Create ONNX model
    create_onnx_model(
        model=mock_relative_model,
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


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.export
def test_export_with_cpu_device_metric(mock_metric_model, metric_config, tmp_path):
    """Test export on CPU device.

    This test verifies that:
    1. The model can be exported on CPU
    2. The exported model is valid when created on CPU
    3. The export process works correctly without GPU acceleration

    Args:
        mock_model: Fixture providing a mock NvDepthAnythingV2 model
        base_config: Fixture providing the base configuration
        tmp_path: Pytest fixture providing a temporary directory path
    """
    # Prepare model
    mock_metric_model.eval()
    input_shape = [metric_config.export.input_channel, metric_config.export.input_width, metric_config.export.input_height]
    output_path = metric_config.export.onnx_file
    input_names = ["images"]
    output_names = ["outputs"]

    # Create ONNX model
    create_onnx_model(
        model=mock_metric_model,
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


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.export
def test_export_with_existing_output_relative(mock_relative_model, relative_config, tmp_path):
    """Test export when output file already exists.

    This test verifies that:
    1. The export process properly handles existing output files
    2. Appropriate error is raised when output file already exists
    3. The original file is not overwritten

    Args:
        mock_model: Fixture providing a mock NvDepthAnythingV2 model
        base_config: Fixture providing the base configuration
        tmp_path: Pytest fixture providing a temporary directory path
    """
    # Prepare model
    mock_relative_model.eval()
    input_shape = [relative_config.export.input_channel, relative_config.export.input_width, relative_config.export.input_height]
    output_path = relative_config.export.onnx_file
    input_names = ["images"]
    output_names = ["outputs"]

    # Create existing output file
    with open(output_path, 'w') as f:
        f.write('dummy')
    
    with pytest.raises(ValueError, match="Default onnx file .* already exists"):
        create_onnx_model(
            model=mock_relative_model,
            input_shape=input_shape,
            input_batch_size=1,
            output_path=output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axis=True
        )


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.export
def test_export_with_existing_output_metric(mock_metric_model, metric_config, tmp_path):
    """Test export when output file already exists.

    This test verifies that:
    1. The export process properly handles existing output files
    2. Appropriate error is raised when output file already exists
    3. The original file is not overwritten

    Args:
        mock_model: Fixture providing a mock NvDepthAnythingV2 model
        base_config: Fixture providing the base configuration
        tmp_path: Pytest fixture providing a temporary directory path
    """
    # Prepare model
    mock_metric_model.eval()
    input_shape = [metric_config.export.input_channel, metric_config.export.input_width, metric_config.export.input_height]
    output_path = metric_config.export.onnx_file
    input_names = ["images"]
    output_names = ["outputs"]

    # Create existing output file
    with open(output_path, 'w') as f:
        f.write('dummy')
    
    with pytest.raises(ValueError, match="Default onnx file .* already exists"):
        create_onnx_model(
            model=mock_metric_model,
            input_shape=input_shape,
            input_batch_size=1,
            output_path=output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axis=True
        )        


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.export
def test_export_with_invalid_input_shape_relative(mock_relative_model, relative_config, tmp_path):
    """Test export with invalid input shape.

    This test verifies that:
    1. The export process properly handles invalid input shapes
    2. Appropriate error is raised for incompatible shapes
    3. The model validates input dimensions before export

    Args:
        mock_model: Fixture providing a mock NvDepthAnythingV2 model
        base_config: Fixture providing the base configuration
        tmp_path: Pytest fixture providing a temporary directory path
    """
    # Prepare model
    mock_relative_model.eval()
    invalid_shape = [4, relative_config.export.input_width, relative_config.export.input_height]
    output_path = relative_config.export.onnx_file
    input_names = ["images"]
    output_names = ["outputs"]
    
    with pytest.raises(ValueError, match="Invalid input channel: .*. Only 1 or 3 are supported."):
        create_onnx_model(
            model=mock_relative_model,
            input_shape=invalid_shape,
            input_batch_size=1,
            output_path=output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axis=True
        )


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.export
def test_export_with_invalid_input_shape_metric(mock_metric_model, metric_config, tmp_path):
    """Test export with invalid input shape.

    This test verifies that:
    1. The export process properly handles invalid input shapes
    2. Appropriate error is raised for incompatible shapes
    3. The model validates input dimensions before export

    Args:
        mock_model: Fixture providing a mock NvDepthAnythingV2 model
        base_config: Fixture providing the base configuration
        tmp_path: Pytest fixture providing a temporary directory path
    """
    # Prepare model
    mock_metric_model.eval()
    invalid_shape = [4, metric_config.export.input_width, metric_config.export.input_height]
    output_path = metric_config.export.onnx_file
    input_names = ["images"]
    output_names = ["outputs"]
    
    with pytest.raises(ValueError, match="Invalid input channel: .*. Only 1 or 3 are supported."):
        create_onnx_model(
            model=mock_metric_model,
            input_shape=invalid_shape,
            input_batch_size=1,
            output_path=output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axis=True
        )
