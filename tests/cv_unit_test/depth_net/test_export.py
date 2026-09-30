
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
import nvidia_tao_pytorch.cv.depth_net


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


@pytest.fixture
def stereo_config(tmp_path):
    """FastFoundationStereo config taken from the shipped bp2 spec.

    The bp2 topology is a tightly coupled set of width overrides, so the spec is loaded rather
    than hand-assembled -- the schema defaults are the full-FoundationStereo ones and do not
    form a valid FastFoundationStereo. Only the export block is shrunk, to keep the test quick.
    """
    spec = os.path.join(
        os.path.dirname(nvidia_tao_pytorch.cv.depth_net.__file__),
        "experiment_specs", "experiment_fast_foundation_stereo.yaml",
    )
    experiment_config = OmegaConf.merge(
        OmegaConf.structured(ExperimentConfig()), OmegaConf.load(spec),
    )
    experiment_config.model.valid_iters = 1
    experiment_config.export.onnx_file = os.path.join(tmp_path, "stereo_dynbatch.onnx")
    experiment_config.export.input_channel = 3
    experiment_config.export.input_height = 64
    experiment_config.export.input_width = 96
    experiment_config.export.batch_size = -1
    experiment_config.export.opset_version = 17
    yield experiment_config


TRACE_BATCH = 1
PROBE_BATCH = 2


def _assert_batch_axis_is_dynamic(output_path, trace_batch=TRACE_BATCH, probe_batch=PROBE_BATCH):
    """Assert an exported stereo graph is valid at a batch size it was not traced at.

    Two independent checks:
      1. no axis-0 Slice on a 4-D tensor carries a constant bound equal to `trace_batch`
      2. shape inference with the input batch pinned to `probe_batch` yields that output batch
    """
    from onnx import numpy_helper, shape_inference

    model_onnx = shape_inference.infer_shapes(onnx.load(output_path))

    # Only Constant *nodes* can supply Slice bounds here, and materialising every weight
    # initializer would cost hundreds of MB for nothing.
    const = {}
    for node in model_onnx.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    const[node.output[0]] = numpy_helper.to_array(attr.t).tolist()
    for tensor in model_onnx.graph.initializer:
        if len(tensor.dims) <= 1 and int(numpy_helper.to_array(tensor).size) <= 8:
            const[tensor.name] = numpy_helper.to_array(tensor).tolist()

    ranks = {vi.name: len(vi.type.tensor_type.shape.dim)
             for vi in list(model_onnx.graph.value_info) + list(model_onnx.graph.input)}

    examined = 0
    for node in model_onnx.graph.node:
        if node.op_type != "Slice" or len(node.input) < 4:
            continue
        if const.get(node.input[3]) != [0]:
            continue
        # Rank distinguishes a feature slice from a shape-vector slice (shape vectors are
        # rank 1). Rank 3 is included so ViT token tensors (B, N, C) are covered too --
        # FoundationStereo slices vit_feat with the same batch size. A missing rank must not
        # silently skip the node, or the test could "pass" having checked nothing; the
        # `examined` counter below guards that.
        if ranks.get(node.input[0]) not in (3, 4):
            continue
        examined += 1
        starts, ends = const.get(node.input[1]), const.get(node.input[2])
        assert ends != [trace_batch] and starts != [trace_batch], (
            f"Slice node {node.name!r} has the traced batch size baked into its bounds "
            f"(starts={starts}, ends={ends}). Read the batch size with .shape[0], not len(), "
            f"so the tracer keeps it dynamic."
        )

    # Guard against the check above degenerating into a no-op: this graph splits left from right
    # with axis-0 slices on 4-D feature maps, so there must be some.
    assert examined > 0, (
        "no axis-0 Slice on a 3-D/4-D tensor was examined, so assertion 1 checked nothing. "
        "Shape inference probably failed to assign ranks; the test cannot vouch for this graph."
    )

    for tensor in model_onnx.graph.input:
        dim = tensor.type.tensor_type.shape.dim[0]
        dim.dim_value = probe_batch
        dim.ClearField("dim_param")
    inferred = shape_inference.infer_shapes(model_onnx, strict_mode=True, data_prop=True)
    out_batch = inferred.graph.output[0].type.tensor_type.shape.dim[0].dim_value
    assert out_batch == probe_batch, (
        f"with the input batch pinned to {probe_batch} the output batch inferred as {out_batch}; "
        "the batch axis is annotated dynamic but the graph body is not batch-polymorphic"
    )


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.export
def test_stereo_export_batch_axis_is_really_dynamic(stereo_config, tmp_path):
    """A dynamic-batch stereo export must be valid at batch sizes other than the traced one.

    Regression test. The stereo models run left and right through the backbone as a single
    batch-axis concat and then un-concatenate by slicing with the batch size. When that batch
    size is read with len(), it is a Python int and torch.onnx's tracer freezes it into the
    graph, so the slices become Slice(starts=[0], ends=[1]) / Slice(starts=[1], ends=[INT_MAX]).
    The resulting ONNX is annotated dynamic but is only numerically valid at the traced batch
    size of 1 — TensorRT rejects it with "reshape changes volume" and ONNX Runtime with a
    Reshape volume mismatch.

    Two assertions, either of which alone would have caught the defect:
      1. no axis-0 Slice on a feature tensor carries a constant bound equal to the trace batch
      2. shape inference with the input batch pinned to 2 yields an output batch of 2

    Only FastFoundationStereo is exercised. FoundationStereo shares the idiom verbatim, and has
    one more frozen site (it slices `vit_feat` as well), but exporting it needs a multiple-of-14
    input for the ViT backbone and produces a ~1.5 GB ONNX in ~55 s, which is too heavy for a
    unit test. `_assert_batch_axis_is_dynamic` is factored out so that gap can be closed cheaply
    if the parent model gains a lighter export path.
    """
    from nvidia_tao_pytorch.cv.depth_net.scripts.export import stereo_onnx_export

    model = build_pl_model(stereo_config, export=True)
    model.eval()
    output_path = stereo_config.export.onnx_file

    stereo_onnx_export(
        model=model.model,
        input_shape=[stereo_config.export.input_channel,
                     stereo_config.export.input_height,
                     stereo_config.export.input_width],
        input_batch_size=TRACE_BATCH,
        output_file=output_path,
        on_cpu=True,
        opset_version=stereo_config.export.opset_version,
        valid_iters=stereo_config.model.valid_iters,
        dynamic_axis=True,
    )
    _assert_batch_axis_is_dynamic(output_path)
