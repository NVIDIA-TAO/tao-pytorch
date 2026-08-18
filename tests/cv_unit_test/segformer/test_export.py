# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SegFormer Export Unit Tests
"""
import os
import onnx
import subprocess
import sys
import torch
import pytest
import tempfile
import onnxruntime as ort
from omegaconf import OmegaConf

from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.config.segformer.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.segformer.model.segformer_pl_model import SegFormerPlModel
from nvidia_tao_pytorch.cv.segformer.utils.onnx_export import ONNXExporter


tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
# always use batch size 1 for onnx export, otherwise nvdinov2 will thorw gridsample not found error.
BATCH_SIZE = 1
OUTPUT_SHAPE = 224
TEST_TOPOLOGIES = [
    # ConvNeXtV2.
    ("mit_b0"),
    # DINOV2.
    ("vit_large_nvdinov2"),
    # FAN.
    ("fan_tiny_8_p4_hybrid"),
    # OpenCLIP.
    ("vit_base_nvclip_16_siglip"),
    # ("vit_huge_nvclip_14_siglip"),  # Too big to export.
    # RADIO.
    ("c_radio_v2_vit_base_patch16_224"),
]


@pytest.fixture
def _test_experiment_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config["dataset"]['segment']["img_size"] = OUTPUT_SHAPE
    experiment_config["dataset"]['segment']["batch_size"] = BATCH_SIZE
    experiment_config["results_dir"] = tmp_top_dir

    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
@pytest.mark.parametrize("batch_size", [-1])
@pytest.mark.parametrize("opset_version", [17])
def test_segformer_onnx_compare_output(_test_experiment_spec, backbone, batch_size, opset_version):
    """Unit test for ONNX export on SegFormer model.
    Includes unit test for onnxruntime inference and comparison with torch output
    Includes unit test for trt engine generation.
    """

    _test_experiment_spec.model.backbone.type = backbone
    if 'vit' in backbone:
        _test_experiment_spec.model.decode_head.feature_strides = [4, 8, 16, 32]

    # Define input and output of ONNX
    if batch_size == -1:
        input_batch_size = 1
    else:
        input_batch_size = batch_size

    input_channel, input_height, input_width = 3, OUTPUT_SHAPE, OUTPUT_SHAPE
    input_names = ['input']
    output_names = ['output']

    model = SegFormerPlModel(_test_experiment_spec).model
    model.eval()

    dummy_input = torch.ones(1, input_channel, input_height, input_width, device='cpu')
    print(dummy_input.dtype)
    with torch.no_grad():
        torch_output = model(dummy_input)
    print(torch_output[0].dtype)
    print("input shape",dummy_input.shape)
    print("output shape",torch_output.shape)
    onnx_path = os.path.join(tmp_top_dir, f"{backbone}_opset{opset_version}.onnx")
    print("==========================", input_batch_size, batch_size)
    onnx_export = ONNXExporter()
    onnx_export.export_model(
        model, batch_size,
        onnx_path,
        dummy_input,
        input_names=input_names,
        opset_version=opset_version,
        output_names=output_names,
        do_constant_folding=True,
        verbose=_test_experiment_spec.export.verbose
    )

    onnx_export.check_onnx(onnx_path)

    assert os.path.exists(onnx_path), "ONNX file was not generated properly!"

    # Load ONNX and ONNXRuntime
    ort.set_seed(47)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    sess = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']
    )

    dummy_input2 = torch.ones(1, input_channel, input_height, input_width, device='cpu')
    with torch.no_grad():
        torch_output = model(dummy_input2)

    ort_inputs = {sess.get_inputs()[0].name: dummy_input2.detach().cpu().numpy()}

    onnx_output = sess.run(None, ort_inputs)
    print("onnx_len", len(onnx_output))
    print(f"Output shapes: {list(torch_output.shape)} vs {list(onnx_output[0].shape)}")
    if not (list(torch_output.shape) == list(onnx_output[0].shape)):
        raise ValueError(f"Size Mismatch {list(torch_output.shape)} vs {list(onnx_output[0].shape)}")
    else:
        torch.testing.assert_close(torch_output.cpu(), torch.from_numpy(onnx_output[0]), rtol=5e-4, atol=1e-3)


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
@pytest.mark.parametrize("batch_size", [-1])
@pytest.mark.parametrize("opset_version", [17])
def test_segformer_onnx_export(_test_experiment_spec, backbone, batch_size, opset_version):
    """Unit test for ONNX export on SegFormer model.
    Includes unit test for onnxruntime inference and comparison with torch output
    Includes unit test for trt engine generation.
    """

    _test_experiment_spec.model.backbone.type = backbone
    if 'vit' in backbone:
        _test_experiment_spec.model.decode_head.feature_strides = [4, 8, 16, 32]

    # Define input and output of ONNX
    if batch_size == -1:
        input_batch_size = 1
    else:
        input_batch_size = batch_size

    input_channel, input_height, input_width = 3, OUTPUT_SHAPE, OUTPUT_SHAPE
    input_names = ['input']
    output_names = ['output']

    model = SegFormerPlModel(_test_experiment_spec).model
    model.eval()
    model.cuda()

    dummy_input = torch.ones(input_batch_size, input_channel, input_height, input_width, device='cuda')


    onnx_path = os.path.join(tmp_top_dir, f"{backbone}_opset{opset_version}.onnx")

    onnx_export = ONNXExporter()
    onnx_export.export_model(
        model, batch_size,
        onnx_path,
        dummy_input,
        input_names=input_names,
        opset_version=opset_version,
        output_names=output_names,
        do_constant_folding=True,
        verbose=_test_experiment_spec.export.verbose
    )

    onnx_export.check_onnx(onnx_path)

    assert os.path.exists(onnx_path), "ONNX file was not generated properly!"


@pytest.mark.cv_unit
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
@pytest.mark.parametrize("opset_version", [17])
@pytest.mark.skip(reason="Skipping flaky test to be fixed")
def test_cls_trtexec(_test_experiment_spec, backbone, batch_size, opset_version):
    check_and_create(tmp_top_dir)

    onnx_path = os.path.join(tmp_top_dir, f"{backbone}_opset{opset_version}.onnx")

    input_height, input_width = OUTPUT_SHAPE, OUTPUT_SHAPE
    # Test TensorRT engine generation for dynamic batch size ONNX
    call = (
        f"trtexec --onnx={onnx_path} "
        f"--minShapes=input:{batch_size}x3x{input_height}x{input_width} "
        f"--optShapes=input:{batch_size}x3x{input_height}x{input_width} "
        f"--maxShapes=input:{batch_size}x3x{input_height}x{input_width} "
    )
    print(call)

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)


@pytest.mark.cv_unit
def tmp_obj_cleanup():
    tmp_top_obj.cleanup()
