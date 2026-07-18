# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Visual ChangeNet-Segmentation Export Unit Tests
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
from nvidia_tao_pytorch.config.visual_changenet.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models.cn_pl_model import ChangeNetPlModel as ChangeNetPlSegment
from nvidia_tao_pytorch.cv.visual_changenet.utils.onnx_export import ONNXExporter


tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
BATCH_SIZE = 2
OUTPUT_SHAPE = 224
TEST_TOPOLOGIES = [
    ("fan_tiny_8_p4_hybrid"),
    ("vit_large_nvdinov2"),
    ("c_radio_v2_vit_base_patch16_224"),
]
if not os.getenv("CI_PROJECT_DIR", None):
    TEST_TOPOLOGIES.extend([
        ("c_radio_p3_vit_huge_patch16_224_mlpnorm"),
    ])


@pytest.fixture
def _test_experiment_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config["dataset"]['segment']["img_size"] = OUTPUT_SHAPE
    experiment_config["dataset"]['segment']["batch_size"] = BATCH_SIZE
    experiment_config["results_dir"] = tmp_top_dir

    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
@pytest.mark.parametrize("task", ['segment'])
@pytest.mark.parametrize("batch_size", [-1])
@pytest.mark.parametrize("opset_version", [16])
def test_changenet_onnx_compare_output(_test_experiment_spec, backbone, batch_size, task, opset_version):
    """Unit test for ONNX export on ChangeNet model.
    Includes unit test for onnxruntime inference and comparison with torch output
    Includes unit test for trt engine generation.
    """

    _test_experiment_spec.model.backbone.type = backbone
    _test_experiment_spec.task = task
    if 'vit' in backbone:
        _test_experiment_spec.model.decode_head.feature_strides = [4, 8, 16, 32]

    # Define input and output of ONNX
    if batch_size == -1:
        input_batch_size = 1
    else:
        input_batch_size = batch_size

    input_channel, input_height, input_width = 3, OUTPUT_SHAPE, OUTPUT_SHAPE
    input_names = ['input0', 'input1']
    output_names = ["output0", "output1", 'output2', 'output3', 'output_final']

    model = ChangeNetPlSegment(_test_experiment_spec).model
    model.eval()

    dummy_input0 = torch.ones(input_batch_size, input_channel, input_height, input_width, device='cpu')
    dummy_input1 = torch.ones(input_batch_size, input_channel, input_height, input_width, device='cpu')
    dummy_input = (dummy_input0, dummy_input1)

    check_and_create(os.path.join(tmp_top_dir, f"{backbone}_opset{opset_version}_compare"))
    onnx_path = os.path.join(tmp_top_dir, f"{backbone}_opset{opset_version}_compare", f"{backbone}_opset{opset_version}_compare.onnx")

    onnx_export = ONNXExporter()
    onnx_export.export_model(model, batch_size,
                                onnx_path,
                                dummy_input,
                                input_names=input_names,
                                opset_version=opset_version,
                                output_names=output_names,
                                do_constant_folding=True,
                                verbose=_test_experiment_spec.export.verbose,
                                task=task)

    onnx_export.check_onnx(onnx_path)

    assert os.path.exists(onnx_path), "ONNX file was not generated properly!"

    # Load ONNX and ONNXRuntime
    ort.set_seed(47)
    onnx.checker.check_model(onnx_path)
    sess = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']
    )

    with torch.no_grad():
        torch_output = model(dummy_input0, dummy_input1)

    ort_inputs = {sess.get_inputs()[0].name: dummy_input0.detach().cpu().numpy(), sess.get_inputs()[1].name: dummy_input1.detach().cpu().numpy()}

    onnx_output = sess.run(None, ort_inputs)
    for i in range(len(onnx_output)):
        print(list(torch_output[i].shape), list(onnx_output[i].shape))
        if not (list(torch_output[i].shape) == list(onnx_output[i].shape)):
            raise ValueError(f"Size Mismatch {list(torch_output.shape)} vs {list(onnx_output[0].shape)} for output {i}")
        torch.testing.assert_close(torch_output[i].cpu(), torch.from_numpy(onnx_output[i]), rtol=5e-4, atol=1e-3)


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
@pytest.mark.parametrize("task", ['segment'])
@pytest.mark.parametrize("batch_size", [-1])
@pytest.mark.parametrize("opset_version", [16])
def test_changenet_onnx_export(_test_experiment_spec, backbone, batch_size, task, opset_version):
    """Unit test for ONNX export on ChangeNet model.
    Includes unit test for onnxruntime inference and comparison with torch output
    Includes unit test for trt engine generation.
    """

    _test_experiment_spec.model.backbone.type = backbone
    _test_experiment_spec.task = task
    if 'vit' in backbone:
        _test_experiment_spec.model.decode_head.feature_strides = [4, 8, 16, 32]

    # Define input and output of ONNX
    if batch_size == -1:
        input_batch_size = 1
    else:
        input_batch_size = batch_size

    input_channel, input_height, input_width = 3, OUTPUT_SHAPE, OUTPUT_SHAPE
    input_names = ['input0', 'input1']
    output_names = ["output0", "output1", 'output2', 'output3', 'output_final']

    model = ChangeNetPlSegment(_test_experiment_spec).model
    model.eval()
    model.cuda()

    dummy_input0 = torch.ones(input_batch_size, input_channel, input_height, input_width, device='cuda')
    dummy_input1 = torch.ones(input_batch_size, input_channel, input_height, input_width, device='cuda')
    dummy_input = (dummy_input0, dummy_input1)

    check_and_create(os.path.join(tmp_top_dir, f"{backbone}_opset{opset_version}"))
    onnx_path = os.path.join(tmp_top_dir, f"{backbone}_opset{opset_version}", f"{backbone}_opset{opset_version}.onnx")

    onnx_export = ONNXExporter()
    onnx_export.export_model(model, batch_size,
                                onnx_path,
                                dummy_input,
                                input_names=input_names,
                                opset_version=opset_version,
                                output_names=output_names,
                                do_constant_folding=True,
                                verbose=_test_experiment_spec.export.verbose,
                                task=task)

    onnx_export.check_onnx(onnx_path)

    assert os.path.exists(onnx_path), "ONNX file was not generated properly!"


def _export_onnx_if_not_exists(_test_experiment_spec, backbone, opset_version):
    """Helper function to export ONNX model if it doesn't exist."""
    onnx_path = os.path.join(tmp_top_dir, f"{backbone}_opset{opset_version}", f"{backbone}_opset{opset_version}.onnx")

    if os.path.exists(onnx_path):
        return onnx_path

    _test_experiment_spec.model.backbone.type = backbone
    _test_experiment_spec.task = 'segment'
    if 'vit' in backbone:
        _test_experiment_spec.model.decode_head.feature_strides = [4, 8, 16, 32]

    input_batch_size = 1
    input_channel, input_height, input_width = 3, OUTPUT_SHAPE, OUTPUT_SHAPE
    input_names = ['input0', 'input1']
    output_names = ["output0", "output1", 'output2', 'output3', 'output_final']

    model = ChangeNetPlSegment(_test_experiment_spec).model
    model.eval()
    model.cuda()

    dummy_input0 = torch.ones(input_batch_size, input_channel, input_height, input_width, device='cuda')
    dummy_input1 = torch.ones(input_batch_size, input_channel, input_height, input_width, device='cuda')
    dummy_input = (dummy_input0, dummy_input1)

    check_and_create(os.path.join(tmp_top_dir, f"{backbone}_opset{opset_version}"))

    onnx_export = ONNXExporter()
    onnx_export.export_model(model, -1,
                             onnx_path,
                             dummy_input,
                             input_names=input_names,
                             opset_version=opset_version,
                             output_names=output_names,
                             do_constant_folding=True,
                             verbose=_test_experiment_spec.export.verbose,
                             task='segment')

    onnx_export.check_onnx(onnx_path)
    return onnx_path


@pytest.mark.cv_unit
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
@pytest.mark.parametrize("opset_version", [16])
def test_cls_trtexec(_test_experiment_spec, backbone, batch_size, opset_version):
    # Ensure ONNX file exists (export if needed)
    onnx_path = _export_onnx_if_not_exists(_test_experiment_spec, backbone, opset_version)

    input_height, input_width = OUTPUT_SHAPE, OUTPUT_SHAPE
    # Test TensorRT engine generation for dynamic batch size ONNX
    call = (
        f"trtexec --onnx={onnx_path} "
        f"--minShapes=input0:{batch_size}x3x{input_height}x{input_width},input1:{batch_size}x3x{input_height}x{input_width} "
        f"--optShapes=input0:{batch_size}x3x{input_height}x{input_width},input1:{batch_size}x3x{input_height}x{input_width} "
        f"--maxShapes=input0:{batch_size}x3x{input_height}x{input_width},input1:{batch_size}x3x{input_height}x{input_width} "
    )
    print(call)

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)


@pytest.mark.cv_unit
def tmp_obj_cleanup():
    tmp_top_obj.cleanup()
