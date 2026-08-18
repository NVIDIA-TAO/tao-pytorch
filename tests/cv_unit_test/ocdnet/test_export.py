# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf

import os
import copy
import tempfile
import onnx
import torch
import onnx_graphsurgeon as onnx_gs
import onnxruntime as ort

from torch.onnx import register_custom_op_symbolic
from torchvision.ops import DeformConv2d
from nvidia_tao_pytorch.config.ocdnet.default_config import OCDNetModelConfig
from nvidia_tao_pytorch.cv.ocdnet.model.model import Model

TEST_CHANNEL = 3
TEST_HEIGHT = 640
TEST_WIDTH = 640
OPSET_VERSION = 11


def symbolic_dcnv2_forward(g, *inputs):
    """symbolic_dcnv2_forward"""
    # weights as last input to align with TRT plugin
    return g.op("ModulatedDeformConv2d", inputs[0], inputs[2], inputs[3], inputs[1])


@pytest.fixture
def _test_tensor():
    torch.manual_seed(47)
    tensor = torch.randn(1, TEST_CHANNEL, TEST_HEIGHT, TEST_WIDTH)
    
    yield tensor

@pytest.fixture
def _test_dcnresnet_model_spec():
    model_config = OmegaConf.structured(OCDNetModelConfig())
    model_config = OmegaConf.to_container(model_config)
    model_config["backbone"] = 'deformable_resnet18'
    model_config["neck"] = 'FPN'
    model_config["load_pruned_graph"] = False

    yield model_config


@pytest.fixture
def _test_fan_model_spec():
    model_config = OmegaConf.structured(OCDNetModelConfig())
    model_config = OmegaConf.to_container(model_config)
    model_config["backbone"] = 'fan_tiny_8_p4_hybrid'
    model_config["neck"] = 'FANNeck'
    model_config["load_pruned_graph"] = False
    model_config["enlarge_feature_map_size"] = True

    yield model_config

@pytest.mark.cv_unit
def test_ocdnet_dcnresnet_onnx_export(_test_dcnresnet_model_spec, _test_tensor):
    """Unit test for ONNX export on OCDNet with backbone deformable_resnet18 """
    # Register custom symbolic function

    register_custom_op_symbolic("torchvision::deform_conv2d", symbolic_dcnv2_forward, OPSET_VERSION)
    model = Model(_test_dcnresnet_model_spec)
    model.eval()

    dummy_input = _test_tensor
    os_handle, tmp_onnx_file = tempfile.mkstemp(suffix=".onnx")
    os.close(os_handle)

    with torch.no_grad():
        torch.onnx.export(
            model, (dummy_input,),
            tmp_onnx_file,
            opset_version=OPSET_VERSION,
            keep_initializers_as_inputs=True,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            input_names=['input'],
            output_names=['pred'],
            dynamic_axes={
                "input": {0: "batch"},
            },
            dynamo=False,
        )
    onnx_model = onnx.load(tmp_onnx_file)
    gs_graph = onnx_gs.import_onnx(onnx_model)
    layer_dict = {}
    attrs_dict = {}
    for name, layer in model.named_modules():
        if isinstance(layer, DeformConv2d):
            attrs_dict["stride"] = list(layer.stride)
            attrs_dict["padding"] = list(layer.padding)
            attrs_dict["dilation"] = list(layer.dilation)
            attrs_dict["group"] = 1
            attrs_dict["deformable_group"] = 1
            name = name.replace("backbone.", "") + ".ModulatedDeformConv2d"
            layer_dict[name] = copy.deepcopy(attrs_dict)
    for node in gs_graph.nodes:
        if node.op == "ModulatedDeformConv2d":
            key = (".".join(node.name.split("/")[-3:]))
            node.attrs = layer_dict[key]

    gs_graph.fold_constants(size_threshold=1024 * 1024 * 1024)
    gs_graph.cleanup().toposort()
    new_onnx_model = onnx_gs.export_onnx(gs_graph)

    onnx.save(new_onnx_model, tmp_onnx_file)
    assert os.path.exists(tmp_onnx_file), "ONNX file was not generated properly!"

    os.remove(tmp_onnx_file)

@pytest.mark.cv_unit
def test_ocdnet_fan_onnx_export(_test_fan_model_spec, _test_tensor):
    """Unit test for ONNX export on OCDNet with backbone deformable_resnet18 """

    model = Model(_test_fan_model_spec)
    model.eval()

    dummy_input = _test_tensor
    os_handle, tmp_onnx_file = tempfile.mkstemp(suffix=".onnx")
    os.close(os_handle)

    with torch.no_grad():
        torch.onnx.export(
            model, (dummy_input,),
            tmp_onnx_file,
            opset_version=OPSET_VERSION,
            keep_initializers_as_inputs=True,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            input_names=['input'],
            output_names=['pred'],
            dynamic_axes={
                "input": {0: "batch"},
            },
            dynamo=False,
        )
    
    assert os.path.exists(tmp_onnx_file), "ONNX file was not generated properly!"

    # Load ONNX and ONNXRuntime
    ort.set_seed(47)
    onnx_model = onnx.load(tmp_onnx_file)
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(
            tmp_onnx_file
        )

    with torch.no_grad():
        torch_output = model(dummy_input)

    onnx_output = sess.run(None, {
        "input": dummy_input.numpy()
    })

    if not (list(torch_output.shape) == list(onnx_output[0].shape)):
        raise ValueError(f"Size Mismatch {list(torch_output.shape)} vs {list(onnx_output[0].shape)}")
    torch.testing.assert_close(torch_output.cpu(), torch.from_numpy(onnx_output[0]), rtol=1e-5, atol=1e-4)
    os.remove(tmp_onnx_file)