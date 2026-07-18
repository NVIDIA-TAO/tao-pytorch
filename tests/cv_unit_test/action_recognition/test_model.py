# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import onnx
from onnxsim import simplify
import onnx_graphsurgeon as gs
import os
import pytest
import torch
from omegaconf import OmegaConf
from nvidia_tao_pytorch.config.action_recognition.default_config import ARModelConfig, ARDatasetConfig
from nvidia_tao_pytorch.cv.action_recognition.model.build_nn_model import build_ar_model


@pytest.fixture
def _test_experiment_spec():
    dataset_config = OmegaConf.structured(ARDatasetConfig())
    dataset_config.label_map = {"a": 0, "b": 1, "c": 2}
    model_config = OmegaConf.structured(ARModelConfig())
    experiment_config = {"dataset": dataset_config,
                         "model": model_config}
    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone, model_type, input_type, export",
                         [("resnet_18", "rgb", "2d", False),
                          ("resnet_34", "of", "2d", False),
                          ("resnet_50", "joint", "2d", False),
                          ("resnet_101", "rgb", "3d", False),
                          ("resnet_18", "of", "3d", False),
                          ("resnet_18", "joint", "3d", True)])
def test_ar_model(_test_experiment_spec, backbone, model_type, input_type, export):
    _test_experiment_spec["model"].backbone = backbone
    _test_experiment_spec["model"].model_type = model_type
    _test_experiment_spec["model"].input_type = input_type

    build_ar_model(_test_experiment_spec, False, export)


@pytest.mark.cv_unit
def test_i3d_model_export(_test_experiment_spec):
    _test_experiment_spec["model"].backbone="i3d"
    _test_experiment_spec["model"].model_type="rgb"
    _test_experiment_spec["model"].input_type="3d"
    _test_experiment_spec["model"].rgb_seq_length=64

    model = build_ar_model(_test_experiment_spec, False, False)
    model = model.cuda()

    input_names = ["input_rgb"]
    output_names = ["fc_pred"]

    # create dummy input
    output_shape = [_test_experiment_spec["model"]["input_height"],
                    _test_experiment_spec["model"]["input_width"]]
    rgb_seq_length = _test_experiment_spec['model']['rgb_seq_length']

    dummy_input = torch.randn(3, 3, rgb_seq_length,
                                output_shape[0], output_shape[1]).cuda()
    dynamic_axes = {"input_rgb": {0: "batch"}, "fc_pred": {0: "batch"}}

    output_file = "/tmp/i3d_test.onnx"
    # export
    torch.onnx.export(model,
                      dummy_input,
                      output_file,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      opset_version=17,
                      verbose=True,
                      dynamo=False)

    optimized_model, _ = simplify(onnx.load(output_file))
    graph = gs.import_onnx(optimized_model)
    graph.cleanup()

    for node in graph.nodes:
        if node.op == "If":
            os.remove(output_file)
            raise ValueError("Unexpected If node in exported ONNX model.")

    os.remove(output_file)
