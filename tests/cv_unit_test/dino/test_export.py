# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf

import os
import tempfile
import onnx
import onnxruntime as ort
import torch

from nvidia_tao_pytorch.config.dino.dataset import DINODatasetConfig
from nvidia_tao_pytorch.config.dino.default_config import ExperimentConfig
from nvidia_tao_pytorch.config.dino.model import DINOModelConfig
from nvidia_tao_pytorch.cv.dino.model.build_nn_model import build_model
from nvidia_tao_pytorch.cv.dino.utils.onnx_export import ONNXExporter


@pytest.fixture
def _test_experiment_spec():
    dataset_config = OmegaConf.structured(DINODatasetConfig())
    model_config = OmegaConf.structured(DINOModelConfig())
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset = dataset_config
    experiment_config.model = model_config
    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", ["resnet_50"])
@pytest.mark.parametrize("batch_size", [-1, 1])
def test_dino_onnx_export(_test_experiment_spec, backbone, batch_size):
    """Unit test for ONNX export on DINO model. Here, custom DMHA is used."""
    _test_experiment_spec["model"].backbone = backbone
    _test_experiment_spec["model"].aux_loss = False
    _test_experiment_spec["model"].num_feature_levels = 2
    _test_experiment_spec["model"].return_interm_indices = [1, 2]
    _test_experiment_spec["model"].num_queries = 100

    # Define input and output of ONNX
    if batch_size == -1:
        input_batch_size = 1
    else:
        input_batch_size = batch_size
    input_channel, input_height, input_width = 3, 544, 960
    input_names = ['inputs']
    output_names = ["pred_logits", "pred_boxes"]

    model = build_model(_test_experiment_spec, export=True)
    model.eval()
    model.cuda()

    dummy_input = torch.ones(input_batch_size, input_channel, input_height, input_width, device='cuda')
    os_handle, tmp_onnx_file = tempfile.mkstemp(suffix=".onnx")
    os.close(os_handle)

    onnx_export = ONNXExporter()
    onnx_export.export_model(model, batch_size, tmp_onnx_file, dummy_input,
                             input_names=input_names, output_names=output_names,
                             do_constant_folding=False)
    onnx_export.check_onnx(tmp_onnx_file)
    onnx_export.onnx_change(tmp_onnx_file)

    assert os.path.exists(tmp_onnx_file), "ONNX file was not generated properly!"
    os.remove(tmp_onnx_file)


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", ["fan_tiny",
                                      "vit_large_nvdinov2"])
@pytest.mark.parametrize("batch_size", [-1])
def test_dino_compare_onnx_output(_test_experiment_spec, backbone, batch_size):
    """Unit test for ONNX export on DINO model. Here pytorch DMHA is used for ONNXRuntime."""
    _test_experiment_spec["model"].backbone = backbone
    _test_experiment_spec["model"].aux_loss = False
    _test_experiment_spec["model"].num_feature_levels = 2
    _test_experiment_spec["model"].return_interm_indices = [1, 2]
    _test_experiment_spec["model"].num_queries = 100

    # Define input and output of ONNX
    if batch_size == -1:
        input_batch_size = 1
    else:
        input_batch_size = batch_size

    if backbone.startswith("vit"):
        # LSJ
        input_channel, input_height, input_width = 3, 224, 224
        _test_experiment_spec["dataset"].augmentation.fixed_random_crop = 224
    else:
        input_channel, input_height, input_width = 3, 544, 960

    input_names = ['inputs']
    output_names = ["pred_logits", "pred_boxes"]

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


    # Seed BEFORE build_model so head-layer random init is deterministic across
    # test orderings — otherwise prior tests in the same process leave different
    # RNG state, which produces slightly different head weights and exposes
    # numerical edge cases in the legacy ONNX exporter.
    torch.manual_seed(47)

    # To run ONNXRuntime, we run models in CPU so that deformable attention does not
    # use custom TRT Plugin
    model = build_model(_test_experiment_spec, export=True)
    model.eval()

    dummy_input = torch.randn(input_batch_size, input_channel, input_height, input_width)
    os_handle, tmp_onnx_file = tempfile.mkstemp(suffix=".onnx")
    os.close(os_handle)

    onnx_export = ONNXExporter()
    onnx_export.export_model(model, batch_size, tmp_onnx_file,
                             dummy_input,
                             opset_version=16,  # Required for gridsample
                             input_names=input_names,
                             output_names=output_names,
                             do_constant_folding=False)
    onnx_export.check_onnx(tmp_onnx_file)
    onnx_export.onnx_change(tmp_onnx_file)

    assert os.path.exists(tmp_onnx_file), "ONNX file was not generated properly!"

    # Load ONNX and ONNXRuntime
    ort.set_seed(47)
    onnx_model = onnx.load(tmp_onnx_file)
    onnx.checker.check_model(onnx_model)
    sess = ort.InferenceSession(
        tmp_onnx_file,
        providers=['CPUExecutionProvider']
    )

    with torch.no_grad():
        torch_output = model(dummy_input)

    onnx_output = sess.run(None, {
        "inputs": dummy_input.numpy()
    })
    if not (list(torch_output["pred_logits"].shape) == list(onnx_output[0].shape)):
        raise ValueError(f"Size Mismatch {list(torch_output['pred_logits'].shape)} vs {list(onnx_output[0].shape)}")
    torch.testing.assert_close(torch_output["pred_logits"].cpu(), torch.from_numpy(onnx_output[0]), rtol=1e-5, atol=1e-4)

    if not (list(torch_output["pred_boxes"].shape) == list(onnx_output[1].shape)):
        raise ValueError(f"Size Mismatch {list(torch_output['pred_boxes'].shape)} vs {list(onnx_output[1].shape)}")
    torch.testing.assert_close(torch_output["pred_boxes"].cpu(), torch.from_numpy(onnx_output[1]), rtol=1e-5, atol=1e-4)
    os.remove(tmp_onnx_file)
