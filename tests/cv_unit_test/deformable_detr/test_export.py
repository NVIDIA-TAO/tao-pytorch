# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf

import os
import tempfile
import onnx
import onnxruntime as ort
import torch

from nvidia_tao_pytorch.config.deformable_detr.default_config import ExperimentConfig
from nvidia_tao_pytorch.config.deformable_detr.model import DDModelConfig
from nvidia_tao_pytorch.config.deformable_detr.dataset import DDDatasetConfig
from nvidia_tao_pytorch.core.modules.activation.activation import MultiheadAttention
from nvidia_tao_pytorch.cv.deformable_detr.model.build_nn_model import build_model
from nvidia_tao_pytorch.cv.deformable_detr.utils.onnx_export import ONNXExporter


@pytest.fixture
def _test_experiment_spec():
    dataset_config = OmegaConf.structured(DDDatasetConfig())
    model_config = OmegaConf.structured(DDModelConfig())
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset = dataset_config
    experiment_config.model = model_config
    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("batch_size, num_queries, device",
                         [(-1, 300, "cpu"),
                          (-1, 900, "cpu"),
                          (1, 300, "cpu"),
                          (1, 900, "cpu"),
                          (8, 300, "cpu"),
                          (8, 900, "cpu"),
                         ])
def test_multihead_attn_onnx(batch_size, num_queries, device):
    """Unit test for ONNX export on MHA PyTorch module"""
    if device == "cuda":
        # Disable TF32 for Ampere architectures as it may cause discrepancy between ONNX vs PyT
        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
        torch.backends.cuda.matmul.allow_tf32 = False

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = False

    embed_dim = 256
    num_heads = 8
    is_dynamic = False

    if batch_size == -1:
        is_dynamic = True
        # For dynamic, override batch size to 1
        batch_size = 1

    query = torch.rand(batch_size, num_queries, embed_dim).to(device)  # [N, Q, D]
    key = query
    value = torch.rand(batch_size, num_queries, embed_dim).to(device)  # [N, Q, D]

    model = MultiheadAttention(embed_dim, num_heads)
    model.to(device)
    model.eval()
    os_handle, tmp_onnx_file = tempfile.mkstemp(suffix=".onnx")
    os.close(os_handle)

    input_names=["query", "key", "value"]
    output_names=["output", "attn"]

    if is_dynamic:
        dynamic_axes = {}
        for i_name in input_names:
            dynamic_axes[i_name] = {0: "batch"}
        for o_name in output_names:
            dynamic_axes[o_name] = {0: "batch"}
    else:
        dynamic_axes = None

    # Export to ONNX
    dummy_input = (query.transpose(0, 1), key.transpose(
            0, 1), value.transpose(0, 1))
    torch.onnx.export(model, dummy_input, tmp_onnx_file,
            input_names=input_names, output_names=output_names, export_params=True,
            training=torch.onnx.TrainingMode.EVAL, opset_version=16, do_constant_folding= False,
            verbose = False, dynamic_axes = dynamic_axes,
            dynamo=False)

    # Load ONNX and ONNXRuntime
    onnx_model = onnx.load(tmp_onnx_file)
    onnx.checker.check_model(onnx_model)
    sess = ort.InferenceSession(
        tmp_onnx_file,
        providers=['CPUExecutionProvider']
    )
    input_name = sess.get_inputs()

    with torch.no_grad():
        torch_output, torch_attn = model(query.transpose(0, 1), key.transpose(
            0, 1), value.transpose(0, 1))

    onnx_output = sess.run(None, {
        input_name[0].name: query.transpose(0, 1).cpu().numpy(),
        input_name[1].name: key.transpose(0, 1).cpu().numpy(),
        input_name[2].name: value.transpose(0, 1).cpu().numpy(),
    })
    if not (list(torch_output.shape) == list(onnx_output[0].shape)):
        raise ValueError(f"Size Mismatch {list(torch_output.shape)} vs {list(onnx_output[0].shape)}")
    torch.testing.assert_close(torch_output.cpu(), torch.from_numpy(onnx_output[0]))

    if not (list(torch_attn.shape) == list(onnx_output[1].shape)):
        raise ValueError(f"Size Mismatch {list(torch_attn.shape)} vs {list(onnx_output[1].shape)}")
    torch.testing.assert_close(torch_attn.cpu(), torch.from_numpy(onnx_output[1]))


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", ["resnet_50"])
@pytest.mark.parametrize("batch_size", [-1, 4])
def test_ddetr_onnx_export(_test_experiment_spec, backbone, batch_size):
    """Unit test for ONNX export on D-DETR model. Here, custom DMHA is used."""
    _test_experiment_spec["model"].backbone = backbone
    _test_experiment_spec["model"].aux_loss = False
    _test_experiment_spec["model"].num_feature_levels = 2
    _test_experiment_spec["model"].return_interm_indices = [1, 2]
    _test_experiment_spec["model"].num_queries = 100

    # Define input and output of ONNX
    if batch_size == -1:
        input_batch_size = 1
    else:
        input_batch_size = 4
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
    onnx_export.export_model(model, batch_size, tmp_onnx_file, dummy_input, input_names=input_names, output_names=output_names, do_constant_folding=False)
    onnx_export.check_onnx(tmp_onnx_file)
    onnx_export.onnx_change(tmp_onnx_file)

    assert os.path.exists(tmp_onnx_file), "ONNX file was not generated properly!"
    os.remove(tmp_onnx_file)


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", ["resnet_50", "gc_vit_xxtiny"])
@pytest.mark.parametrize("batch_size", [-1, 1])
def test_ddetr_compare_onnx_output(_test_experiment_spec, backbone, batch_size):
    """Unit test for ONNX export on D-DETR model. Here pytorch DMHA is used for ONNXRuntime."""
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

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    # To run ONNXRuntime, we run models in CPU so that deformable attention does not
    # use custom TRT Plugin
    model = build_model(_test_experiment_spec, export=True)
    model.eval()

    torch.manual_seed(47)
    dummy_input = torch.randn(input_batch_size, input_channel, input_height, input_width)
    os_handle, tmp_onnx_file = tempfile.mkstemp(suffix=".onnx")
    os.close(os_handle)

    onnx_export = ONNXExporter()
    onnx_export.export_model(model, batch_size, tmp_onnx_file,
                             dummy_input,
                             opset_version=16,  # Required for gridsample
                             input_names=input_names,
                             output_names=output_names, do_constant_folding=False)
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
