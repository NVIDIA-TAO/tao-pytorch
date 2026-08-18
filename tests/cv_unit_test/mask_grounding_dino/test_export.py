# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf

import os
import tempfile
import onnx
import onnxruntime as ort
import numpy as np
import torch
from copy import deepcopy

from nvidia_tao_pytorch.config.mask_grounding_dino.default_config import MaskGDINOModelConfig, MaskGDINODatasetConfig, ExperimentConfig
from nvidia_tao_pytorch.cv.mask_grounding_dino.model.build_nn_model import build_model
from nvidia_tao_pytorch.cv.mask_grounding_dino.utils.onnx_export import ONNXExporter


@pytest.fixture
def _test_experiment_spec():
    dataset_config = MaskGDINODatasetConfig()
    model_config = MaskGDINOModelConfig()
    experiment_config = ExperimentConfig()
    experiment_config.dataset = dataset_config
    experiment_config.model = model_config
    experiment_config = OmegaConf.structured(experiment_config)

    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", ["swin_tiny_224_1k"])
@pytest.mark.parametrize("batch_size", [-1])
@pytest.mark.parametrize("num_region_queries", [0, 200])
def test_mask_grounding_dino_onnx_export(_test_experiment_spec, backbone, batch_size, num_region_queries):
    """Unit test for ONNX export on Mask Grounding DINO model. Here, custom DMHA is used."""
    _test_experiment_spec.model.backbone = backbone
    _test_experiment_spec.model.num_region_queries = num_region_queries
    # Define input and output of ONNX
    if batch_size == -1:
        input_batch_size = 1
    else:
        input_batch_size = batch_size
    input_channel, input_height, input_width = 3, 544, 960

    input_names = ["inputs", "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"]
    output_names = ["pred_logits", "pred_boxes", "pred_masks"]
    if num_region_queries > 0:
        output_names.extend(['no_targets', 'union_mask_logits'])

    model = build_model(_test_experiment_spec, export=True)
    model.eval()
    model.cuda()

    device = "cuda"
    caption = "the running dog ."
    input_ids = model.model.tokenizer([caption], return_tensors="pt")["input_ids"].to(device)
    position_ids = torch.tensor([[0, 0, 1, 2, 3, 0]]).to(device)
    token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0]]).to(device)
    attention_mask = torch.tensor([[True, True, True, True, True, True]]).to(device)
    text_token_mask = torch.tensor([[[True, False, False, False, False, False],
                                   [False,  True,  True,  True,  True, False],
                                   [False,  True,  True,  True,  True, False],
                                   [False,  True,  True,  True,  True, False],
                                   [False,  True,  True,  True,  True, False],
                                   [False, False, False, False, False,  True]]]).to(device)

    dummy_input = torch.randn(input_batch_size, input_channel, input_height, input_width).to(device)
    args = (dummy_input, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask)

    os_handle, tmp_onnx_file = tempfile.mkstemp(suffix=".onnx")
    os.close(os_handle)

    onnx_export = ONNXExporter()
    onnx_export.export_model(model, batch_size, tmp_onnx_file, args,
                             input_names=input_names, output_names=output_names,
                             do_constant_folding=False)
    onnx_export.check_onnx(tmp_onnx_file)
    onnx_export.onnx_change(tmp_onnx_file)

    assert os.path.exists(tmp_onnx_file), "ONNX file was not generated properly!"
    os.remove(tmp_onnx_file)


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", ["swin_tiny_224_1k"])
@pytest.mark.parametrize("batch_size", [-1])
@pytest.mark.parametrize("num_region_queries", [0, 200])
def test_mask_grounding_dino_compare_onnx_output(_test_experiment_spec, backbone, batch_size, num_region_queries):
    """Unit test for ONNX export on Mask Grounding DINO model. Here pytorch DMHA is used for ONNXRuntime."""
    _test_experiment_spec.model.backbone = backbone
    _test_experiment_spec.model.aux_loss = False
    _test_experiment_spec.model.num_feature_levels = 4
    _test_experiment_spec.model.return_interm_indices = [1, 2, 3, 4]
    _test_experiment_spec.model.num_queries = 100
    _test_experiment_spec.model.num_region_queries = num_region_queries
    # Define input and output of ONNX
    if batch_size == -1:
        input_batch_size = 1
    else:
        input_batch_size = batch_size

    input_channel, input_height, input_width = 3, 544, 960

    input_names = ["inputs", "input_ids", "attention_mask", "position_ids", "token_type_ids", "text_token_mask"]
    output_names = ["pred_logits", "pred_boxes", "pred_masks"]
    if num_region_queries > 0:
        output_names.extend(['no_targets', 'union_mask_logits'])

    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    # To run ONNXRuntime, we run models in CPU so that deformable attention does not
    # use custom TRT Plugin
    
    model = build_model(_test_experiment_spec, export=True)
    model.eval()
    device = "cpu"

    torch.manual_seed(47)
    caption = "the running dog ."
    input_ids = model.model.tokenizer([caption], return_tensors="pt")["input_ids"].to(device)
    position_ids = torch.tensor([[0, 0, 1, 2, 3, 0]]).to(device)
    token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0]]).to(device)
    attention_mask = torch.tensor([[True, True, True, True, True, True]]).to(device)
    text_token_mask = torch.tensor([[[True, False, False, False, False, False],
                                   [False,  True,  True,  True,  True, False],
                                   [False,  True,  True,  True,  True, False],
                                   [False,  True,  True,  True,  True, False],
                                   [False,  True,  True,  True,  True, False],
                                   [False, False, False, False, False,  True]]]).to(device)

    dummy_input = torch.randn(input_batch_size, input_channel, input_height, input_width).to(device)
    args = (dummy_input, input_ids, attention_mask, position_ids, token_type_ids, text_token_mask)

    os_handle, tmp_onnx_file = tempfile.mkstemp(suffix=".onnx")
    os.close(os_handle)

    onnx_export = ONNXExporter()
    onnx_export.export_model(model, batch_size, tmp_onnx_file,
                             args,
                             opset_version=17,
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
        torch_output = model(samples=dummy_input,
                             input_ids=input_ids,
                             attention_mask=attention_mask,
                             position_ids=position_ids,
                             token_type_ids=token_type_ids,
                             text_self_attention_masks=text_token_mask)

    onnx_input = {}
    for i, a in zip(input_names, args):
        onnx_input[i] = a.numpy()

    onnx_output = sess.run(None, onnx_input)
    if not (list(torch_output["pred_logits"].shape) == list(onnx_output[0].shape)):
        raise ValueError(f"Size Mismatch {list(torch_output['pred_logits'].shape)} vs {list(onnx_output[0].shape)}")
    torch.testing.assert_close(torch_output["pred_logits"].cpu(), torch.from_numpy(onnx_output[0]), rtol=5e-3, atol=2e-4)

    if not (list(torch_output["pred_boxes"].shape) == list(onnx_output[1].shape)):
        raise ValueError(f"Size Mismatch {list(torch_output['pred_boxes'].shape)} vs {list(onnx_output[1].shape)}")
    torch.testing.assert_close(torch_output["pred_boxes"].cpu(), torch.from_numpy(onnx_output[1]), rtol=5e-3, atol=2e-4)
    os.remove(tmp_onnx_file)
