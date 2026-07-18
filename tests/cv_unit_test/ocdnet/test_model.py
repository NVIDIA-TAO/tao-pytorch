# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from omegaconf import OmegaConf
from nvidia_tao_pytorch.config.ocdnet.default_config import OCDNetModelConfig
from nvidia_tao_pytorch.cv.ocdnet.model.model import Model


TEST_CHANNEL = 3
TEST_HEIGHT = 640
TEST_WIDTH = 640


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


@pytest.fixture
def _test_tensor():
    torch.manual_seed(47)
    tensor = torch.randn(1, TEST_CHANNEL, TEST_HEIGHT, TEST_WIDTH)
    
    yield tensor


@pytest.mark.cv_unit
def test_dcnresnet_model(_test_dcnresnet_model_spec, _test_tensor):
    model = Model(_test_dcnresnet_model_spec)

    input_tensor = _test_tensor.cuda()
    model.train().cuda()
    with torch.no_grad():
        pred = model(input_tensor)
    
    model.eval()
    with torch.no_grad():
        pred = model(input_tensor)


@pytest.mark.cv_unit
def test_fan_model(_test_fan_model_spec, _test_tensor):
    model = Model(_test_fan_model_spec)

    input_tensor = _test_tensor.cuda()
    model.train().cuda()
    with torch.no_grad():
        pred = model(input_tensor)
    
    model.eval()
    with torch.no_grad():
        pred = model(input_tensor)
