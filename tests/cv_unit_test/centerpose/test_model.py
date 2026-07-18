# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.centerpose.default_config import ExperimentConfig
from nvidia_tao_pytorch.config.centerpose.dataset import CenterPoseDatasetConfig
from nvidia_tao_pytorch.config.centerpose.model import CenterPoseModelConfig

from nvidia_tao_pytorch.cv.centerpose.model.centerpose import create_model


@pytest.fixture
def _test_experiment_spec():
    dataset_config = OmegaConf.structured(CenterPoseDatasetConfig())
    model_config = OmegaConf.structured(CenterPoseModelConfig())
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset = dataset_config
    experiment_config.model = model_config
    yield experiment_config

@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone, use_pretrained", 
                         [("DLA34", False),
                          # TODO @vpraveen/@jianhey: enable this test once torchhub
                          # connection to torch hub is restored.
                          # ("DLA34", True),  
                          ("fan_small", False),
                          ("fan_base", False),
                          ("fan_large", False)])
@pytest.mark.parametrize("down_ratio", [2, 4, 8, 16])
def test_centerpose_model(_test_experiment_spec, backbone, use_pretrained, down_ratio):
    _test_experiment_spec["model"].backbone.model_type = backbone
    _test_experiment_spec["model"].use_pretrained = use_pretrained
    _test_experiment_spec["model"].down_ratio = down_ratio

    create_model(_test_experiment_spec["model"])
