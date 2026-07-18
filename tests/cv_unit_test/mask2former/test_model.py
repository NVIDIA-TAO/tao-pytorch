# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.mask2former.model import Mask2FormerModelConfig, Swin, Backbone
from nvidia_tao_pytorch.config.mask2former.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.mask2former.model.mask2former import MaskFormerModel


@pytest.fixture
def _test_experiment_spec():
    swin_config = OmegaConf.structured(Swin())
    bb_config = OmegaConf.structured(Backbone())
    model_config = OmegaConf.structured(Mask2FormerModelConfig())
    model_config.backbone.swin = swin_config
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.model = model_config
    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", ["swin"])
@pytest.mark.parametrize("name", ['tiny', 'large'])
@pytest.mark.parametrize("export", [False, True])
def test_mask2former_model(_test_experiment_spec, backbone, name, export):
    _test_experiment_spec["model"].backbone.type = backbone
    _test_experiment_spec["model"].backbone.swin.type = name

    model = MaskFormerModel(_test_experiment_spec)
