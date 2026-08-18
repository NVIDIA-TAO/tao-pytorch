# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Unit test for NVPanoptix3D 3D stage model. """

import pytest
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.nvpanoptix3d.model import NVPanoptix3DModelConfig
from nvidia_tao_pytorch.config.nvpanoptix3d.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.model_3d import NVPanoptix3DModel


@pytest.fixture
def _test_experiment_spec():
    """Test experiment spec."""
    model_config = OmegaConf.structured(NVPanoptix3DModelConfig())
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.model = model_config
    yield experiment_config


@pytest.mark.cv_unit
def test_3d_model(_test_experiment_spec):
    """Test 3D model."""
    model = NVPanoptix3DModel(_test_experiment_spec)
    assert model is not None, "Model instantiation failed."
