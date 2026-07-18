# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf
from nvidia_tao_pytorch.config.depth_net.default_config import ExperimentConfig, DepthNetDatasetConfig, DepthNetModelConfig
from nvidia_tao_pytorch.cv.depth_net.model.build_pl_model import build_pl_model


@pytest.fixture
def _test_experiment_spec():
    dataset_config = OmegaConf.structured(DepthNetDatasetConfig())
    model_config = OmegaConf.structured(DepthNetModelConfig())
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset = dataset_config
    experiment_config.model = model_config
    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.model
@pytest.mark.parametrize("model_type", [
    "RelativeDepthAnything",
    "MetricDepthAnything",
    "FoundationStereo",
    "FastFoundationStereo",
])
def test_depth_net_model_build(_test_experiment_spec, model_type):
    """Tests if the DepthNetPlModel can be instantiated."""
    _test_experiment_spec.model.model_type = model_type
    try:
        model = build_pl_model(_test_experiment_spec)
        assert model is not None, "Depth Model instantiation failed."
    except Exception as e:
        pytest.fail(f"Depth Model instantiation raised an exception: {e}")
