# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf
from nvidia_tao_pytorch.config.dino.dataset import DINODatasetConfig
from nvidia_tao_pytorch.config.dino.default_config import ExperimentConfig
from nvidia_tao_pytorch.config.dino.model import DINOModelConfig
from nvidia_tao_pytorch.cv.dino.model.build_nn_model import build_model


@pytest.fixture
def _test_experiment_spec():
    dataset_config = OmegaConf.structured(DINODatasetConfig())
    model_config = OmegaConf.structured(DINOModelConfig())
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset = dataset_config
    experiment_config.model = model_config
    yield experiment_config

@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", ["resnet_50", "fan_small", "vit_large_nvdinov2"])
@pytest.mark.parametrize("num_feature_levels", [2, 4])
@pytest.mark.parametrize("return_interm_indices", [[1, 2], [1, 2, 3, 4]])
@pytest.mark.parametrize("num_queries", [300, 900])
@pytest.mark.parametrize("export", [False, True])
def test_dino_model(_test_experiment_spec, backbone, num_feature_levels, return_interm_indices, num_queries, export):
    _test_experiment_spec["model"].backbone = backbone
    _test_experiment_spec["model"].num_feature_levels = num_feature_levels
    if num_feature_levels != len(return_interm_indices):
        pytest.skip("num_feature_levels and length of return_interm_indices must match.")
    _test_experiment_spec["model"].return_interm_indices = return_interm_indices
    _test_experiment_spec["model"].num_queries = num_queries

    if backbone.startswith("vit"):
        _test_experiment_spec["dataset"].augmentation.fixed_random_crop = 224

    # aux_loss must be set to False during export
    if export:
        _test_experiment_spec["model"].aux_loss = False

    build_model(_test_experiment_spec, export)
