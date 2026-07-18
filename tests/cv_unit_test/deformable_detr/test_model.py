# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.deformable_detr.default_config import ExperimentConfig
from nvidia_tao_pytorch.config.deformable_detr.model import DDModelConfig
from nvidia_tao_pytorch.config.deformable_detr.dataset import DDDatasetConfig
from nvidia_tao_pytorch.cv.deformable_detr.model.build_nn_model import build_model


@pytest.fixture
def _test_experiment_spec():
    dataset_config = OmegaConf.structured(DDDatasetConfig())
    model_config = OmegaConf.structured(DDModelConfig())
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset = dataset_config
    experiment_config.model = model_config
    yield experiment_config

@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", ["resnet_50", "gc_vit_tiny"])
@pytest.mark.parametrize("num_feature_levels", [2, 4])
@pytest.mark.parametrize("return_interm_indices", [[1, 2], [1, 2, 3, 4]])
@pytest.mark.parametrize("with_box_refine", [False, True])
@pytest.mark.parametrize("aux_loss", [False, True])
@pytest.mark.parametrize("export", [False, True])
def test_ddetr_model(_test_experiment_spec, backbone, num_feature_levels, return_interm_indices, with_box_refine, aux_loss, export):
    _test_experiment_spec["model"].backbone = backbone
    _test_experiment_spec["model"].num_feature_levels = num_feature_levels
    if num_feature_levels != len(return_interm_indices):
        pytest.skip("num_feature_levels and length of return_interm_indices must match.")
    _test_experiment_spec["model"].return_interm_indices = return_interm_indices
    _test_experiment_spec["model"].with_box_refine = with_box_refine
    if export and aux_loss:
        pytest.skip("aux_loss must be set to False during export")
    _test_experiment_spec["model"].aux_loss = aux_loss

    build_model(_test_experiment_spec, export)
