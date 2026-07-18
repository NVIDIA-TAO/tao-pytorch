# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf
from nvidia_tao_pytorch.config.rtdetr.default_config import RTModelConfig, RTDatasetConfig, ExperimentConfig
from nvidia_tao_pytorch.cv.rtdetr.model.build_nn_model import build_model


@pytest.fixture
def _test_experiment_spec():
    dataset_config = OmegaConf.structured(RTDatasetConfig())
    model_config = OmegaConf.structured(RTModelConfig())
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset = dataset_config
    experiment_config.model = model_config
    yield experiment_config

@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", ["resnet_50", "efficientvit_b0", "efficientvit_l0", "fan_tiny_8_p4_hybrid", "convnext_tiny"])
@pytest.mark.parametrize("feats", [
                                    (
                                        [256, 256], 
                                        [8, 16],
                                        2,
                                        [2, 3],
                                        [1]
                                    ),
                                    (
                                        [256, 256, 256],
                                        [8, 16, 32],
                                        3,
                                        [1, 2, 3],
                                        [2]
                                    )
                                  ])
@pytest.mark.parametrize("aux_loss", [False, True])
@pytest.mark.parametrize("export", [False, True])
def test_rtdetr_model(_test_experiment_spec, backbone, feats, aux_loss, export):
    print(feats)
    _test_experiment_spec["model"].backbone = backbone
    _test_experiment_spec["model"].num_feature_levels = feats[2]
    _test_experiment_spec["model"].return_interm_indices = list(feats[3])
    _test_experiment_spec["model"].feat_strides = list(feats[1])
    _test_experiment_spec["model"].feat_channels = list(feats[0])
    _test_experiment_spec["model"].use_encoder_idx = list(feats[4])

    if backbone.startswith("convnext") and len(feats[1]) < 3:
        pytest.skip("Convnext is exception for now")
    if export and aux_loss:
        pytest.skip("aux_loss must be set to False during export")
    _test_experiment_spec["model"].aux_loss = aux_loss

    build_model(_test_experiment_spec, export)
