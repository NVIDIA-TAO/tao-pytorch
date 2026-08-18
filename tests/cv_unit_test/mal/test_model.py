# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.mal.default_config import (
    ExperimentConfig,
    MALModelConfig
)
from nvidia_tao_pytorch.cv.mal.models.vit_builder import build_model


@pytest.fixture
def _test_cfg():
    model_config = OmegaConf.structured(MALModelConfig())
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.model = model_config
    yield cfg


@pytest.mark.cv_unit
@pytest.mark.parametrize("arch",
                         ["vit-mae-base/16",
                          "vit-mae-large/16",
                          "fan_tiny_12_p16_224",
                          "fan_small_12_p16_224",
                          "fan_base_18_p16_224",
                          "fan_large_24_p16_224",
                          "fan_tiny_8_p4_hybrid",
                          "fan_small_12_p4_hybrid",
                          "fan_base_16_p4_hybrid",
                          "fan_large_16_p4_hybrid"])
def test_mal_backbone(_test_cfg, arch):
    _test_cfg.model.arch = arch
    _test_cfg.model.frozen_stages = [0, -1]
    build_model(_test_cfg)
