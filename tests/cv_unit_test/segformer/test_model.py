# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SegFormer_pl Model builder Unit Tests
"""
import pytest
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.segformer.default_config import SFModelConfig, SFDatasetConfig, ExperimentConfig
from nvidia_tao_pytorch.cv.segformer.model.segformer_pl_model import build_model
from nvidia_tao_pytorch.cv.segformer.model.segformer import SegFormer

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
OUTPUT_SHAPE = 224
TEST_TOPOLOGIES = [
    # ConvNeXtV2.
    ("mit_b0"),
    # DINOV2.
    ("vit_large_nvdinov2"),
    # FAN.
    ("fan_tiny_8_p4_hybrid"),
    # OpenCLIP.
    ("vit_base_nvclip_16_siglip"),
    ("vit_huge_nvclip_14_siglip"),
    # RADIO.
    ("c_radio_v2_vit_base_patch16_224"),
]


@pytest.fixture
def _test_experiment_spec():
    dataset_config = OmegaConf.structured(SFDatasetConfig())
    model_config = OmegaConf.structured(SFModelConfig())
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset = dataset_config
    experiment_config.model = model_config
    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
@pytest.mark.parametrize("export", [False, True])
def test_segformer_model(_test_experiment_spec, backbone, export):
    _test_experiment_spec["model"].backbone['type'] = backbone
    _test_experiment_spec["dataset"]['segment']["img_size"] = OUTPUT_SHAPE

    model = build_model(_test_experiment_spec, export)
    assert(isinstance(model, SegFormer))
