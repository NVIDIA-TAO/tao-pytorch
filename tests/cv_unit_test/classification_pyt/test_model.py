# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Classification_pl Model builder Unit Tests
"""
import os
import tempfile

import pytest
import torch
from nvidia_tao_pytorch.config.classification_pyt.default_config import DatasetConfig, ExperimentConfig, ModelConfig
from omegaconf import OmegaConf

from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase
from nvidia_tao_pytorch.cv.classification_pyt.model.classifier import build_model


IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
OUTPUT_SHAPE = 224

TEST_TOPOLOGIES_CHANGENET_CLASSIFY_PTM = [
    # difference_module, visual_changenet, classification_pyt
    ("euclidean", "fan_tiny_8_p4_hybrid", "fan_tiny_8_p4_hybrid"),
    ("euclidean", "vit_large_nvdinov2", "vit_large_patch14_dinov2_swiglu"),
    ("learnable", "vit_large_nvdinov2", "vit_large_patch14_dinov2_swiglu"),
    ("euclidean", "c_radio_v2_vit_base_patch16_224", "c_radio_v2_vit_base_patch16"),
    ("learnable", "c_radio_v2_vit_base_patch16_224", "c_radio_v2_vit_base_patch16"),
]
TEST_TOPOLOGIES_CHANGENET_SEGMENT_PTM = [
    # visual_changenet, classification_pyt
    ("fan_tiny_8_p4_hybrid", "fan_tiny_8_p4_hybrid"),
    ("vit_large_nvdinov2", "vit_large_patch14_dinov2_swiglu"),
    ("c_radio_v2_vit_base_patch16_224", "c_radio_v2_vit_base_patch16"),
]


@pytest.fixture
def _test_experiment_spec():
    dataset_config = OmegaConf.structured(DatasetConfig())
    model_config = OmegaConf.structured(ModelConfig())
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset = dataset_config
    experiment_config.dataset.num_classes = 0
    experiment_config.model = model_config
    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize(
    "backbone",
    [
        # ConvNeXtV2.
        ("convnextv2_atto"),
        # DINOV2.
        ("vit_large_patch14_dinov2_swiglu"),
        ("vit_giant_patch14_reg4_dinov2_swiglu"),
        # FAN.
        ("fan_small_12_p16_224"),
        ("fan_small_12_p4_hybrid"),
        ("fan_small_12_p16_224_se_attn"),
        # FasterViT.
        ("faster_vit_1_224"),
        # GCViT.
        ("gc_vit_xxtiny"),
        # OpenCLIP.
        ("vit_l_14_siglip_clipa_336"),
        # RADIO.
        ("c_radio_p3_vit_huge_patch16_mlpnorm"),
        ("c_radio_v2_vit_base_patch16"),
        # SwinTransformer.
        ("swin_tiny_patch4_window7_224"),
        ("swin_small_patch4_window7_224"),
        # EdgeNext,
        ("edgenext_small"),
        ("edgenext_base"),
    ],
)
# for classification_pyt, export or not is not affecting anything
@pytest.mark.parametrize("export", [False, True])
def test_classifier_model(_test_experiment_spec, backbone, export):
    _test_experiment_spec["model"].backbone['type'] = backbone
    _test_experiment_spec["dataset"]["img_size"] = OUTPUT_SHAPE

    model = build_model(_test_experiment_spec, export)
    assert(isinstance(model, BackboneBase))


@pytest.mark.cv_unit
@pytest.mark.parametrize(
    "difference_module, visual_changenet_backbone, backbone",
    TEST_TOPOLOGIES_CHANGENET_CLASSIFY_PTM,
)
def test_classifier_model_from_changenet_classify(
    _test_experiment_spec, difference_module, visual_changenet_backbone, backbone
):
    from nvidia_tao_pytorch.config.visual_changenet.default_config import CNDatasetConfig, CNModelConfig
    from nvidia_tao_pytorch.config.visual_changenet.default_config import ExperimentConfig as CNExperimentConfig

    from nvidia_tao_pytorch.cv.visual_changenet.classification.models.cn_pl_model import ChangeNetPlModel

    # Create a dummy visual changenet checkpoint.
    tmp_top_obj = tempfile.TemporaryDirectory()
    tmp_top_dir = tmp_top_obj.name
    pretrained_backbone_path = os.path.join(tmp_top_dir, 'pretrained_backbone.pth')
    dataset_config = OmegaConf.structured(CNDatasetConfig())
    model_config = OmegaConf.structured(CNModelConfig())
    visual_changenet_experiment_config = OmegaConf.structured(CNExperimentConfig())
    visual_changenet_experiment_config.dataset = dataset_config
    visual_changenet_experiment_config["dataset"]['classify']["image_width"] = IMAGE_WIDTH
    visual_changenet_experiment_config["dataset"]['classify']["image_height"] = IMAGE_HEIGHT
    visual_changenet_experiment_config["dataset"]['classify'].num_golden = 1
    visual_changenet_experiment_config["train"]['classify'].loss = (
        "ce" if difference_module == "learnable" else "contrastive"
    )
    visual_changenet_experiment_config.model = model_config
    visual_changenet_experiment_config["model"].backbone['type'] = visual_changenet_backbone
    visual_changenet_experiment_config["model"]['classify'].difference_module = difference_module
    visual_changenet_experiment_config.task = "classify"
    visual_changenet_model = ChangeNetPlModel(visual_changenet_experiment_config, dm=None)
    # Visual ChangeNet has 3 types of model archs.
    tao_model_type = None
    if "radio" in visual_changenet_backbone and difference_module == "learnable":
        tao_model_type = "radio_learnable"
    torch.save(
        {
            "state_dict": visual_changenet_model.state_dict(),
            "tao_model": "visual_changenet_classify",
            "tao_model_type": tao_model_type,
        },
        pretrained_backbone_path,
    )

    _test_experiment_spec["model"].backbone['type'] = backbone
    _test_experiment_spec["model"].backbone['pretrained_backbone_path'] = pretrained_backbone_path
    _test_experiment_spec["dataset"]["img_size"] = OUTPUT_SHAPE

    model = build_model(_test_experiment_spec)
    assert(isinstance(model, BackboneBase))


@pytest.mark.cv_unit
@pytest.mark.parametrize(
    "visual_changenet_backbone, backbone",
    TEST_TOPOLOGIES_CHANGENET_SEGMENT_PTM,
)
def test_classifier_model_from_changenet_segment(
    _test_experiment_spec, visual_changenet_backbone, backbone
):
    from nvidia_tao_pytorch.config.visual_changenet.default_config import CNDatasetConfig, CNModelConfig
    from nvidia_tao_pytorch.config.visual_changenet.default_config import ExperimentConfig as CNExperimentConfig

    from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models.cn_pl_model import ChangeNetPlModel

    # Create a dummy visual changenet checkpoint.
    tmp_top_obj = tempfile.TemporaryDirectory()
    tmp_top_dir = tmp_top_obj.name
    pretrained_backbone_path = os.path.join(tmp_top_dir, 'pretrained_backbone.pth')
    dataset_config = OmegaConf.structured(CNDatasetConfig())
    model_config = OmegaConf.structured(CNModelConfig())
    visual_changenet_experiment_config = OmegaConf.structured(CNExperimentConfig())
    visual_changenet_experiment_config.dataset = dataset_config
    visual_changenet_experiment_config["dataset"]['segment']["img_size"] = IMAGE_WIDTH
    visual_changenet_experiment_config.model = model_config
    visual_changenet_experiment_config["model"].backbone['type'] = visual_changenet_backbone
    visual_changenet_experiment_config.task = "segment"
    visual_changenet_model = ChangeNetPlModel(visual_changenet_experiment_config)
    # Visual ChangeNet has 3 types of model archs.
    tao_model_type = None
    if "radio" in visual_changenet_backbone:
        tao_model_type = "radio"
    torch.save(
        {
            "state_dict": visual_changenet_model.state_dict(),
            "tao_model": "visual_changenet_segment",
            "tao_model_type": tao_model_type,
        },
        pretrained_backbone_path,
    )

    _test_experiment_spec["model"].backbone['type'] = backbone
    _test_experiment_spec["model"].backbone['pretrained_backbone_path'] = pretrained_backbone_path
    _test_experiment_spec["dataset"]["img_size"] = OUTPUT_SHAPE

    model = build_model(_test_experiment_spec)
    assert(isinstance(model, BackboneBase))
