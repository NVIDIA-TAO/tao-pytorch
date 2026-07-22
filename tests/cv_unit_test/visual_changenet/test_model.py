# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Visual ChangeNet-Segmentation/Classification Model builder Unit Tests
"""
import os
import tempfile

import pytest
import torch
from nvidia_tao_pytorch.config.visual_changenet.default_config import CNDatasetConfig, CNModelConfig, ExperimentConfig
from omegaconf import OmegaConf

from nvidia_tao_pytorch.cv.visual_changenet.classification.models.changenet import ChangeNetClassify, build_model
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models.changenet import ChangeNetSegment
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models.changenet import build_model as build_model_segment


IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
OUTPUT_SHAPE = 224
TEST_TOPOLOGIES = [
    ("fan_tiny_8_p4_hybrid"),
    ("vit_large_nvdinov2"),
    ("c_radio_p3_vit_huge_patch16_224_mlpnorm"),
    ("c_radio_v2_vit_base_patch16_224"),
]
TEST_TOPOLOGIES_CLASSIFICATION_PTM = [
    # classification_pyt, visual_changenet
    ("fan_tiny_8_p4_hybrid", "fan_tiny_8_p4_hybrid"),
    ("vit_large_patch14_dinov2_swiglu", "vit_large_nvdinov2"),
    ("c_radio_p3_vit_huge_patch16_mlpnorm", "c_radio_p3_vit_huge_patch16_224_mlpnorm"),
    ("c_radio_v2_vit_base_patch16", "c_radio_v2_vit_base_patch16_224"),
]

@pytest.fixture
def _test_experiment_spec():
    dataset_config = OmegaConf.structured(CNDatasetConfig())
    model_config = OmegaConf.structured(CNModelConfig())
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset = dataset_config
    experiment_config.model = model_config
    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
@pytest.mark.parametrize("export", [False, True])
@pytest.mark.parametrize("difference_module, num_golden",
                         [('learnable', 1),
                          ('learnable', 4),
                          ('euclidean', 1)])
@pytest.mark.parametrize("task", ['classify'])
def test_changenet_model(_test_experiment_spec, backbone, export, task, difference_module, num_golden):
    if "fan" in backbone and num_golden != 1:
        pytest.skip(f"Invalid combination: {backbone} and num_golden={num_golden}")

    _test_experiment_spec["model"].backbone['type'] = backbone
    _test_experiment_spec.task = task
    _test_experiment_spec["dataset"]['classify']["image_width"] = IMAGE_WIDTH
    _test_experiment_spec["dataset"]['classify']["image_height"] = IMAGE_HEIGHT
    _test_experiment_spec["dataset"]['classify'].num_golden = num_golden
    _test_experiment_spec["model"]['classify'].difference_module = difference_module

    model = build_model(_test_experiment_spec, export)
    assert(isinstance(model, ChangeNetClassify))


@pytest.mark.cv_unit
@pytest.mark.parametrize("classification_backbone, backbone", TEST_TOPOLOGIES_CLASSIFICATION_PTM)
def test_changenet_model_from_classification(_test_experiment_spec, classification_backbone, backbone):
    from nvidia_tao_pytorch.config.classification_pyt.default_config import DatasetConfig, ExperimentConfig, ModelConfig

    from nvidia_tao_pytorch.cv.classification_pyt.model.classifier_pl_model import ClassifierPlModel

    # Create a dummy classification checkpoint.
    tmp_top_obj = tempfile.TemporaryDirectory()
    tmp_top_dir = tmp_top_obj.name
    classes_file = os.path.join(tmp_top_dir, 'test_data.csv')
    pretrained_backbone_path = os.path.join(tmp_top_dir, 'pretrained_backbone.pth')
    with open(classes_file, 'w') as f:
        f.write("class1\nclass2")
    dataset_config = OmegaConf.structured(DatasetConfig())
    model_config = OmegaConf.structured(ModelConfig())
    classification_experiment_config = OmegaConf.structured(ExperimentConfig())
    classification_experiment_config.dataset = dataset_config
    classification_experiment_config.dataset.num_classes = 2
    classification_experiment_config.model = model_config
    classification_experiment_config["model"].backbone['type'] = classification_backbone
    classification_experiment_config.dataset.classes_file = classes_file
    classification_model = ClassifierPlModel(classification_experiment_config)
    torch.save(
        {"state_dict": classification_model.state_dict(), "tao_model": "classification"},
        pretrained_backbone_path,
    )

    _test_experiment_spec["model"].backbone['type'] = backbone
    _test_experiment_spec["model"].backbone['pretrained_backbone_path'] = pretrained_backbone_path
    _test_experiment_spec.task = "classify"
    _test_experiment_spec["dataset"]['classify']["image_width"] = IMAGE_WIDTH
    _test_experiment_spec["dataset"]['classify']["image_height"] = IMAGE_HEIGHT

    model = build_model(_test_experiment_spec)
    assert(isinstance(model, ChangeNetClassify))


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
@pytest.mark.parametrize("export", [False, True])
@pytest.mark.parametrize("task", ['segment'])
def test_changenet_model_segment(_test_experiment_spec, backbone, export, task):
    _test_experiment_spec["model"].backbone['type'] = backbone
    _test_experiment_spec.task = task
    _test_experiment_spec["dataset"]['segment']["img_size"] = OUTPUT_SHAPE

    model = build_model_segment(_test_experiment_spec, export)
    assert(isinstance(model, ChangeNetSegment))


@pytest.mark.cv_unit
@pytest.mark.parametrize("classification_backbone, backbone", TEST_TOPOLOGIES_CLASSIFICATION_PTM)
def test_changenet_model_segment_from_classification(_test_experiment_spec, classification_backbone, backbone):
    from nvidia_tao_pytorch.config.classification_pyt.default_config import DatasetConfig, ExperimentConfig, ModelConfig

    from nvidia_tao_pytorch.cv.classification_pyt.model.classifier_pl_model import ClassifierPlModel

    # Create a dummy classification checkpoint.
    tmp_top_obj = tempfile.TemporaryDirectory()
    tmp_top_dir = tmp_top_obj.name
    classes_file = os.path.join(tmp_top_dir, 'test_data.csv')
    pretrained_backbone_path = os.path.join(tmp_top_dir, 'pretrained_backbone.pth')
    with open(classes_file, 'w') as f:
        f.write("class1\nclass2")
    dataset_config = OmegaConf.structured(DatasetConfig())
    model_config = OmegaConf.structured(ModelConfig())
    classification_experiment_config = OmegaConf.structured(ExperimentConfig())
    classification_experiment_config.dataset = dataset_config
    classification_experiment_config.dataset.num_classes = 2
    classification_experiment_config.model = model_config
    classification_experiment_config["model"].backbone['type'] = classification_backbone
    classification_experiment_config.dataset.classes_file = classes_file
    classification_model = ClassifierPlModel(classification_experiment_config)
    torch.save(
        {"state_dict": classification_model.state_dict(), "tao_model": "classification"},
        pretrained_backbone_path,
    )

    _test_experiment_spec["model"].backbone['type'] = backbone
    _test_experiment_spec["model"].backbone['pretrained_backbone_path'] = pretrained_backbone_path
    _test_experiment_spec.task = "segment"
    _test_experiment_spec["dataset"]['segment']["img_size"] = OUTPUT_SHAPE

    model = build_model_segment(_test_experiment_spec)
    assert(isinstance(model, ChangeNetSegment))
