# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple test cases to test config load and microservices jsonschema conversion."""

import os
import pytest
import json

from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.classification_pyt.default_config import (
    DatasetConfig,
    DataPathFormat,
    ModelConfig,
    OptimConfig,
    BackboneConfig,
    HeadConfig,
    RandomFlip,
    RandomRotation,
    RandomColor,
    RandomCropWithScale,
    AugmentationConfig,
    TrainExpConfig,
    ExperimentConfig
)
from nvidia_tao_core.api_utils.dataclass2json_converter import create_json_schema, dataclass_to_json

sample_dataset_config = """
dataset: "CLDataset"
root_dir: /tao-pt/mount/data/imagenet2012
batch_size: 128
workers: 8
num_classes: 1000
img_size: 224
augmentation:
    random_flip:
        vflip_probability: 0.5
        hflip_probability: 0.5
        enable: True
    random_rotate:
        rotate_probability: 0.5
        angle_list: [90, 180, 270]
        enable: True
    random_color:
        brightness: 0.3
        contrast: 0.3
        saturation: 0.3
        hue: 0.3
        enable: True
    with_scale_random_crop:
        enable: True
    with_random_crop: True
    with_random_blur: False
train_dataset:
    images_dir: /tao-pt/mount/data/imagenet2012/imagenet/train
val_dataset:
    images_dir: /tao-pt/mount/data/imagenet2012/imagenet/val
test_dataset:
    images_dir: /tao-pt/mount/data/imagenet2012/imagenet/val
"""

sample_model_config = """
backbone:
    type: "vit_large_patch14_dinov2_swiglu"
    pretrained_backbone_path: /tao-pt/mount/pretrained_models/NVDINOv2/ViTL/NV_DINOV2_518.ckpt
    # pretrained_backbone_path: null
    freeze_backbone: True
head:
    type: "TAOLinearClsHead"
    binary: False
    topk: [1, 5]
    loss:
        type: CrossEntropyLoss
        label_smooth_val: 0.1
"""

sample_train_config = """
resume_training_checkpoint_path: null
num_epochs: 8
num_nodes: 1
num_gpus: 8
gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
validation_interval: 1
checkpoint_interval: 8
tensorboard:
    enabled: True
optim:
    lr: 0.01
    optim: "adamw"
    policy: "linear"
    weight_decay: 0.0005
"""

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )
)
CONFIG_ROOT = os.path.join(
    ROOT_DIR,"nvidia_tao_pytorch/cv/classification_pyt/experiment_specs"
)
train_config = os.path.join(CONFIG_ROOT, "experiment_spec.yaml")

with open(train_config, "r") as config_file:
    sample_experiment_config = config_file.read()

@pytest.fixture
def _test_dataset_spec():
    dataset_config = DatasetConfig()
    yield dataset_config

@pytest.fixture
def _test_data_path_format_spec():
    data_path_config = DataPathFormat()
    yield data_path_config

@pytest.fixture
def _test_model_spec():
    model_config = ModelConfig()
    yield model_config

@pytest.fixture
def _test_optimizer_spec():
    optimizer_config = OptimConfig()
    yield optimizer_config

@pytest.fixture
def _test_backbone_spec():
    backbone_config = BackboneConfig()
    yield backbone_config

@pytest.fixture
def _test_head_spec():
    head_config = HeadConfig()
    yield head_config

@pytest.fixture
def _test_random_crop_spec():
    random_crop_config = RandomCropWithScale()
    yield random_crop_config

@pytest.fixture
def _test_random_color_spec():
    random_color_config = RandomColor()
    yield random_color_config

@pytest.fixture
def _test_random_rotate_spec():
    random_rotate_config = RandomRotation()
    yield random_rotate_config

@pytest.fixture
def _test_random_flip_spec():
    random_flip_config = RandomFlip()
    yield random_flip_config

@pytest.fixture
def _test_augmentation_spec():
    augmentation_config = AugmentationConfig()
    yield augmentation_config

@pytest.fixture
def _test_train_spec():
    train_config = TrainExpConfig()
    yield train_config

@pytest.fixture
def _test_experiment_spec():
    experiment_config = ExperimentConfig()
    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.config
def test_dataset_jsonschema_conversion(_test_dataset_spec):
    """Test jsonschema conversion for dataset spec."""
    json_with_meta_config = dataclass_to_json(_test_dataset_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )

@pytest.mark.cv_unit
@pytest.mark.config
def test_data_path_jsonschema_conversion(_test_data_path_format_spec):
    """Test jsonschema conversion for data path spec."""
    json_with_meta_config = dataclass_to_json(_test_data_path_format_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )

@pytest.mark.cv_unit
@pytest.mark.config
def test_model_jsonschema_config(_test_model_spec):
    """Test jsonschema conversion for model spec."""
    json_with_meta_config = dataclass_to_json(_test_model_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )

@pytest.mark.cv_unit
@pytest.mark.config
def test_optimizer_jsonschema_config(_test_optimizer_spec):
    """Test jsonschema conversion for optimizer spec."""
    json_with_meta_config = dataclass_to_json(_test_optimizer_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )

@pytest.mark.cv_unit
@pytest.mark.config
def test_optimizer_jsonschema_config(_test_backbone_spec):
    """Test jsonschema conversion for optimizer spec."""
    json_with_meta_config = dataclass_to_json(_test_backbone_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )

@pytest.mark.cv_unit
@pytest.mark.config
def test_optimizer_jsonschema_config(_test_head_spec):
    """Test jsonschema conversion for optimizer spec."""
    json_with_meta_config = dataclass_to_json(_test_head_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )

@pytest.mark.cv_unit
@pytest.mark.config
def test_augmentation_rf_jsonschema_conversion(_test_random_flip_spec):
    """Test jsonschema conversion for augmentation random flip spec."""
    json_with_meta_config = dataclass_to_json(_test_random_flip_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )

@pytest.mark.cv_unit
@pytest.mark.config
def test_augmentation_rr_jsonschema_conversion(_test_random_rotate_spec):
    """Test jsonschema conversion for augmentation random rotate spec."""
    json_with_meta_config = dataclass_to_json(_test_random_rotate_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )

@pytest.mark.cv_unit
@pytest.mark.config
def test_augmentation_rc_jsonschema_conversion(_test_random_color_spec):
    """Test jsonschema conversion for augmentation random color spec."""
    json_with_meta_config = dataclass_to_json(_test_random_color_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )

@pytest.mark.cv_unit
@pytest.mark.config
def test_augmentation_rcp_jsonschema_conversion(_test_random_crop_spec):
    """Test jsonschema conversion for augmentation random crop spec."""
    json_with_meta_config = dataclass_to_json(_test_random_crop_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )

@pytest.mark.cv_unit
@pytest.mark.config
def test_augmentation_jsonschema_conversion(_test_augmentation_spec):
    """Test jsonschema conversion for augmentation spec."""
    json_with_meta_config = dataclass_to_json(_test_augmentation_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )

@pytest.mark.cv_unit
@pytest.mark.config
def test_train_jsonschema_conversion(_test_train_spec):
    """Test jsonschema conversion for train spec."""
    json_with_meta_config = dataclass_to_json(_test_train_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )

@pytest.mark.cv_unit
@pytest.mark.config
def test_experiment_jsonschema_conversion(_test_experiment_spec):
    """Test jsonschema conversion for experiment spec."""
    json_with_meta_config = dataclass_to_json(_test_experiment_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )



TEST_CONFIG_BLOCKS = [
    (sample_model_config, ModelConfig),
    (sample_dataset_config, DatasetConfig),
    (sample_train_config, TrainExpConfig),
    (sample_experiment_config, ExperimentConfig),
]

@pytest.mark.cv_unit
@pytest.mark.config
@pytest.mark.schema_validation
@pytest.mark.parametrize(
    "yaml_string, dataclass_class_name",
    TEST_CONFIG_BLOCKS
)
def test_load_experiment_spec(
    yaml_string,
    dataclass_class_name,
):
    """Simple function to load and validate the structure config from a yaml file."""
    schema = OmegaConf.structured(dataclass_class_name)
    config = OmegaConf.create(yaml_string)
    assert OmegaConf.merge(schema, config)
