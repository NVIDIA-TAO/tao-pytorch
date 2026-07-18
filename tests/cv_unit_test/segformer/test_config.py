# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple test cases to test config load and microservices jsonschema conversion."""

import os
import pytest
import json

from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.segformer.default_config import (
    SFDatasetConfig,
    DataPathFormat,
    SFDatasetSegmentConfig,
    SFModelConfig,
    SFOptimConfig,
    BackboneConfig,
    SegFormerHeadConfig,
    RandomFlip,
    RandomRotation,
    RandomColor,
    RandomCropWithScale,
    SFAugmentationSegmentConfig,
    SFTrainSegmentConfig,
    SFTrainExpConfig,
    ExperimentConfig
)
from nvidia_tao_core.api_utils.dataclass2json_converter import create_json_schema, dataclass_to_json

sample_dataset_config = """
segment:
    root_dir: data/dir
    label_transform: "norm"
    dataset: "SFDataset"
    num_classes: 2
    img_size: 224
    batch_size: 4
    workers: 4
    shuffle: True
    train_split: "train"
    validation_split: "val"
    test_split: 'test'
    predict_split: 'test'
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
        with_random_blur: True
        mean: [0.5, 0.5, 0.5]
        std: [0.5, 0.5, 0.5]
"""

sample_model_config = """
backbone:
    type: "vit_giant_nvdinov2"
    feat_downsample: False
    pretrained_backbone_path: /path/to/random/dir.file
    freeze_backbone: False
decode_head:
    in_channels: [64, 128, 320, 512]
    in_index: [0, 1, 2, 3]
    feature_strides: [4, 8, 16, 32]
    align_corners: False
    decoder_params: {"embed_dim": 768}
"""

sample_train_config = """
resume_training_checkpoint_path: /path/to/random/dir.file
pretrained_model_path:  /path/to/changenet_segment.pth
segment:
    loss: "ce"
num_epochs: 300
num_nodes: 1
validation_interval: 1
checkpoint_interval: 300
optim:
    lr: 0.0001
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
    ROOT_DIR,"nvidia_tao_pytorch/cv/segformer/experiment_specs"
)
train_config = os.path.join(CONFIG_ROOT, "experiment_spec.yaml")

with open(train_config, "r") as config_file:
    sample_experiment_config = config_file.read()

@pytest.fixture
def _test_dataset_spec():
    dataset_config = SFDatasetConfig()
    yield dataset_config

@pytest.fixture
def _test_data_path_format_spec():
    data_path_config = DataPathFormat()
    yield data_path_config

@pytest.fixture
def _test_dataset_segment_spec():
    dataset_segment_config = SFDatasetSegmentConfig()
    yield dataset_segment_config

@pytest.fixture
def _test_model_spec():
    model_config = SFModelConfig()
    yield model_config

@pytest.fixture
def _test_optimizer_spec():
    optimizer_config = SFOptimConfig()
    yield optimizer_config

@pytest.fixture
def _test_backbone_spec():
    backbone_config = BackboneConfig()
    yield backbone_config

@pytest.fixture
def _test_head_spec():
    head_config = SegFormerHeadConfig()
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
def _test_augmentation_segment_spec():
    augmentation_segment_config = SFAugmentationSegmentConfig()
    yield augmentation_segment_config

@pytest.fixture
def _test_train_segment_spec():
    train_segment_config = SFTrainSegmentConfig()
    yield train_segment_config

@pytest.fixture
def _test_train_spec():
    train_config = SFTrainExpConfig()
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
def test_dataset_segment_jsonschema_conversion(_test_dataset_segment_spec):
    """Test jsonschema conversion for dataset_segment spec."""
    json_with_meta_config = dataclass_to_json(_test_dataset_segment_spec)
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
def test_augmentation_segment_jsonschema_conversion(_test_augmentation_segment_spec):
    """Test jsonschema conversion for augmentation-segment spec."""
    json_with_meta_config = dataclass_to_json(_test_augmentation_segment_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )

@pytest.mark.cv_unit
@pytest.mark.config
def test_train_segment_jsonschema_conversion(_test_train_segment_spec):
    """Test jsonschema conversion for train-segment spec."""
    json_with_meta_config = dataclass_to_json(_test_train_segment_spec)
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
    (sample_model_config, SFModelConfig),
    (sample_dataset_config, SFDatasetConfig),
    (sample_train_config, SFTrainExpConfig),
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
