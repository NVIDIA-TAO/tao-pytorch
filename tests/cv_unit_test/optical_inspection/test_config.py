# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple test cases to test config load and microservices jsonschema conversion."""

import os
import pytest
import json

from omegaconf import OmegaConf

from nvidia_tao_core.api_utils.dataclass2json_converter import create_json_schema, dataclass_to_json
from nvidia_tao_pytorch.config.optical_inspection.default_config import (
    ModelConfig,
    OptimConfig,
    RandomFlip, 
    RandomRotation, 
    RandomColor, 
    AugmentationConfig,
    DataPathFormat, 
    DatasetConfig,
    TensorBoardLogger,
    TrainExpConfig,
    ExperimentConfig
)

sample_dataset_config = """
train_dataset:
    csv_path: /data/dataset_convert/train_combined.csv
    images_dir: /data/images/
validation_dataset:
    csv_path: /data/dataset_convert/valid_combined.csv
    images_dir: /data/images/
test_dataset:
    csv_path: /data/dataset_convert/valid_combined.csv
    images_dir: /data/images/
infer_dataset:
    csv_path: /data/dataset_convert/valid_combined.csv
    images_dir: /data/images/
image_ext: .jpg
batch_size: 16
workers: 2
fpratio_sampling: 0.2
num_input: 4
input_map:
    LowAngleLight: 0
    SolderLight: 1
    UniformLight: 2
    WhiteLight: 3
concat_type: linear
grid_map:
    x: 2
    y: 2
image_width: 128
image_height: 128
augmentation_config:
    rgb_input_mean: [0.485, 0.456, 0.406]
    rgb_input_std: [0.229, 0.224, 0.225]
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
    with_random_crop: True
    with_random_blur: True
    augment: False
"""

sample_model_config = """
model_type: Siamese_3
model_backbone: custom
embedding_vectors: 5
margin: 2.0
"""

sample_train_config = """
pretrained_model_path: /path/to/pretrained.pth
optim:
    type: Adam
    lr: 0.0005
loss: contrastive
num_epochs: 15
checkpoint_interval: 5
validation_interval: 5
results_dir: "/results/train"
tensorboard:
    enabled: True
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
    ROOT_DIR,"nvidia_tao_pytorch/cv/optical_inspection/experiment_specs"
)
train_config = os.path.join(CONFIG_ROOT, "experiment.yaml")

with open(train_config, "r") as config_file:
    sample_experiment_config = config_file.read()


@pytest.fixture
def _test_tensorboard_spec():
    tensorboard_config = TensorBoardLogger()
    yield tensorboard_config


@pytest.fixture
def _test_dataset_spec():
    dataset_config = DatasetConfig()
    yield dataset_config

@pytest.fixture
def _test_train_spec():
    train_config = TrainExpConfig()
    yield train_config


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
def _test_experiment_spec():
    experiment_config = ExperimentConfig()
    yield experiment_config


@pytest.fixture
def _test_augmentation_spec():
    augmentation_config = AugmentationConfig()
    yield augmentation_config

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


@pytest.mark.cv_unit
@pytest.mark.config
def test_tensorboard_jsonschema_conversion(_test_tensorboard_spec):
    """Test jsonschema conversion for train spec."""
    json_with_meta_config = dataclass_to_json(_test_tensorboard_spec)
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
def test_augmentation_classify_jsonschema_conversion(_test_augmentation_spec):
    """Test jsonschema conversion for augmentation-classify spec."""
    json_with_meta_config = dataclass_to_json(_test_augmentation_spec)
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
def test_dataset_jsonschema_conversion(_test_dataset_spec):
    """Test jsonschema conversion for dataset spec."""
    json_with_meta_config = dataclass_to_json(_test_dataset_spec)
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
