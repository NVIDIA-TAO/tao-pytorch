# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple test cases to test config load and microservices jsonschema conversion."""

import os
import pytest
import json

from omegaconf import OmegaConf

from nvidia_tao_core.api_utils.dataclass2json_converter import create_json_schema, dataclass_to_json
from nvidia_tao_core.api_utils.json_schema_validation import validate_jsonschema
from nvidia_tao_pytorch.config.action_recognition.default_config import (ARModelConfig, ARDatasetConfig,
                                                                             ARAugmentationConfig, ARTrainExpConfig,
                                                                             OptimConfig, ExperimentConfig)

sample_dataset_config = """
train_dataset_dir: /data/train
val_dataset_dir: /data/test
label_map:
  fall_floor: 0
  ride_bike: 1
batch_size: 32
workers: 8
clips_per_video: 5
augmentation_config:
  train_crop_type: no_crop
  horizontal_flip_prob: 0.5
  rgb_input_mean: [0.5]
  rgb_input_std: [0.5]
  val_center_crop: False
"""

sample_model_config = """
model_type: rgb
backbone: resnet_18
rgb_seq_length: 3
input_height: 224
input_width: 224
input_type: 3d
sample_strategy: consecutive
dropout_ratio: 0.0
"""

sample_train_config = """
optim:
  lr: 0.001
  momentum: 0.9
  weight_decay: 0.0001
  lr_scheduler: MultiStep
  lr_steps: [5, 15, 20]
  lr_decay: 0.1
num_epochs: 1
checkpoint_interval: 1
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
    ROOT_DIR,"nvidia_tao_pytorch/cv/action_recognition/experiment_specs"
)
train_config = os.path.join(CONFIG_ROOT, "experiment.yaml")
with open(train_config, "r") as config_file:
    sample_experiment_config = config_file.read()


def generate_json_schema(dataclass_instance):
    """Simple function to generate json schema from an instance of the dataclass."""
    json_with_meta_config = dataclass_to_json(dataclass_instance)
    return create_json_schema(json_with_meta_config)


@pytest.fixture
def _test_dataset_spec():
    dataset_config = ARDatasetConfig()
    yield dataset_config


@pytest.fixture
def _test_augmentation_spec():
    augmentation_config = ARAugmentationConfig()
    yield augmentation_config


@pytest.fixture
def _test_model_spec():
    model_config = ARModelConfig()
    yield model_config


@pytest.fixture
def _test_optimizer_spec():
    optimizer_config = OptimConfig()
    yield optimizer_config

@pytest.fixture
def _test_experiment_spec():
    experiment_config = ExperimentConfig()
    yield experiment_config


@pytest.mark.ar
@pytest.mark.cv_unit
@pytest.mark.config
def test_augmentation_jsonschema_conversion(_test_augmentation_spec):
    """Test jsonschema conversion for dataset spec."""
    json_with_meta_config = dataclass_to_json(_test_augmentation_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )

@pytest.mark.ar
@pytest.mark.cv_unit
@pytest.mark.config
def test_dataset_jsonschema_conversion(_test_dataset_spec):
    """Test jsonschema conversion for augmentation spec."""
    json_with_meta_config = dataclass_to_json(_test_dataset_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )

@pytest.mark.ar
@pytest.mark.cv_unit
@pytest.mark.config
def test_model_jsonschema_config(_test_model_spec):
    """Test jsonschema conversion for augmentation spec."""
    json_with_meta_config = dataclass_to_json(_test_model_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )
@pytest.mark.ar
@pytest.mark.cv_unit
@pytest.mark.config
def test_optimizer_jsonschema_config(_test_optimizer_spec):
    """Test jsonschema conversion for augmentation spec."""
    json_with_meta_config = dataclass_to_json(_test_optimizer_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )
@pytest.mark.ar
@pytest.mark.cv_unit
@pytest.mark.config
def test_experiment_jsonschema_conversion(_test_experiment_spec):
    """Test jsonschema conversion for augmentation spec."""
    json_with_meta_config = dataclass_to_json(_test_experiment_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )


TEST_CONFIG_BLOCKS = [
    (sample_model_config, ARModelConfig),
    (sample_dataset_config, ARDatasetConfig),
    (sample_train_config, ARTrainExpConfig),
    (sample_experiment_config, ExperimentConfig)
]
@pytest.mark.ar
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
    assert OmegaConf.merge(schema, config), (
        "Json schema loading failed."
    )
    json_schema = generate_json_schema(dataclass_class_name())
    validation_status = validate_jsonschema(config, json_schema["properties"])
    assert not(validation_status), (
        f"Json schema validation failed with error: {validation_status}"
    )
