# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple test cases to test config load and microservices jsonschema conversion."""

import os
import pytest
import json

from omegaconf import OmegaConf

from nvidia_tao_core.api_utils.dataclass2json_converter import create_json_schema, dataclass_to_json
from nvidia_tao_core.api_utils.json_schema_validation import validate_jsonschema
from nvidia_tao_pytorch.config.deformable_detr.dataset import DDDatasetConfig, DDAugmentationConfig
from nvidia_tao_pytorch.config.deformable_detr.model import DDModelConfig
from nvidia_tao_pytorch.config.deformable_detr.train import DDTrainExpConfig, OptimConfig
from nvidia_tao_pytorch.config.deformable_detr.default_config import ExperimentConfig

sample_dataset_config = """
train_data_sources:
- image_dir: /path/to/random/dir
  json_file: /path/to/random/dir
val_data_sources:
- image_dir:  /path/to/random/dir
  json_file:  /path/to/random/dir
num_classes: 91
batch_size: 2
workers: 8
"""

sample_model_config = """
pretrained_backbone_path:  /path/to/random/dir.file
backbone: gc_vit_tiny
train_backbone: False
num_feature_levels: 4
dec_layers: 6
enc_layers: 6
num_queries: 300
dropout_ratio: 0.3
dim_feedforward: 1024
"""

sample_train_config = """
num_gpus: 1
num_nodes: 1
validation_interval: 1
optim:
  lr_backbone: 2e-05
  lr: 2e-4
  lr_steps: [40]
  momentum: 0.9
num_epochs: 50
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
    ROOT_DIR,"nvidia_tao_pytorch/cv/deformable_detr/experiment_specs"
)
train_config = os.path.join(CONFIG_ROOT, "train.yaml")
with open(train_config, "r") as config_file:
    sample_experiment_config = config_file.read()


def generate_json_schema(dataclass_instance):
    """Simple function to generate json schema from an instance of the dataclass."""
    json_with_meta_config = dataclass_to_json(dataclass_instance)
    return create_json_schema(json_with_meta_config)


@pytest.fixture
def _test_dataset_spec():
    dataset_config = DDDatasetConfig()
    yield dataset_config


@pytest.fixture
def _test_augmentation_spec():
    augmentation_config = DDAugmentationConfig()
    yield augmentation_config


@pytest.fixture
def _test_model_spec():
    model_config = DDModelConfig()
    yield model_config


@pytest.fixture
def _test_optimizer_spec():
    optimizer_config = OptimConfig()
    yield optimizer_config

@pytest.fixture
def _test_experiment_spec():
    experiment_config = ExperimentConfig()
    yield experiment_config



@pytest.mark.cv_unit
@pytest.mark.config
def test_augmentation_jsonschema_conversion(_test_augmentation_spec):
    """Test jsonschema conversion for dataset spec."""
    json_with_meta_config = dataclass_to_json(_test_augmentation_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )


@pytest.mark.cv_unit
@pytest.mark.config
def test_dataset_jsonschema_conversion(_test_dataset_spec):
    """Test jsonschema conversion for augmentation spec."""
    json_with_meta_config = dataclass_to_json(_test_dataset_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )


@pytest.mark.cv_unit
@pytest.mark.config
def test_model_jsonschema_config(_test_model_spec):
    """Test jsonschema conversion for augmentation spec."""
    json_with_meta_config = dataclass_to_json(_test_model_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )

@pytest.mark.cv_unit
@pytest.mark.config
def test_optimizer_jsonschema_config(_test_optimizer_spec):
    """Test jsonschema conversion for augmentation spec."""
    json_with_meta_config = dataclass_to_json(_test_optimizer_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )

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
    (sample_model_config, DDModelConfig),
    (sample_dataset_config, DDDatasetConfig),
    (sample_train_config, DDTrainExpConfig),
    (sample_experiment_config, ExperimentConfig)
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
    assert OmegaConf.merge(schema, config), (
        "Json schema loading failed."
    )
    json_schema = generate_json_schema(dataclass_class_name())
    validation_status = validate_jsonschema(config, json_schema["properties"])
    assert not(validation_status), (
        f"Json schema validation failed with error: {validation_status}"
    )
