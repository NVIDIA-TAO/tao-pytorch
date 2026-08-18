# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple test cases to test config load and microservices jsonschema conversion."""

import os
import pytest
import json

from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.depth_net.dataset import DepthNetDatasetConfig, DepthNetAugmentationConfig
from nvidia_tao_pytorch.config.depth_net.model import DepthNetModelConfig
from nvidia_tao_pytorch.config.depth_net.train import DepthNetTrainExpConfig, OptimConfig
from nvidia_tao_pytorch.config.depth_net.default_config import ExperimentConfig
from nvidia_tao_core.api_utils.dataclass2json_converter import create_json_schema, dataclass_to_json
from nvidia_tao_core.api_utils.json_schema_validation import validate_jsonschema

sample_dataset_config = """
dataset_name: MonoDataset
train_dataset:
    data_sources:
        - dataset_name: Middlebury
          data_file: /media/projects.metropolis4/datasets/Middlebury/test_clean.txt
    batch_size: 12
val_dataset:
    data_sources:
        - dataset_name: Middlebury
          data_file: /media/projects.metropolis4/datasets/Middlebury/test_clean.txt
    batch_size: 1
test_dataset:
    data_sources:
        - dataset_name: Middlebury
          data_file: /media/projects.metropolis4/datasets/Middlebury/test_clean.txt
    batch_size: 10
infer_dataset:
    data_sources:
        - dataset_name: Middlebury
          data_file: /media/projects.metropolis4/datasets/Middlebury/test_clean.txt
    batch_size: 10
"""

sample_model_config = """
model_type: RelativeDepthAnything
encoder: vitl
"""

sample_train_config = """
num_gpus: 1
num_nodes: 1
num_epochs: 1
optim:
  lr: 0.000006
  lr_scheduler: LambdaLR
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
    ROOT_DIR,"nvidia_tao_pytorch/cv/depth_net/experiment_specs"
)
train_config = os.path.join(CONFIG_ROOT, "experiment_mono_relative.yaml")
with open(train_config, "r") as config_file:
    sample_experiment_config = config_file.read()


def generate_json_schema(dataclass_instance):
    """Simple function to generate json schema from an instance of the dataclass."""
    json_with_meta_config = dataclass_to_json(dataclass_instance)
    return create_json_schema(json_with_meta_config)


@pytest.fixture
def _test_dataset_spec():
    dataset_config = DepthNetDatasetConfig()
    yield dataset_config


@pytest.fixture
def _test_augmentation_spec():
    augmentation_config = DepthNetAugmentationConfig()
    yield augmentation_config


@pytest.fixture
def _test_model_spec():
    model_config = DepthNetModelConfig()
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
    (sample_model_config, DepthNetModelConfig),
    (sample_dataset_config, DepthNetDatasetConfig),
    (sample_train_config, DepthNetTrainExpConfig),
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
