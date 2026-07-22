# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple test cases to test config load and microservices jsonschema conversion."""

import os
import pytest
import json

from omegaconf import OmegaConf

from nvidia_tao_core.api_utils.dataclass2json_converter import create_json_schema, dataclass_to_json
from nvidia_tao_core.api_utils.json_schema_validation import validate_jsonschema
from nvidia_tao_pytorch.config.centerpose.default_config import ExperimentConfig
from nvidia_tao_pytorch.config.centerpose.dataset import CenterPoseDatasetConfig
from nvidia_tao_pytorch.config.centerpose.model import CenterPoseModelConfig
from nvidia_tao_pytorch.config.centerpose.train import CenterPoseTrainExpConfig, OptimConfig

sample_dataset_config = """
train_data: /path/to/random/dir
test_data:  /path/to/random/dir
num_classes: 1
batch_size: 2
workers: 4
category: bike
num_symmetry: 1
max_objs: 10
"""

sample_model_config = """
down_ratio: 4
use_pretrained: True
backbone:
  model_type: fan_small
  pretrained_backbone_path: /path/to/random/dir.file
"""

sample_train_config = """
num_gpus: 1
validation_interval: 20
checkpoint_interval: 20
num_epochs: 40
clip_grad_val: 100.0
seed: 317
resume_training_checkpoint_path: /path/to/random/model.pth
precision: fp32

optim:
  lr: 6e-05
  lr_steps: [90, 120]
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
    ROOT_DIR, "nvidia_tao_pytorch/cv/centerpose/experiment_specs"
)
train_config = os.path.join(CONFIG_ROOT, "train.yaml")
with open(train_config, "r") as config_file:
    sample_experiment_config = config_file.read()
experiment_config = OmegaConf.create(sample_experiment_config)
experiment_config.dataset = OmegaConf.create(sample_dataset_config)
experiment_config.results_dir = "/path/to/experiment/results/dir"
sample_experiment_config = OmegaConf.to_yaml(experiment_config)


def generate_json_schema(dataclass_instance):
    """Simple function to generate json schema from an instance of the dataclass."""
    json_with_meta_config = dataclass_to_json(dataclass_instance)
    return create_json_schema(json_with_meta_config)


@pytest.fixture
def _test_dataset_spec():
    dataset_config = CenterPoseDatasetConfig()
    yield dataset_config


@pytest.fixture
def _test_model_spec():
    model_config = CenterPoseModelConfig()
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
    (sample_model_config, CenterPoseModelConfig),
    (sample_dataset_config, CenterPoseDatasetConfig),
    (sample_train_config, CenterPoseTrainExpConfig),
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
