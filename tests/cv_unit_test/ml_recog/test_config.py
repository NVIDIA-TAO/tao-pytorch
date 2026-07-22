# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple test cases to test config load and microservices jsonschema conversion."""

import os
import pytest
import json

from omegaconf import OmegaConf

from nvidia_tao_core.api_utils.dataclass2json_converter import create_json_schema, dataclass_to_json
from nvidia_tao_core.api_utils.json_schema_validation import validate_jsonschema
from nvidia_tao_pytorch.config.ml_recog.dataset import MLDatasetConfig, ColorAugmentation
from nvidia_tao_pytorch.config.ml_recog.model import MLModelConfig
from nvidia_tao_pytorch.config.ml_recog.train import MLTrainExpConfig, OptimConfig
from nvidia_tao_pytorch.config.ml_recog.default_config import ExperimentConfig


sample_dataset_config = """
train_dataset: /path/to/random/dir
val_dataset:
  reference: /path/to/random/dir
  query: /path/to/random/dir
workers: 12
pixel_mean: [0.485, 0.456, 0.406]
pixel_std: [0.226, 0.226, 0.226]
prob: 0.5
re_prob: 0.5
num_instance: 4
color_augmentation:
  enabled: True
gaussian_blur:
  enabled: True
"""

sample_model_config = """
backbone: resnet_101
pretrained_model_path: /path/to/random/model.pth
pretrained_trunk_path: /path/tao/random/trunk.pth
pretrained_embedder_path: /path/to/random/embedder.pth
input_width: 224
input_height: 224
feat_dim: 2048
"""

sample_train_config = """
optim:
  name: Adam
  steps: [40, 70]
  gamma: 0.1
  embedder:
    bias_lr_factor: 1
    weight_decay: 0.0001
    weight_decay_bias: 0.0005
    base_lr: 0.000001
    momentum: 0.9
  trunk:
    bias_lr_factor: 1
    weight_decay: 0.0001
    weight_decay_bias: 0.0005
    base_lr: 0.00001
    momentum: 0.9
  warmup_factor: 0.01
  warmup_iters: 10
  warmup_method: linear
  triplet_loss_margin: 0.3
  miner_function_margin: 0.1
num_epochs: 150
resume_training_checkpoint_path: /path/to/random/resume/checkpoint.pth
checkpoint_interval: 1
validation_interval: 1
smooth_loss: False
batch_size: 16
val_batch_size: 16
train_trunk: True
train_embedder: True
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
    ROOT_DIR, "nvidia_tao_pytorch/cv/ml_recog/experiment_specs"
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
    dataset_config = MLDatasetConfig()
    yield dataset_config


@pytest.fixture
def _test_augmentation_spec():
    augmentation_config = ColorAugmentation()
    yield augmentation_config


@pytest.fixture
def _test_model_spec():
    model_config = MLModelConfig()
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
    (sample_model_config, MLModelConfig),
    (sample_dataset_config, MLDatasetConfig),
    (sample_train_config, MLTrainExpConfig),
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
