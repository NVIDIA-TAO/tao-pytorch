# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple test cases to test config load and microservices jsonschema conversion."""

import os
import pytest
import json

from omegaconf import OmegaConf

from nvidia_tao_core.api_utils.dataclass2json_converter import create_json_schema, dataclass_to_json
from nvidia_tao_core.api_utils.json_schema_validation import validate_jsonschema
from nvidia_tao_pytorch.config.ocdnet.default_config import ExperimentConfig, OCDNetDataConfig, OCDNetModelConfig, OCDNetTrainExpConfig


sample_model_config = """
load_pruned_graph: False
pruned_graph_path: '/results/prune/pruned_0.1.pth'
pretrained_model_path: '/data/ocdnet/ocdnet_deformable_resnet18.pth'
backbone: deformable_resnet18
"""

sample_train_config = """
results_dir: /results/train
num_gpus: 1
num_epochs: 5
checkpoint_interval: 1
validation_interval: 1
trainer:
  clip_grad_norm: 5.0
optimizer:
  type: Adam
  args:
    lr: 0.001
lr_scheduler:
  type: WarmupPolyLR
  args:
    warmup_epoch: 3
post_processing:
  type: SegDetectorRepresenter
  args:
    thresh: 0.3
    box_thresh: 0.55
    max_candidates: 1000
    unclip_ratio: 1.5
metric:
  type: QuadMetric
  args:
    is_output_polygon: false
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
    ROOT_DIR,"nvidia_tao_pytorch/cv/ocdnet/experiment_specs"
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
    dataset_config = OCDNetDataConfig()
    yield dataset_config


@pytest.fixture
def _test_model_spec():
    model_config = OCDNetModelConfig()
    yield model_config


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
def test_experiment_jsonschema_conversion(_test_experiment_spec):
    """Test jsonschema conversion for augmentation spec."""
    json_with_meta_config = dataclass_to_json(_test_experiment_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )


TEST_CONFIG_BLOCKS = [
    (sample_model_config, OCDNetModelConfig),
    (sample_train_config, OCDNetTrainExpConfig),
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
