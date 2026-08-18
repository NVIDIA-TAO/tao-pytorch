# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple test cases to test config load and microservices jsonschema conversion."""

import os
import pytest
import json

from omegaconf import OmegaConf

from nvidia_tao_core.api_utils.dataclass2json_converter import create_json_schema, dataclass_to_json
from nvidia_tao_pytorch.config.mask2former.dataset import Mask2FormerDatasetConfig, AugmentationConfig
from nvidia_tao_pytorch.config.mask2former.model import Mask2FormerModelConfig
from nvidia_tao_pytorch.config.mask2former.train import Mask2FormerTrainExpConfig, OptimConfig
from nvidia_tao_pytorch.config.mask2former.default_config import ExperimentConfig

sample_dataset_config = """
  contiguous_id: False
  label_map: /tao_experiments/mask2former/ade_color_map.json
  train:
    type: 'ade'
    name: "ade_train"
    annot_file: "/datasets/ade_train.jsonl"
    root_dir: ""
    batch_size: 16
    num_workers: 20
  val:
    type: 'ade'
    name: "ade_val"
    annot_file: "/datasets/ade_val.jsonl"
    root_dir: ""
    batch_size: 1
    num_workers: 2
  test:
    img_dir: /datasets/ade20k_test/
    batch_size: 2
    # target_size: [640, 640]
  augmentation:
    train_min_size: [640]
    train_max_size: 2560
    train_crop_size: [640, 640]
    test_min_size: 640
    test_max_size: 2560
"""

sample_model_config = """
  mode: "semantic"
  backbone:
    pretrained_weights: "/tao_experiments/mask2former/swin_large_patch4_window12_384_22k.pth"
    type: "swin"
    swin:
      type: "large"
      window_size: 12
      ape: False
      pretrain_img_size: 384
  mask_former:
    num_object_queries: 200
  sem_seg_head:
    norm: "GN"
"""

sample_train_config = """
  precision: 'fp16'
  num_gpus: 1
  checkpoint_interval: 1
  validation_interval: 1
  num_epochs: 130
  optim:
    type: "AdamW"
    lr: 0.0001
    weight_decay: 0.05
    lr_scheduler: "WarmupPoly"
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
    ROOT_DIR,"nvidia_tao_pytorch/cv/mask2former/experiment_specs"
)
train_config = os.path.join(CONFIG_ROOT, "spec.yaml")
with open(train_config, "r") as config_file:
    sample_experiment_config = config_file.read()


@pytest.fixture
def _test_dataset_spec():
    dataset_config = Mask2FormerDatasetConfig()
    yield dataset_config


@pytest.fixture
def _test_augmentation_spec():
    augmentation_config = AugmentationConfig()
    yield augmentation_config


@pytest.fixture
def _test_model_spec():
    model_config = Mask2FormerModelConfig()
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
    (sample_model_config, Mask2FormerModelConfig),
    (sample_dataset_config, Mask2FormerDatasetConfig),
    (sample_train_config, Mask2FormerTrainExpConfig),
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
    assert OmegaConf.merge(schema, config)
