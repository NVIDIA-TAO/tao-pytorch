# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple test cases to test config load and microservices jsonschema conversion."""
import json
import os

import pytest
from omegaconf import OmegaConf

from nvidia_tao_core.api_utils.dataclass2json_converter import (create_json_schema,
                                                                dataclass_to_json)
from nvidia_tao_pytorch.config.oneformer.default_config import ExperimentConfig
from nvidia_tao_pytorch.config.oneformer.dataset import OneFormerDatasetConfig
from nvidia_tao_pytorch.config.oneformer.model import OneFormerModelConfig
from nvidia_tao_pytorch.config.oneformer.train import (OneFormerTrainExpConfig,
                                                    OptimConfig)

sample_dataset_config = """
    train:
        images: /workspace/datasets/coco/train2017
        annotations: /workspace/datasets/coco/annotations/panoptic_train2017.json
        panoptic: /workspace/datasets/coco/panoptic_train2017
        batch_size: 4
        num_workers: 4
    val:
        images: /workspace/datasets/coco/val2017
        annotations: /workspace/datasets/coco/annotations/panoptic_val2017.json
        panoptic: /workspace/datasets/coco/panoptic_val2017
        batch_size: 4
        num_workers: 4
    test:
        images: /workspace/datasets/coco/val2017
        annotations: /workspace/datasets/coco/annotations/panoptic_val2017.json
        panoptic: /workspace/datasets/coco/panoptic_val2017
        batch_size: 4
        num_workers: 4
    image_size: 1024
    label_map: /workspace/datasets/coco/label_map.json
    cutmix_prob: 0.0
"""

sample_model_config = """
    backbone:
      name: D2SwinTransformer
      freeze_at: 0
      swin:
        embed_dim: 192
        depths: [2, 2, 18, 2]
        num_heads: [6, 12, 24, 48]
        window_size: 12
        mlp_ratio: 4.0
        patch_size: 4
        patch_norm: true
        ape: false
        pretrain_img_size: 384
        qkv_bias: true
        qk_scale: null
        attn_drop_rate: 0.0
        drop_rate: 0.0
        drop_path_rate: 0.3
        out_features: [res2, res3, res4, res5]
        out_indices: [0, 1, 2, 3]
        use_checkpoint: false
    one_former:
        num_object_queries: 150
    sem_seg_head:
        num_classes: 133
    test:
        test_topk_per_image: 100
        object_mask_threshold: 0.8
"""

sample_train_config = """
    num_epochs: 50
    num_gpus: 8
    num_nodes: 4
    pretrained_model: nvidia_tao_pytorch/cv/oneformer/checkpoints/nvppnet/swin/train/model_epoch_011_step_42348.pth
    pretrained_backbone:
    precision: 32
    iters_per_epoch: 15000
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
    ROOT_DIR, "nvidia_tao_pytorch/cv/oneformer/experiment_specs"
)
train_config = os.path.join(CONFIG_ROOT, "spec_coco.yaml")
with open(train_config, "r") as config_file:
    sample_experiment_config = config_file.read()


@pytest.fixture
def _test_dataset_spec():
    """Create a sample OneFormerDatasetConfig."""
    dataset_config = OneFormerDatasetConfig()
    yield dataset_config


@pytest.fixture
def _test_model_spec():
    """Create a sample OneFormerModelConfig."""
    model_config = OneFormerModelConfig()
    yield model_config


@pytest.fixture
def _test_optimizer_spec():
    """Create a sample OptimConfig."""
    optimizer_config = OptimConfig()
    yield optimizer_config


@pytest.fixture
def _test_experiment_spec():
    """Create a sample ExperimentConfig."""
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
    (sample_model_config, OneFormerModelConfig),
    (sample_dataset_config, OneFormerDatasetConfig),
    (sample_train_config, OneFormerTrainExpConfig),
    (sample_experiment_config, ExperimentConfig)
]


@pytest.mark.cv_unit
@pytest.mark.config
@pytest.mark.schema_validation
@pytest.mark.parametrize(
    "yaml_string, dataclass_class_name",
    TEST_CONFIG_BLOCKS
)
@pytest.mark.skip(reason="flaky test pending fix")
def test_load_experiment_spec(
    yaml_string,
    dataclass_class_name,
):
    """Simple function to load and validate the structure config from a yaml file."""
    schema = OmegaConf.structured(dataclass_class_name)
    config = OmegaConf.create(yaml_string)
    assert OmegaConf.merge(schema, config)
