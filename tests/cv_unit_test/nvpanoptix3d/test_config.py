# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Unit test for NVPanoptix3D config. """

import os
import pytest
import json

from omegaconf import OmegaConf

from nvidia_tao_core.api_utils.dataclass2json_converter import create_json_schema, dataclass_to_json
from nvidia_tao_pytorch.config.nvpanoptix3d.dataset import NVPanoptix3DDatasetConfig, AugmentationConfig
from nvidia_tao_pytorch.config.nvpanoptix3d.model import NVPanoptix3DModelConfig
from nvidia_tao_pytorch.config.nvpanoptix3d.train import NVPanoptix3DTrainExpConfig, OptimConfig
from nvidia_tao_pytorch.config.nvpanoptix3d.default_config import ExperimentConfig

sample_dataset_config = """
  name: "front3d"
  contiguous_id: True
  label_map: "nvidia_tao_pytorch/cv/nvpanoptix3d/experiment_specs/colormap.json"
  downsample_factor: 1
  frustum_mask_path: "nvidia_tao_pytorch/cv/nvpanoptix3d/experiment_specs/frustum_mask.npz"
  iso_value: 1.0
  ignore_label: 255
  enable_3d: True
  enable_mp_occ: True
  train:
    json_path: "/workspace/tao-pytorch/datasets/front3d/meta/train_3d.json"
    base_dir: "nvidia_tao_pytorch/cv/nvpanoptix3d/experiment_specs/front3d"
    batch_size: 2
    num_workers: 4
  val:
    json_path: "/workspace/tao-pytorch/datasets/front3d/meta/val_3d.json"
    base_dir: "nvidia_tao_pytorch/cv/nvpanoptix3d/experiment_specs/front3d"
    batch_size: 1
    num_workers: 2
  test:
    json_path: "/workspace/tao-pytorch/datasets/front3d/meta/test_3d.json"
    base_dir: "/workspace/tao-pytorch/datasets/front3d"
    batch_size: 2
    num_workers: 2
  augmentation:
    train_min_size: [240]
    train_max_size: 960
    test_min_size: 240
    test_max_size: 960
    size_divisibility: 32
    gen_aug_weight: 0.0
"""

sample_model_config = """
  object_mask_threshold: 0.8
  overlap_threshold: 0.8
  test_topk_per_image: 100
  mode: "panoptic"
  backbone:
    backbone_type: "vggt"
    pretrained_model_path: "nvidia_tao_pytorch/cv/nvpanoptix3d/experiment_specs/weights/model_commercial.pt"
  sem_seg_head:
    num_classes: 13
  mask_former:
    dropout: 0.0
    num_object_queries: 100
    deep_supervision: True
    no_object_weight: 0.1
    class_weight: 2.0
    mask_weight: 5.0
    dice_weight: 5.0
    depth_weight: 5.0
    mp_occ_weight: 5.0
    size_divisibility: 32
  frustum3d:
    truncation: 3.0
    iso_recon_value: 1.0
    panoptic_weight: 25.0
    completion_weights: [50.0, 25.0, 10.0]
    surface_weight: 5.0
    unet_output_channels: 16
    unet_features: 16
    use_multi_scale: True
    grid_dimensions: 256
    signed_channel: 3
    frustum_dims: 256
  projection:
    voxel_size: 0.03
    sign_channel: True
"""

sample_train_config = """
  checkpoint_2d: ""
  freeze: []
  precision: "fp32"
  num_gpus: 1
  num_nodes: 1
  checkpoint_interval_unit: step
  checkpoint_interval: 3000
  val_check_interval: 1
  num_epochs: 30
  clip_grad_norm: 0.01
  clip_grad_norm_type: 2.0
  clip_grad_type: "full"
  activation_checkpoint: False
  optim:
    type: "AdamW"
    lr: 0.0001
    weight_decay: 0.05
    lr_scheduler: "WarmupPoly"
    max_steps: 180000
    warmup_factor: 1.0
    warmup_iters: 0
    monitor_name: "train_loss"
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
    ROOT_DIR, "nvidia_tao_pytorch/cv/nvpanoptix3d/experiment_specs"
)
train_config = os.path.join(CONFIG_ROOT, "spec_front3d_3d.yaml")
with open(train_config, "r") as config_file:
    sample_experiment_config = config_file.read()


@pytest.fixture
def _test_dataset_spec():
    dataset_config = NVPanoptix3DDatasetConfig()
    yield dataset_config


@pytest.fixture
def _test_augmentation_spec():
    augmentation_config = AugmentationConfig()
    yield augmentation_config


@pytest.fixture
def _test_model_spec():
    model_config = NVPanoptix3DModelConfig()
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
    """Test jsonschema conversion for augmentation spec."""
    json_with_meta_config = dataclass_to_json(_test_augmentation_spec)
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
    (sample_model_config, NVPanoptix3DModelConfig),
    (sample_dataset_config, NVPanoptix3DDatasetConfig),
    (sample_train_config, NVPanoptix3DTrainExpConfig),
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
