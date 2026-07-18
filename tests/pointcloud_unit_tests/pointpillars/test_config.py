# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple test cases to test config load and microservices jsonschema conversion for PointPillars."""

import os
import pytest
import json

from omegaconf import OmegaConf

from nvidia_tao_core.api_utils.dataclass2json_converter import create_json_schema, dataclass_to_json
from nvidia_tao_pytorch.config.pointpillars.default_config import (
    ExperimentConfig,
    PPDatasetConfig,
    PPModelConfig,
    PPTrainConfig,
)


sample_model_config = """
name: PointPillar
pretrained_model_path: null
vfe:
    name: PillarVFE
    with_distance: False
    use_absolue_xyz: True
    use_norm: True
    num_filters: [64]
map_to_bev:
    name: PointPillarScatter
    num_bev_features: 64
backbone_2d:
    name: BaseBEVBackbone
    layer_nums: [3, 5, 5]
    layer_strides: [2, 2, 2]
    num_filters: [64, 128, 256]
    upsample_strides: [1, 2, 4]
    num_upsample_filters: [128, 128, 128]
dense_head:
    name: AnchorHeadSingle
    class_agnostic: False
    use_direction_classifier: True
    dir_offset: 0.78539
    dir_limit_offset: 0.0
    num_dir_bins: 2
    anchor_generator_config: [
        {
            'class_name': 'Car',
            'anchor_sizes': [[3.9, 1.6, 1.56]],
            'anchor_rotations': [0, 1.57],
            'anchor_bottom_heights': [-1.78],
            'align_center': False,
            'feature_map_stride': 2,
            'matched_threshold': 0.6,
            'unmatched_threshold': 0.45
        },
        {
            'class_name': 'Pedestrian',
            'anchor_sizes': [[0.8, 0.6, 1.73]],
            'anchor_rotations': [0, 1.57],
            'anchor_bottom_heights': [-0.6],
            'align_center': False,
            'feature_map_stride': 2,
            'matched_threshold': 0.5,
            'unmatched_threshold': 0.35
        },
        {
            'class_name': 'Cyclist',
            'anchor_sizes': [[1.76, 0.6, 1.73]],
            'anchor_rotations': [0, 1.57],
            'anchor_bottom_heights': [-0.6],
            'align_center': False,
            'feature_map_stride': 2,
            'matched_threshold': 0.5,
            'unmatched_threshold': 0.35
        }
    ]
    target_assigner_config:
        name: AxisAlignedTargetAssigner
        pos_fraction: -1.0
        sample_size: 512
        norm_by_num_examples: False
        match_height: False
        box_coder: ResidualCoder
    loss_config:
        loss_weights: {
            'cls_weight': 1.0,
            'loc_weight': 2.0,
            'dir_weight': 0.2,
            'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }
post_processing:
    recall_thresh_list: [0.3, 0.5, 0.7]
    score_thresh: 0.1
    output_raw_score: False
    eval_metric: kitti
    nms_config:
        multi_classes_nms: False
        nms_type: nms_gpu
        nms_thresh: 0.01
        nms_pre_max_size: 4096
        nms_post_max_size: 500
sync_bn: False
"""

sample_train_config = """
batch_size: 4
num_epochs: 80
optimizer: adam_onecycle
lr: 0.003
weight_decay: 0.01
momentum: 0.9
moms: [0.95, 0.85]
pct_start: 0.4
div_factor: 10
decay_step_list: [35, 45]
lr_decay: 0.1
lr_clip: 0.0000001
lr_warmup: False
warmup_epoch: 1
grad_norm_clip: 10
resume_training_checkpoint_path: null #"/data/zhimengf/pointpillar_workspace/21/ckpt/checkpoint_epoch_80.pth"
pruned_model_path: "/data/zhimengf/pointpillar_workspace/33/pruned_0.5.tlt"
tcp_port: 18888
random_seed: null
checkpoint_interval: 1
max_checkpoint_save_num: 30
merge_all_iters_to_one_epoch: False
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
CONFIG_ROOT = os.path.join(ROOT_DIR, "nvidia_tao_pytorch/pointcloud/pointpillars/tools/cfgs/")
train_config = os.path.join(CONFIG_ROOT, "pointpillar_general.yaml")
with open(train_config, "r") as config_file:
    sample_experiment_config = config_file.read()


@pytest.fixture
def _test_dataset_spec():
    dataset_config = PPDatasetConfig()
    yield dataset_config


@pytest.fixture
def _test_model_spec():
    model_config = PPModelConfig()
    yield model_config


@pytest.fixture
def _test_experiment_spec():
    experiment_config = ExperimentConfig()
    yield experiment_config


@pytest.mark.pointcloud_unit
@pytest.mark.config
def test_dataset_jsonschema_conversion(_test_dataset_spec):
    """Test jsonschema conversion for augmentation spec."""
    json_with_meta_config = dataclass_to_json(_test_dataset_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )


@pytest.mark.pointcloud_unit
@pytest.mark.config
def test_model_jsonschema_config(_test_model_spec):
    """Test jsonschema conversion for augmentation spec."""
    json_with_meta_config = dataclass_to_json(_test_model_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )


@pytest.mark.pointcloud_unit
@pytest.mark.config
def test_experiment_jsonschema_conversion(_test_experiment_spec):
    """Test jsonschema conversion for augmentation spec."""
    json_with_meta_config = dataclass_to_json(_test_experiment_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )


TEST_CONFIG_BLOCKS = [
    (sample_model_config, PPModelConfig),
    (sample_train_config, PPTrainConfig),
    (sample_experiment_config, ExperimentConfig)
]

@pytest.mark.pointcloud_unit
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
