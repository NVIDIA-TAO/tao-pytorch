# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple test cases to test config load and microservices jsonschema conversion."""

import os
import pytest
import json

from omegaconf import OmegaConf
from platform import machine

pytestmark = pytest.mark.skipif(
    ("aarch64" in machine().lower()) or ("arm" in machine().lower()),
    reason="Sparse4D tests take very long (~12 hours) on ARM architecture. TODO: Fix this.",
)

from nvidia_tao_core.api_utils.dataclass2json_converter import create_json_schema, dataclass_to_json
from nvidia_tao_core.api_utils.json_schema_validation import validate_jsonschema
from nvidia_tao_pytorch.config.sparse4d.default_config import ExperimentConfig
from nvidia_tao_pytorch.config.sparse4d.train import (
    Sparse4DOptimizerConfig,
    Sparse4DTrainConfig,
)
from nvidia_tao_pytorch.config.sparse4d.model import Sparse4DModelConfig

from nvidia_tao_pytorch.config.sparse4d.dataset import (
    Sparse4DAugmentationConfig,
    Omniverse3DDetTrackDatasetConfig
)

sample_dataset_config = """
use_h5_file_for_rgb: false
use_h5_file_for_depth: true
batch_size: 6
num_bev_groups: 1
num_workers: 6
num_ids: 70
classes: [
"person",
"humanoid",
"nova_carter",
"transporter",
"forklift",
"box",
"pallet",
"crate"
]
type: "omniverse_3d_det_track"
data_root: "/path/to/random/dir"
train_dataset:
    ann_file: "/path/to/random/dir/random_train_file.pkl"
    test_mode: false
    use_valid_flag: true
    with_seq_flag: true
    sequences_split_num: 100
    keep_consistent_seq_aug: true
    same_scene_in_batch: true
val_dataset:
    ann_file: "/path/to/random/dir/random_val_file.pkl"
    test_mode: true
    use_valid_flag: true
    tracking: true
    tracking_threshold: 0.2
test_dataset:
    ann_file: "/path/to/random/dir/ov_test_split/random_test_file.pkl"
    test_mode: true
    use_valid_flag: true
    tracking: true
    tracking_threshold: 0.2
augmentation:
    resize_lim: [0.7, 0.77]
    final_dim: [512, 1408]
    bot_pct_lim: [0.0, 0.0]
    rot_lim: [-5.4, 5.4]
    image_size: [1080, 1920]
    rand_flip: true
    rot3d_range: [-0.3925, 0.3925]
normalize:
    mean: [123.675, 116.28, 103.53]
    std: [58.395, 57.12, 57.375]
    to_rgb: true
sequences:
    split_num: 100
    keep_consistent_aug: true
    same_scene_in_batch: true
"""

sample_model_config = """
type: "sparse4d"
use_grid_mask: true
use_deformable_func: true
use_temporal_align: false
input_shape: [1408, 512]
embed_dims: 256
backbone:
    type: resnet101
neck:
    type: "FPN"
    num_outs: 4
    start_level: 0
    out_channels: 256
    in_channels: [256, 512, 1024, 2048]
    add_extra_convs: "on_output"
    relu_before_extra_convs: true
depth_branch:
    type: "dense_depth"
    embed_dims: "${model.embed_dims}"
    num_depth_layers: 3
    loss_weight: 0.2
head:
    type: "sparse4d"
    num_output: 300
    cls_threshold_to_reg: 0.05
    decouple_attn: true
    return_feature: true
    use_reid_sampling: false
    embed_dims: "${model.embed_dims}"
    num_groups: 8
    num_decoder: 6
    num_single_frame_decoder: 1
    drop_out: 0.1
    temporal: true
    with_quality_estimation: true
    instance_bank:
        num_anchor: 900
        anchor: "/path/to/random/file.npy"
        num_temp_instances: 600
        confidence_decay: 0.8
        feat_grad: false
        default_time_interval: 0.033333
        embed_dims: "${model.embed_dims}"
        use_temporal_align: "${model.use_temporal_align}"
    anchor_encoder:
        type: 'SparseBox3DEncoder'
        vel_dims: 3
        embed_dims: [128, 32, 32, 64]
        mode: 'cat'
        output_fc: false
        in_loops: 1
        out_loops: 4
    operation_order: [
        "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm", 
        "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm", 
        "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm", 
        "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm", 
        "deformable", "ffn", "norm", "refine", "temp_gnn", "gnn", "norm", 
        "deformable", "ffn", "norm", "refine"
    ]
    temp_graph_model:
        type: "MultiheadAttention"
        embed_dims: 512
        num_heads: 8
        batch_first: true
        dropout: 0.1
    graph_model:
        type: "MultiheadAttention"
        embed_dims: "${model.head.temp_graph_model.embed_dims}"
        num_heads: "${model.head.temp_graph_model.num_heads}"
        batch_first: true
        dropout: "${model.head.temp_graph_model.dropout}"
    norm_layer:
        type: "LN"
        normalized_shape: "${model.embed_dims}"
    ffn:
        type: "AsymmetricFFN"
        in_channels: 512
        pre_norm:
            type: "LN"
        embed_dims: 256
        feedforward_channels: 1024
        num_fcs: 2
        ffn_drop: 0.1
        act_cfg:
            type: "ReLU"
            inplace: true
    deformable_model:
        embed_dims: "${model.embed_dims}"
        num_groups: 8
        num_levels: 4
        attn_drop: 0.15
        use_deformable_func: true
        use_camera_embed: false
        residual_mode: "cat"
        kps_generator:
            embed_dims: "${model.embed_dims}"
            num_learnable_pts: 6
            fix_scale:
                - [0, 0, 0]
                - [0.45, 0, 0]
                - [-0.45, 0, 0]
                - [0, 0.45, 0]
                - [0, -0.45, 0]
                - [0, 0, 0.45]
                - [0, 0, -0.45]
    refine_layer:
        type: "SparseBox3DRefinementModule"
        embed_dims: "${model.embed_dims}"
        refine_yaw: true
        with_quality_estimation: true
    sampler:
        num_dn_groups: 5
        num_temp_dn_groups: 3
        dn_noise_scale: [2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        max_dn_gt: 128
        add_neg_dn: true
        cls_weight: 2.0
        box_weight: 0.25
        reg_weights: [2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        use_temporal_align: "${model.use_temporal_align}"
    visibility_net:
        type: "visibility_net"
        embedding_dim: 256
        hidden_channels: 32
    loss:
        reg:
            type: "sparse_box_3d"
            box_weight: 0.25
        cls:
            type: "focal"
            use_sigmoid: true
            gamma: 2.0
            alpha: 0.25
            loss_weight: 2.0
        id:
            type: "cross_entropy_label_smooth"
            num_ids: "${dataset.num_ids}"
    bnneck:
        type: "bnneck"
        feat_dim: 256
        num_ids: "${dataset.num_ids}"
    decoder:
        type: "SparseBox3DDecoder"
        score_threshold: 0.05
    reg_weights: [2.0, 2.0, 2.0, 1 ,1, 1, 1, 1, 1, 1, 1]
"""

sample_train_config = """
num_epochs: 50
num_nodes: 1
num_gpus: 1
validation_interval: 1
checkpoint_interval: 10
pretrained_model_path: "/path/to/random/dir.file"
precision: bf16
optim:
    type: "adamw"
    lr: 1e-5
    weight_decay: 0.001
    paramwise_cfg:
        custom_keys:
        img_backbone:
            lr_mult: 0.2
    grad_clip:
        max_norm: 25
        norm_type: 2
    lr_scheduler:
        policy: "cosine"
        warmup: "linear"
        warmup_iters: 500
        warmup_ratio: 0.333333
        min_lr_ratio: 0.001
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
    ROOT_DIR, "nvidia_tao_pytorch/cv/sparse4d/experiment_specs"
)
train_config = os.path.join(CONFIG_ROOT, "experiment_spec.yaml")
with open(train_config, "r") as config_file:
    sample_experiment_config = config_file.read()


@pytest.fixture
def _test_dataset_spec():
    dataset_config = Omniverse3DDetTrackDatasetConfig()
    yield dataset_config


@pytest.fixture
def _test_augmentation_spec():
    augmentation_config = Sparse4DAugmentationConfig()
    yield augmentation_config

@pytest.fixture
def _test_optimizer_spec():
    optimizer_config = Sparse4DOptimizerConfig()
    yield optimizer_config

@pytest.fixture
def _test_model_spec():
    model_config = Sparse4DModelConfig()
    yield model_config


@pytest.fixture
def _test_experiment_spec():
    experiment_config = ExperimentConfig()
    yield experiment_config

def generate_json_schema(dataclass_instance):
    """Simple function to generate json schema from an instance of the dataclass."""
    json_with_meta_config = dataclass_to_json(dataclass_instance)
    return create_json_schema(json_with_meta_config)


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
    json_schema
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
    (sample_model_config, Sparse4DModelConfig),
    (sample_dataset_config, Omniverse3DDetTrackDatasetConfig),
    (sample_train_config, Sparse4DTrainConfig),
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
