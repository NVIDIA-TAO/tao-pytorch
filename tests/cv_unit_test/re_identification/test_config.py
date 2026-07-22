# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple test cases to test config load and microservices jsonschema conversion."""

import os
import pytest
import json

from omegaconf import OmegaConf

from nvidia_tao_core.api_utils.dataclass2json_converter import create_json_schema, dataclass_to_json
from nvidia_tao_pytorch.config.re_identification.default_config import ReIDModelConfig, OptimConfig, ReIDDatasetConfig, ReIDReRankingConfig, ReIDTrainExpConfig, ReIDInferenceExpConfig, ReIDEvalExpConfig, ReIDExportExpConfig, ExperimentConfig

sample_model_config = """
backbone: swin_tiny_patch4_window7_224
last_stride: 1
pretrain_choice: self
pretrained_model_path: "/model/market1501/swintiny_pretrained.pth"
input_channels: 3
input_width: 128
input_height: 256
neck: bnneck
stride_size: [16, 16]
reduce_feat_dim: True
feat_dim: 256
no_margin: True
neck_feat: after
metric_loss_type: triplet
with_center_loss: False
with_flip_feature: False
label_smooth: False
pretrain_hw_ratio: 2
"""

sample_optim_config = """
name: SGD
lr_steps: [40, 70]
gamma: 0.1
bias_lr_factor: 2
weight_decay: 0.0001
weight_decay_bias: 0.0001
warmup_factor: 0.01
warmup_epochs: 20
warmup_method: cosine
base_lr: 0.0008
momentum: 0.9
center_loss_weight: 0.0005
center_lr: 0.5
triplet_loss_margin: 0.3
large_fc_lr: False
"""

sample_dataset_config = """
train_dataset_dir: "/data/market1501/sample_train"
test_dataset_dir: "/data/market1501/sample_test"
query_dataset_dir: "/data/market1501/sample_query"
num_classes: 100
batch_size: 64
val_batch_size: 128
num_workers: 1
pixel_mean: [0.5, 0.5, 0.5]
pixel_std: [0.5, 0.5, 0.5]
padding: 10
prob: 0.5
re_prob: 0.5
sampler: softmax_triplet
num_instances: 4
"""

sample_reranking_config = """
re_ranking: False
k1: 20
k2: 6
lambda_value: 0.3
"""

sample_train_config = """
optim:
    name: SGD
    lr_steps: [40, 70]
    gamma: 0.1
    bias_lr_factor: 2
    weight_decay: 0.0001
    weight_decay_bias: 0.0001
    warmup_factor: 0.01
    warmup_epochs: 20
    warmup_method: cosine
    base_lr: 0.0008
    momentum: 0.9
    center_loss_weight: 0.0005
    center_lr: 0.5
    triplet_loss_margin: 0.3
    large_fc_lr: False
num_epochs: 1
checkpoint_interval: 1
"""

sample_inference_config = """
output_file: "results/output.json"
test_dataset: "/data/market1501/sample_test"
query_dataset: "/data/market1501/sample_query"
"""

sample_eval_config = """
output_sampled_matches_plot: "results/matches.jpg"
output_cmc_curve_plot: "results/cmc_curve.jpg"
test_dataset: "/data/market1501/sample_test"
query_dataset: "/data/market1501/sample_query"
"""

sample_export_config = """
results_dir: "results/"
checkpoint: "results/trained_checkpoint.pth"
onnx_file: "results/output_model.onnx"
gpu_id: 0
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
    ROOT_DIR,"nvidia_tao_pytorch/cv/re_identification/experiment_specs"
)
train_config = os.path.join(CONFIG_ROOT, "experiment_market1501_swin.yaml")
with open(train_config, "r") as config_file:
    sample_experiment_config = config_file.read()


@pytest.fixture
def _test_model_spec():
    model_config = ReIDModelConfig()
    yield model_config


@pytest.fixture
def _test_optimizer_spec():
    optimizer_config = OptimConfig()
    yield optimizer_config


@pytest.fixture
def _test_dataset_spec():
    dataset_config = ReIDDatasetConfig()
    yield dataset_config


@pytest.fixture
def _test_reranking_spec():
    reranking_config = ReIDReRankingConfig()
    yield reranking_config


@pytest.fixture
def _test_train_spec():
    train_config = ReIDTrainExpConfig()
    yield train_config


@pytest.fixture
def _test_inf_spec():
    inference_config = ReIDInferenceExpConfig()
    yield inference_config


@pytest.fixture
def _test_eval_spec():
    eval_config = ReIDEvalExpConfig()
    yield eval_config


@pytest.fixture
def _test_export_spec():
    export_config = ReIDExportExpConfig()
    yield export_config


@pytest.fixture
def _test_experiment_spec():
    experiment_config = ExperimentConfig()
    yield experiment_config



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
def test_dataset_jsonschema_conversion(_test_dataset_spec):
    """Test jsonschema conversion for dataset spec."""
    json_with_meta_config = dataclass_to_json(_test_dataset_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )


@pytest.mark.cv_unit
@pytest.mark.config
def test_experiment_jsonschema_conversion(_test_reranking_spec):
    """Test jsonschema conversion for reranking spec."""
    json_with_meta_config = dataclass_to_json(_test_reranking_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )


@pytest.mark.cv_unit
@pytest.mark.config
def test_experiment_jsonschema_conversion(_test_train_spec):
    """Test jsonschema conversion for train spec."""
    json_with_meta_config = dataclass_to_json(_test_train_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )


@pytest.mark.cv_unit
@pytest.mark.config
def test_experiment_jsonschema_conversion(_test_inf_spec):
    """Test jsonschema conversion for inference spec."""
    json_with_meta_config = dataclass_to_json(_test_inf_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )


@pytest.mark.cv_unit
@pytest.mark.config
def test_experiment_jsonschema_conversion(_test_eval_spec):
    """Test jsonschema conversion for evaluation spec."""
    json_with_meta_config = dataclass_to_json(_test_eval_spec)
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
    (sample_model_config, ReIDModelConfig),
    (sample_optim_config, OptimConfig),
    (sample_dataset_config, ReIDDatasetConfig),
    (sample_reranking_config, ReIDReRankingConfig),
    (sample_train_config, ReIDTrainExpConfig),
    (sample_inference_config, ReIDInferenceExpConfig),
    (sample_eval_config, ReIDEvalExpConfig),
    (sample_export_config, ReIDExportExpConfig),
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
