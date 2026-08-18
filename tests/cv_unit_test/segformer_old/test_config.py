# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple test cases to test config load and microservices jsonschema conversion."""

import os
import pytest
import json

from omegaconf import OmegaConf

from nvidia_tao_core.api_utils.dataclass2json_converter import create_json_schema, dataclass_to_json
from nvidia_tao_pytorch.config.segformer_mmlab.default_config import (
    NormConfig,
    TestModelConfig,
    LossDecodeConfig,
    SegformerHeadConfig,
    BackboneConfig,
    SFModelConfig,
    RandomCropCfg,
    ResizeCfg,
    SFAugmentationConfig,
    ImgNormConfig,
    PipelineConfig,
    SFDatasetConfig,
    SFDatasetExpConfig,
    SFEnvConfig,
    SFExpConfig,
    MultiStepLRConfig,
    LRConfig,
    LinearLRConfig,
    PolyLRConfig,
    ParamwiseConfig,
    SFOptimConfig,
    TrainerConfig,
    SFTrainExpConfig,
    SFInferenceExpConfig,
    SFEvalExpConfig,
    TrtConfig,
    OnnxConfig,
    CodebaseConfig,
    SFExportExpConfig,
    GenTrtEngineExpConfig,
    ExperimentConfig,
)

sample_dataset_config = """
input_type: "grayscale"
img_norm_cfg:
    mean:
        - 127.5
        - 127.5
        - 127.5
    std:
        - 127.5
        - 127.5
        - 127.5
    to_rgb: True
data_root: /tlt-pytorch
train_dataset:
    img_dir:
    - /data/images/train
    ann_dir:
    - /data/masks/train
    pipeline:
        augmentation_config:
            random_crop:
                cat_max_ratio: 0.75
            resize:
                ratio_range:
                    - 0.5
                    - 2.0
            random_flip:
                prob: 0.5
val_dataset:
    img_dir: /data/images/val
    ann_dir: /data/masks/val
palette:
    - seg_class: foreground
      rgb:
        - 0
        - 0
        - 0
      label_id: 0
      mapping_class: foreground
    - seg_class: background
      rgb:
        - 255
        - 255
        - 255
      label_id: 1
      mapping_class: background
repeat_data_times: 500
batch_size: 4
workers_per_gpu: 1
"""


sample_model_config = """
input_height: 512
input_width: 512
pretrained_model_path: null
backbone:
    type: "mit_b1"
"""


sample_train_config = """
exp_config:
    manual_seed: 49
checkpoint_interval: 200
logging_interval: 50
max_iters: 1000
resume_training_checkpoint_path: null
validate: True
validation_interval: 500
trainer:
    find_unused_parameters: True
    sf_optim:
        lr: 0.00006
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
    ROOT_DIR,"nvidia_tao_pytorch/cv/segformer_old/experiment_specs/"
)
train_config = os.path.join(CONFIG_ROOT, "train_isbi.yaml")

with open(train_config, "r") as config_file:
    sample_experiment_config = config_file.read()

@pytest.fixture
def _test_norm_config():
    norm_config = NormConfig()
    yield norm_config

@pytest.fixture
def _test_test_model_config():
    test_model_config = TestModelConfig()
    yield test_model_config

@pytest.fixture
def _test_loss_decode_config():
    loss_decode_config = LossDecodeConfig()
    yield loss_decode_config

@pytest.fixture
def _test_segformer_head_config():
    segformer_head_config = SegformerHeadConfig()
    yield segformer_head_config

@pytest.fixture
def _test_backbone_config():
    backbone_config = BackboneConfig()
    yield backbone_config

@pytest.fixture
def _test_sf_model_config():
    sf_model_config = SFModelConfig()
    yield sf_model_config

@pytest.fixture
def _test_random_crop_cfg():
    random_crop_cfg = RandomCropCfg()
    yield random_crop_cfg

@pytest.fixture
def _test_resize_cfg():
    resize_cfg = ResizeCfg()
    yield resize_cfg

@pytest.fixture
def _test_sf_augmentation_config():
    sf_augmentation_config = SFAugmentationConfig()
    yield sf_augmentation_config

@pytest.fixture
def _test_img_norm_config():
    img_norm_config = ImgNormConfig()
    yield img_norm_config

@pytest.fixture
def _test_pipeline_config():
    pipeline_config = PipelineConfig()
    yield pipeline_config

@pytest.fixture
def _test_sf_dataset_config():
    sf_dataset_config = SFDatasetConfig()
    yield sf_dataset_config

@pytest.fixture
def _test_sf_dataset_exp_config():
    sf_dataset_exp_config = SFDatasetExpConfig()
    yield sf_dataset_exp_config

@pytest.fixture
def _test_sf_env_config():
    sf_env_config = SFEnvConfig()
    yield sf_env_config

@pytest.fixture
def _test_sf_exp_config():
    sf_exp_config = SFExpConfig()
    yield sf_exp_config

@pytest.fixture
def _test_multi_step_lr_config():
    multi_step_lr_config = MultiStepLRConfig()
    yield multi_step_lr_config

@pytest.fixture
def _test_lr_config():
    lr_config = LRConfig()
    yield lr_config

@pytest.fixture
def _test_linear_lr_config():
    linear_lr_config = LinearLRConfig()
    yield linear_lr_config

@pytest.fixture
def _test_poly_lr_config():
    poly_lr_config = PolyLRConfig()
    yield poly_lr_config

@pytest.fixture
def _test_paramwise_config():
    paramwise_config = ParamwiseConfig()
    yield paramwise_config

@pytest.fixture
def _test_sf_optim_config():
    sf_optim_config = SFOptimConfig()
    yield sf_optim_config

@pytest.fixture
def _test_trainer_config():
    trainer_config = TrainerConfig()
    yield trainer_config

@pytest.fixture
def _test_sf_train_exp_config():
    sf_train_exp_config = SFTrainExpConfig()
    yield sf_train_exp_config

@pytest.fixture
def _test_sf_inference_exp_config():
    sf_inference_exp_config = SFInferenceExpConfig()
    yield sf_inference_exp_config

@pytest.fixture
def _test_sf_eval_exp_config():
    sf_eval_exp_config = SFEvalExpConfig()
    yield sf_eval_exp_config

@pytest.fixture
def _test_trt_config():
    trt_config = TrtConfig()
    yield trt_config

@pytest.fixture
def _test_onnx_config():
    onnx_config = OnnxConfig()
    yield onnx_config

@pytest.fixture
def _test_codebase_config():
    codebase_config = CodebaseConfig()
    yield codebase_config

@pytest.fixture
def _test_export_config():
    codebase_config = SFExportExpConfig()
    yield codebase_config

@pytest.fixture
def _test_trtengine_config():
    codebase_config = GenTrtEngineExpConfig()
    yield codebase_config

@pytest.fixture
def _test_experiment_config():
    codebase_config = ExperimentConfig()
    yield codebase_config


@pytest.mark.cv_unit
@pytest.mark.config
def test_norm_jsonschema_conversion(_test_norm_config):
    """Test jsonschema conversion for norm config."""
    json_with_meta_config = dataclass_to_json(_test_norm_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_test_model_jsonschema_conversion(_test_test_model_config):
    """Test jsonschema conversion for test model config."""
    json_with_meta_config = dataclass_to_json(_test_test_model_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_loss_decode_jsonschema_conversion(_test_loss_decode_config):
    """Test jsonschema conversion for loss decode config."""
    json_with_meta_config = dataclass_to_json(_test_loss_decode_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_segformer_head_jsonschema_conversion(_test_segformer_head_config):
    """Test jsonschema conversion for segformer head config."""
    json_with_meta_config = dataclass_to_json(_test_segformer_head_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_backbone_jsonschema_conversion(_test_backbone_config):
    """Test jsonschema conversion for backbone config."""
    json_with_meta_config = dataclass_to_json(_test_backbone_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_sf_model_jsonschema_conversion(_test_sf_model_config):
    """Test jsonschema conversion for sf model config."""
    json_with_meta_config = dataclass_to_json(_test_sf_model_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_random_crop_jsonschema_conversion(_test_random_crop_cfg):
    """Test jsonschema conversion for random crop config."""
    json_with_meta_config = dataclass_to_json(_test_random_crop_cfg)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_resize_jsonschema_conversion(_test_resize_cfg):
    """Test jsonschema conversion for resize config."""
    json_with_meta_config = dataclass_to_json(_test_resize_cfg)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_sf_augmentation_jsonschema_conversion(_test_sf_augmentation_config):
    """Test jsonschema conversion for sf augmentation config."""
    json_with_meta_config = dataclass_to_json(_test_sf_augmentation_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_img_norm_jsonschema_conversion(_test_img_norm_config):
    """Test jsonschema conversion for img norm config."""
    json_with_meta_config = dataclass_to_json(_test_img_norm_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_pipeline_jsonschema_conversion(_test_pipeline_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_pipeline_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_sf_dataset_jsonschema_conversion(_test_sf_dataset_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_sf_dataset_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_daatset_exp_jsonschema_conversion(_test_sf_dataset_exp_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_sf_dataset_exp_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_sf_env_jsonschema_conversion(_test_sf_env_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_sf_env_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_sf_exp_jsonschema_conversion(_test_sf_exp_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_sf_exp_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_multi_step_lr_jsonschema_conversion(_test_multi_step_lr_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_multi_step_lr_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_lr_jsonschema_conversion(_test_lr_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_lr_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_linear_lr_jsonschema_conversion(_test_linear_lr_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_linear_lr_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_poly_jsonschema_conversion(_test_poly_lr_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_poly_lr_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_paramwise_jsonschema_conversion(_test_paramwise_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_paramwise_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_sf_optim_jsonschema_conversion(_test_sf_optim_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_sf_optim_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_trainer_jsonschema_conversion(_test_trainer_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_trainer_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_sf_train_jsonschema_conversion(_test_sf_train_exp_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_sf_train_exp_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_infer_jsonschema_conversion(_test_sf_inference_exp_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_sf_inference_exp_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_eval_jsonschema_conversion(_test_sf_eval_exp_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_sf_eval_exp_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_trt_jsonschema_conversion(_test_trt_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_trt_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_onnx_jsonschema_conversion(_test_onnx_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_onnx_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_codebase_jsonschema_conversion(_test_codebase_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_codebase_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_export_jsonschema_conversion(_test_export_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_export_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_trtengine_jsonschema_conversion(_test_trtengine_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_trtengine_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."

@pytest.mark.cv_unit
@pytest.mark.config
def test_experiment_jsonschema_conversion(_test_experiment_config):
    """Test jsonschema conversion for pipeline config."""
    json_with_meta_config = dataclass_to_json(_test_experiment_config)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), "Json schema generation failed."



TEST_CONFIG_BLOCKS = [
    (sample_model_config, SFModelConfig),
    (sample_dataset_config, SFDatasetExpConfig),
    (sample_train_config, SFTrainExpConfig),
    (sample_experiment_config, ExperimentConfig),
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
