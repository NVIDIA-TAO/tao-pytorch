# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple test cases to test config load and microservices jsonschema conversion."""

import os
import pytest
import json

from omegaconf import OmegaConf

from nvidia_tao_core.api_utils.dataclass2json_converter import create_json_schema, dataclass_to_json
from nvidia_tao_pytorch.config.visual_changenet.default_config import (
    CNDatasetConfig, 
    CNAugmentationClassifyConfig, 
    DataPathFormat, 
    CNDatasetClassifyConfig, 
    CNDatasetSegmentConfig,
    CNModelConfig, 
    CNModelClassifyConfig, 
    CNOptimConfig,
    ChangeNetHeadConfig, 
    BackboneConfig,
    RandomFlip, 
    RandomRotation, 
    RandomColor, 
    RandomCropWithScale, 
    CNAugmentationSegmentConfig, 
    CNAugmentationClassifyConfig,
    CNTrainClassifyConfig, 
    CNTrainSegmentConfig, 
    CNTrainExpConfig,
    ExperimentConfig
)

sample_dataset_config = """
classify:
    train_dataset:
        csv_path: /data/dataset_convert/train_combined.csv
        images_dir: /data/images/
    validation_dataset:
        csv_path: /data/dataset_convert/valid_combined.csv
        images_dir: /data/images/
    test_dataset:
        csv_path: /data/dataset_convert/valid_combined.csv
        images_dir: /data/images/
    infer_dataset:
        csv_path: /data/dataset_convert/valid_combined.csv
        images_dir: /data/images/
    image_ext: .jpg
    batch_size: 16
    workers: 2
    fpratio_sampling: 0.2
    num_input: 4
    input_map:
        LowAngleLight: 0
        SolderLight: 1
        UniformLight: 2
        WhiteLight: 3
    concat_type: linear
    grid_map:
        x: 2
        y: 2
    image_width: 128
    image_height: 128
    augmentation_config:
        rgb_input_mean: [0.485, 0.456, 0.406]
        rgb_input_std: [0.229, 0.224, 0.225]
        random_flip:
            vflip_probability: 0.5
            hflip_probability: 0.5
            enable: True
        random_rotate:
            rotate_probability: 0.5
            angle_list: [90, 180, 270]
            enable: True
        random_color:
            brightness: 0.3
            contrast: 0.3
            saturation: 0.3
            hue: 0.3
            enable: True
        with_scale_random_crop:
            enable: True
        with_random_crop: True
        with_random_blur: True
        augment: False
    num_classes: 2
"""


sample_dataset_segment_config = """
segment:
    dataset: "CNDataset"
    root_dir: /data/dir
    data_name: "LEVIR-CD"
    label_transform: "norm"
    batch_size: 2
    workers: 2
    multi_scale_train: True
    multi_scale_infer: False
    num_classes: 2
    img_size: 256
    image_folder_name: "A"
    change_image_folder_name: 'B'
    list_folder_name: 'list'
    annotation_folder_name: "label"
    train_split: "train"
    validation_split: "val"
    test_split: 'test'
    predict_split: 'test'
    label_suffix: .png
    augmentation: 
        random_flip:
            vflip_probability: 0.5
            hflip_probability: 0.5
            enable: True
        random_rotate:
            rotate_probability: 0.5
            angle_list: [90, 180, 270]
            enable: True
        random_color:
            brightness: 0.3
            contrast: 0.3
            saturation: 0.3
            hue: 0.3
            enable: True
        with_scale_random_crop:
            enable: True
        with_random_crop: True
        with_random_blur: True
"""

sample_model_config = """
backbone:
    type: "fan_small_12_p4_hybrid"
    pretrained_backbone_path: /path/to/random/dir.file
    freeze_backbone: False
classify:
    train_margin_euclid: 2.0
    eval_margin: 0.005
    embedding_vectors: 5
    embed_dec: 30
    difference_module: 'euclidean'
    learnable_difference_modules: 4
"""

sample_model_segment_config = """
backbone:
    type: "fan_small_12_p4_hybrid"
"""

sample_train_config = """
resume_training_checkpoint_path: /path/to/random/dir.file
pretrained_model_path: /path/to/changenet_classifier.pth
classify:
    loss: "contrastive"
    cls_weight: [1.0, 10.0]
num_epochs: 1
num_nodes: 1
validation_interval: 1
checkpoint_interval: 1
optim:
    lr: 0.00005
    optim: "adamw"
    policy: "linear" 
    momentum: 0.9
    weight_decay: 0.01
results_dir: "/path/to/results_train"
tensorboard:
    enabled: True
    infrequent_logging_frequency: 1
"""

sample_train_segment_config = """
resume_training_checkpoint_path: /path/to/random/dir.file
pretrained_model_path:  /path/to/changenet_segment.pth
segment:
    loss: "ce"
    weights: [0.5, 0.5, 0.5, 0.8, 1.0]
num_epochs: 1
num_nodes: 1
validation_interval: 1
checkpoint_interval: 1
optim:
    lr: 0.0002
    optim: "adamw"
    policy: "linear" 
    momentum: 0.9
    weight_decay: 0.01
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
    ROOT_DIR,"nvidia_tao_pytorch/cv/visual_changenet/experiment_specs"
)
train_config = os.path.join(CONFIG_ROOT, "experiment_spec_classify.yaml")
train_segment_config = os.path.join(CONFIG_ROOT, "experiment_spec.yaml")

with open(train_config, "r") as config_file:
    sample_experiment_config = config_file.read()

with open(train_segment_config, "r") as config_file:
    sample_experiment_segment_config = config_file.read()

@pytest.fixture
def _test_dataset_spec():
    dataset_config = CNDatasetConfig()
    yield dataset_config

@pytest.fixture
def _test_train_spec():
    train_config = CNTrainExpConfig()
    yield train_config

@pytest.fixture
def _test_dataset_classify_spec():
    dataset_classify_config = CNDatasetClassifyConfig()
    yield dataset_classify_config


@pytest.fixture
def _test_dataset_segment_spec():
    dataset_segment_config = CNDatasetSegmentConfig()
    yield dataset_segment_config


@pytest.fixture
def _test_data_path_format_spec():
    data_path_config = DataPathFormat()
    yield data_path_config


@pytest.fixture
def _test_model_spec():
    model_config = CNModelConfig()
    yield model_config


@pytest.fixture
def _test_model_classify_spec():
    model_config = CNModelClassifyConfig()
    yield model_config


@pytest.fixture
def _test_optimizer_spec():
    optimizer_config = CNOptimConfig()
    yield optimizer_config

@pytest.fixture
def _test_experiment_spec():
    experiment_config = ExperimentConfig()
    yield experiment_config


@pytest.fixture
def _test_head_spec():
    head_config = ChangeNetHeadConfig()
    yield head_config


@pytest.fixture
def _test_backbone_spec():
    backbone_config = BackboneConfig()
    yield backbone_config


@pytest.fixture
def _test_augmentation_classify_spec():
    augmentation_classify_config = CNAugmentationClassifyConfig()
    yield augmentation_classify_config


@pytest.fixture
def _test_augmentation_segment_spec():
    augmentation_segment_config = CNAugmentationSegmentConfig()
    yield augmentation_segment_config


@pytest.fixture
def _test_random_crop_spec():
    random_crop_config = RandomCropWithScale()
    yield random_crop_config


@pytest.fixture
def _test_random_color_spec():
    random_color_config = RandomColor()
    yield random_color_config


@pytest.fixture
def _test_random_rotate_spec():
    random_rotate_config = RandomRotation()
    yield random_rotate_config


@pytest.fixture
def _test_random_flip_spec():
    random_flip_config = RandomFlip()
    yield random_flip_config


@pytest.fixture
def _test_train_classify_spec():
    train_classify_config = CNTrainClassifyConfig()
    yield train_classify_config


@pytest.fixture
def _test_train_segment_spec():
    train_segment_config = CNTrainSegmentConfig()
    yield train_segment_config


@pytest.mark.cv_unit
@pytest.mark.config
def test_train_jsonschema_conversion(_test_train_spec):
    """Test jsonschema conversion for train spec."""
    json_with_meta_config = dataclass_to_json(_test_train_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )


@pytest.mark.cv_unit
@pytest.mark.config
def test_train_classify_jsonschema_conversion(_test_train_classify_spec):
    """Test jsonschema conversion for traon-classify spec."""
    json_with_meta_config = dataclass_to_json(_test_train_classify_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )


@pytest.mark.cv_unit
@pytest.mark.config
def test_train_segment_jsonschema_conversion(_test_train_segment_spec):
    """Test jsonschema conversion for train-segment spec."""
    json_with_meta_config = dataclass_to_json(_test_train_segment_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )


@pytest.mark.cv_unit
@pytest.mark.config
def test_augmentation_rf_jsonschema_conversion(_test_random_flip_spec):
    """Test jsonschema conversion for augmentation random flip spec."""
    json_with_meta_config = dataclass_to_json(_test_random_flip_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )


@pytest.mark.cv_unit
@pytest.mark.config
def test_augmentation_rr_jsonschema_conversion(_test_random_rotate_spec):
    """Test jsonschema conversion for augmentation random rotate spec."""
    json_with_meta_config = dataclass_to_json(_test_random_rotate_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )


@pytest.mark.cv_unit
@pytest.mark.config
def test_augmentation_rc_jsonschema_conversion(_test_random_color_spec):
    """Test jsonschema conversion for augmentation random color spec."""
    json_with_meta_config = dataclass_to_json(_test_random_color_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )


@pytest.mark.cv_unit
@pytest.mark.config
def test_augmentation_rcp_jsonschema_conversion(_test_random_crop_spec):
    """Test jsonschema conversion for augmentation random crop spec."""
    json_with_meta_config = dataclass_to_json(_test_random_crop_spec)
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

@pytest.mark.cv_unit
@pytest.mark.config
def test_augmentation_classify_jsonschema_conversion(_test_augmentation_classify_spec):
    """Test jsonschema conversion for augmentation-classify spec."""
    json_with_meta_config = dataclass_to_json(_test_augmentation_classify_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )

@pytest.mark.cv_unit
@pytest.mark.config
def test_augmentation_segment_jsonschema_conversion(_test_augmentation_segment_spec):
    """Test jsonschema conversion for augmentation-segment spec."""
    json_with_meta_config = dataclass_to_json(_test_augmentation_segment_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )

@pytest.mark.cv_unit
@pytest.mark.config
def test_data_path_jsonschema_conversion(_test_data_path_format_spec):
    """Test jsonschema conversion for data path spec."""
    json_with_meta_config = dataclass_to_json(_test_data_path_format_spec)
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
def test_dataset_segment_jsonschema_conversion(_test_dataset_segment_spec):
    """Test jsonschema conversion for dataset_segment spec."""
    json_with_meta_config = dataclass_to_json(_test_dataset_segment_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )


@pytest.mark.cv_unit
@pytest.mark.config
def test_dataset_classify_jsonschema_conversion(_test_dataset_classify_spec):
    """Test jsonschema conversion for dataset-classify spec."""
    json_with_meta_config = dataclass_to_json(_test_dataset_classify_spec)
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
def test_model_classify_jsonschema_config(_test_model_classify_spec):
    """Test jsonschema conversion for model-classify spec."""
    json_with_meta_config = dataclass_to_json(_test_model_classify_spec)
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
    (sample_model_config, CNModelConfig),
    (sample_model_segment_config, CNModelConfig),
    (sample_dataset_config, CNDatasetConfig),
    (sample_dataset_segment_config, CNDatasetConfig),
    (sample_train_config, CNTrainExpConfig),
    (sample_train_segment_config, CNTrainExpConfig),
    (sample_experiment_config, ExperimentConfig),
    (sample_experiment_segment_config, ExperimentConfig)
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
