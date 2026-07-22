# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple test cases to test config load and microservices jsonschema conversion."""

import os
import pytest
import json

from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.stylegan_xl.default_config import DatasetConfig, ModelConfig, TrainExpConfig, ExperimentConfig
from nvidia_tao_core.api_utils.dataclass2json_converter import create_json_schema, dataclass_to_json

sample_dataset_config = """
  common:
    cond: True
    num_classes: 6 # Be 0 when cond==True
    img_channels: 3
    img_resolution: 512
  # stylegan:
  #   train_dataset:
  #     images_dir: /dataset/hyperkvasir_16/hyperkvasir_16_class.zip
  #     # images_dir: /dataset/hyperkvasir_512/hyperkvasir_512.zip
  #   validation_dataset:
  #     images_dir: /dataset/hyperkvasir_16/hyperkvasir_16_class.zip
  #     # images_dir: /dataset/hyperkvasir_512/hyperkvasir_512.zip
  #   test_dataset:
  #     images_dir: /dataset/hyperkvasir_16/hyperkvasir_16_class.zip
  #     # images_dir: /dataset/hyperkvasir_512/hyperkvasir_512.zip
  #   infer_dataset:
  #     start_seed: 0
  #     end_seed: 50
  #   mirror: True
  bigdatasetgan:
    train_dataset:
      images_dir: /out_labeled/train/masks
    validation_dataset:
      images_dir: /out_labeled/train/masks
    test_dataset:
      images_dir: /out_labeled/train/masks
  batch_size: 16
  workers: 3
"""

sample_model_config = """
  input_embeddings_path: /tao-pt/nvidia_tao_pytorch/sdg/stylegan_xl/model/in_embeddings/tf_efficientnet_lite0_embed.pth
  generator:
    backbone: "stylegan3-t"
    superres: True
    added_head_superres: # Ignore this sub section when the superres == False
      head_layers: [4, 4, 4, 4, 4]
      up_factor: [2, 2, 2, 2, 2]
      # pretrained_stem_path: null # BigDatasetGAN does not require stem checkpoint
      reinit_stem_anyway: False
    stem: 
      fp32: False
      cbase: 16384
      cmax: 256
      syn_layers: 7
      resolution: 16
  stylegan:
    loss:
      cls_weight: 0.0
    discriminator:
      backbones: ["deit_base_distilled_patch16_224", "tf_efficientnet_lite0"]
    metrics:
      inception_fid_path: /tao-pt/nvidia_tao_pytorch/sdg/stylegan_xl/model/metrics/InceptionV3.pth
  bigdatasetgan:
    feature_extractor:
      stylegan_checkpoint_path: /tao-pt/gastro_from16_to32_to64_to128_to256_to512/trained_ngc/model_epoch_1099.pth
      blocks: [2, 6, 11, 15]
"""

sample_train_config = """
  resume_training_checkpoint_path: null
  pretrained_model_path: null
  num_epochs: 3000
  num_nodes: 1
  num_gpus: 1
  # gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
  deterministic_all: True
  validation_interval: 1
  checkpoint_interval: 1
  # stylegan:
  #   gan_seed_offset: 0  # Try when encountering GAN mode collapsed
  #   optim_generator:
  #     lr: 0.0025
  #     optim: "Adam"
  #     betas: [0, 0.99]
  #     eps: 1e-08
  #   optim_discriminator:
  #     lr: 0.002
  #     optim: "Adam"
  #     betas: [0, 0.99]
  #     eps: 1e-08
  bigdatasetgan:
    optim_labeller:
      lr: 4e-3
      optim: "AdamW"
      betas: [0.9, 0.95]
  results_dir: "${results_dir}/train"
  tensorboard:
    enabled: True
    infrequent_logging_frequency: 1
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
    ROOT_DIR,"nvidia_tao_pytorch/sdg/stylegan_xl/experiment_specs"
)
train_config = os.path.join(CONFIG_ROOT, "bigdatasetgan.yaml")
with open(train_config, "r") as config_file:
    sample_experiment_config = config_file.read()


@pytest.fixture
def _test_dataset_spec():
    dataset_config = DatasetConfig()
    yield dataset_config


@pytest.fixture
def _test_model_spec():
    model_config = ModelConfig()
    yield model_config


@pytest.fixture
def _test_experiment_spec():
    experiment_config = ExperimentConfig()
    yield experiment_config


@pytest.mark.sdg_unit
@pytest.mark.config
def test_dataset_jsonschema_conversion(_test_dataset_spec):
    """Test jsonschema conversion for augmentation spec."""
    json_with_meta_config = dataclass_to_json(_test_dataset_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )


@pytest.mark.sdg_unit
@pytest.mark.config
def test_model_jsonschema_config(_test_model_spec):
    """Test jsonschema conversion for augmentation spec."""
    json_with_meta_config = dataclass_to_json(_test_model_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )

@pytest.mark.sdg_unit
@pytest.mark.config
def test_experiment_jsonschema_conversion(_test_experiment_spec):
    """Test jsonschema conversion for augmentation spec."""
    json_with_meta_config = dataclass_to_json(_test_experiment_spec)
    json_schema = create_json_schema(json_with_meta_config)
    assert json.dumps(json_schema, indent=4), (
        "Json schema generation failed."
    )


TEST_CONFIG_BLOCKS = [
    (sample_model_config, ModelConfig),
    (sample_dataset_config, DatasetConfig),
    (sample_train_config, TrainExpConfig),
    (sample_experiment_config, ExperimentConfig)
]

@pytest.mark.sdg_unit
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
