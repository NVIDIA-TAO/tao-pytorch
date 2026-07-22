# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SegFormer_PL Trainer Unit Tests
"""
import os
import torch
import pytest
import tempfile
import numpy as np
import pandas as pd
from PIL import Image

from omegaconf import OmegaConf

from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.config.segformer.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.segformer.dataloader.pl_segformer_data_module import SFDataModule
from nvidia_tao_pytorch.cv.segformer.model.segformer_pl_model import SegFormerPlModel
from pytorch_lightning import Trainer


FAST_DEV_RUN = 2
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
SAMPLES = 10
BATCH_SIZE = 2
INPUT_SHAPE = 512
OUTPUT_SHAPE = 224
LABEL_TRANSFORM = 'norm'
DATASET = 'SFDataset'
TEST_TOPOLOGIES = [
    # ConvNeXtV2.
    ("mit_b0"),
    # DINOV2.
    ("vit_large_nvdinov2"),
    # FAN.
    ("fan_tiny_8_p4_hybrid"),
    # OpenCLIP.
    ("vit_base_nvclip_16_siglip"),
    ("vit_huge_nvclip_14_siglip"),
    # RADIO.
    ("c_radio_v2_vit_base_patch16_224"),
]


@pytest.fixture
def _test_dir():
    splits = ['train', 'val', 'test']
    img_paths = []
    mask_paths = []

    if not os.path.exists(tmp_top_dir):
        os.makedirs(tmp_top_dir)
    tmp_img_dir = os.path.join(tmp_top_dir, "images")
    tmp_mask_dir = os.path.join(tmp_top_dir, "masks")
    check_and_create(tmp_img_dir)
    check_and_create(tmp_mask_dir)

    for split in splits:
        tmp_split_img_dir = os.path.join(tmp_img_dir, split)
        tmp_split_mask_dir = os.path.join(tmp_mask_dir, split)
        check_and_create(tmp_split_img_dir)
        img_paths.append(tmp_split_img_dir)
        if split != 'test':
            check_and_create(tmp_split_mask_dir)
            mask_paths.append(tmp_split_mask_dir)

    #Input images
    test_data = np.random.rand(INPUT_SHAPE, INPUT_SHAPE, 3) * 255
    test_data = test_data.astype(np.uint8)
    im = Image.fromarray(test_data)
    #GT Label image
    label_data = np.zeros((INPUT_SHAPE, INPUT_SHAPE))
    label_data = label_data.astype(np.uint8)
    im_label = Image.fromarray(label_data)

    total_samples = SAMPLES
    splits = ['train', 'val', 'test']
    for sample in range(total_samples):
        for img_path in img_paths:
            im.save(os.path.join(img_path, str(sample)+'.png'))
        for mask_path in mask_paths:
            im_label.save(os.path.join(mask_path, str(sample)+'.png'))


@pytest.fixture
def _train_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset.segment.root_dir = tmp_top_dir
    experiment_config.dataset.segment.label_transform = LABEL_TRANSFORM
    experiment_config.dataset.segment.img_size = OUTPUT_SHAPE
    experiment_config.dataset.segment.dataset = DATASET
    experiment_config.dataset.segment.batch_size = BATCH_SIZE
    experiment_config.dataset.segment.num_classes = 2
    experiment_config.results_dir = tmp_top_dir

    experiment_config.train.num_epochs = 1
    experiment_config.train.num_gpus = 1
    experiment_config.train.num_nodes = 1

    yield experiment_config

# ("fan_tiny_8_p4_hybrid"),
# ("fan_large_16_p4_hybrid"),
# ("fan_small_12_p4_hybrid"),
# ("fan_base_16_p4_hybrid"),

@pytest.mark.cv_unit
@pytest.mark.segformer
@pytest.mark.train
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
def test_trainer_fit(_test_dir, _train_spec, backbone):

    _train_spec.model.backbone.type = backbone
    if 'vit' in backbone:
        _train_spec.model.decode_head.feature_strides = [4, 8, 16, 32]
        _train_spec.dataset.segment.batch_size = 1

    dm = SFDataModule(_train_spec.dataset.segment)
    dm.setup(stage="fit")
    model = SegFormerPlModel(_train_spec)

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                    num_nodes=_train_spec.train.num_nodes,
                    default_root_dir=_train_spec.results_dir,
                    accelerator='gpu',
                    strategy='auto',
                    precision='32-true',
                    fast_dev_run=FAST_DEV_RUN)

    # Test train
    trainer.fit(model, dm)


@pytest.mark.cv_unit
@pytest.mark.segformer
@pytest.mark.evaluate
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
def test_trainer_evaluate(_test_dir, _train_spec, backbone):

    _train_spec.model.backbone.type = backbone
    if 'vit' in backbone:
        _train_spec.model.decode_head.feature_strides = [4, 8, 16, 32]
        _train_spec.dataset.segment.batch_size = 1

    dm = SFDataModule(_train_spec.dataset.segment)
    dm.setup(stage="test")
    model = SegFormerPlModel(_train_spec)

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                      default_root_dir=_train_spec.results_dir,
                      accelerator='auto',
                      fast_dev_run=FAST_DEV_RUN)

    trainer.test(model, datamodule=dm)


@pytest.mark.cv_unit
@pytest.mark.segformer
@pytest.mark.inference
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
def test_trainer_infer(_test_dir, _train_spec, backbone):

    _train_spec.model.backbone.type = backbone
    if 'vit' in backbone:
        _train_spec.model.decode_head.feature_strides = [4, 8, 16, 32]
        _train_spec.dataset.segment.batch_size = 1

    dm = SFDataModule(_train_spec.dataset.segment)
    dm.setup(stage="predict")
    model = SegFormerPlModel(_train_spec)

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                      default_root_dir=_train_spec.results_dir,
                      accelerator='auto',
                      fast_dev_run=FAST_DEV_RUN)

    trainer.predict(model, datamodule=dm)

    tmp_top_obj.cleanup()
