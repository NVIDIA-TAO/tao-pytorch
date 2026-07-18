# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import tempfile
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from nvidia_tao_pytorch.config.ocdnet.default_config import OCDNetDataConfig
from nvidia_tao_pytorch.cv.ocdnet.data_loader.build_dataloader import get_dataloader

TEST_SAMPEL = 10
TRAIN_IMG_WIDTH  = 640
TRAIN_IMG_HEIGHT = 640
TEST_IMG_WIDTH  = 1280
TEST_IMG_HEIGHT = 736
TEST_DEFAULT_LABEL = '100,300,800,300,800,500,100,500,NVIDIA'
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name


@pytest.fixture
def _test_icdar_dir():
    if not os.path.isdir(tmp_top_dir):
        os.makedirs(tmp_top_dir, exist_ok=True)
    tmp_img_dir = os.path.join(tmp_top_dir, "img")
    tmp_gt_dir = os.path.join(tmp_top_dir, "gt")
    os.makedirs(tmp_img_dir, exist_ok=True)
    os.makedirs(tmp_gt_dir, exist_ok=True)
    for img_idx in range(0, TEST_SAMPEL):
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(TEST_IMG_HEIGHT, TEST_IMG_WIDTH, 3), dtype=np.uint8))
        img_file = os.path.join(tmp_img_dir, f"test_{str(img_idx)}.jpg")
        img.save(img_file)
        gt_file = os.path.join(tmp_gt_dir, f"gt_test_{str(img_idx)}.txt")
        with open(gt_file,'w') as f:
            f.write(f'{TEST_DEFAULT_LABEL}\n')
    
    yield tmp_top_dir

@pytest.fixture
def _test_dataset_spec():
    dataset_config = OmegaConf.structured(OCDNetDataConfig())
    dataset_config = OmegaConf.to_container(dataset_config)
    dataset_config["train_dataset"]['data_path'] = [tmp_top_dir]
    dataset_config["train_dataset"]['loader']['batch_size'] = 1
    dataset_config["validate_dataset"]['data_path'] = [tmp_top_dir]
    dataset_config["validate_dataset"]['loader']['batch_size'] = 1

    yield dataset_config


@pytest.mark.cv_unit
def test_dataloader(_test_icdar_dir, _test_dataset_spec):

    train_dataloader = get_dataloader(_test_dataset_spec['train_dataset'], False)
    assert len(train_dataloader) == TEST_SAMPEL, "Dummy dataset's length should match default value"
    train_items = next(iter(train_dataloader))
    assert train_items['img'][0,...].shape == (3, TRAIN_IMG_HEIGHT, TRAIN_IMG_WIDTH), "Dummy dataset's image output shape should match default value"

    val_dataloader = get_dataloader(_test_dataset_spec['validate_dataset'], False)
    assert len(val_dataloader) == TEST_SAMPEL, "Dummy dataset's length should match default value"
    val_items = next(iter(val_dataloader))
    assert val_items['img'][0,...].shape == (3, TEST_IMG_HEIGHT, TEST_IMG_WIDTH), "Dummy dataset's image output shape should match default value"
    assert np.asarray(val_items['text_polys'][0]).shape == (1, 4, 2), "Dummy dataset's gt output shape should match default value"
    assert val_items['texts'][0] == ['NVIDIA'], "Dummy dataset's gt output shape should match default value"

    tmp_top_obj.cleanup()
