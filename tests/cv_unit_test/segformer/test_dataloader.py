# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SegFormer_PL Dataloader Unit Tests
"""
import os
import pytest
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import tempfile

from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.config.segformer.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.segformer.dataloader.pl_segformer_data_module import SFDataModule


tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
SAMPLES = 10
BATCH_SIZE = 2
INPUT_SHAPE = 512
OUTPUT_SHAPE = 224
LABEL_TRANSFORM = 'norm'
DATASET = 'SFDataset'

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
def _test_exp_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config["dataset"]['segment']["root_dir"] = tmp_top_dir
    experiment_config["dataset"]['segment']["label_transform"] = LABEL_TRANSFORM
    experiment_config["dataset"]['segment']["dataset"] = DATASET
    experiment_config["dataset"]['segment']["img_size"] = OUTPUT_SHAPE
    experiment_config["dataset"]['segment']["batch_size"] = BATCH_SIZE

    experiment_config["results_dir"] = tmp_top_dir


    yield experiment_config


@pytest.mark.parametrize("stage", ['fit', 'test', 'predict'])
@pytest.mark.cv_unit
def test_build_dataloader(_test_exp_spec, stage, _test_dir):

    dm = SFDataModule(_test_exp_spec.dataset.segment)
    dm.setup(stage)
    if stage == 'fit':
        loader = dm.train_dataloader()
    elif stage == 'test':
        loader = dm.test_dataloader()
    elif stage == 'predict':
        loader = dm.predict_dataloader()
    for _, batch in enumerate(loader):
        img = batch['img']
        assert img.shape[0] in (1, BATCH_SIZE), "Incorrect image batch size"
        assert img.shape[2] == _test_exp_spec["dataset"]['segment']["img_size"], "Incorrect image height"
        assert img.shape[3] == _test_exp_spec["dataset"]['segment']["img_size"], "Incorrect image width"
        if stage == 'fit':
            mask = batch['mask']
            print(mask.shape)
            assert mask.shape[0] == BATCH_SIZE, "Incorrect mask batch size"
            assert mask.shape[-2] == _test_exp_spec["dataset"]['segment']["img_size"], "Incorrect mask height"
            assert mask.shape[-1] == _test_exp_spec["dataset"]['segment']["img_size"], "Incorrect mask width"

    tmp_top_obj.cleanup()
