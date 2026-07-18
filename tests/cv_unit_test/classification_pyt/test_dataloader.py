# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Classification_PL Dataloader Unit Tests
"""
import os
import pytest
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import tempfile

from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.config.classification_pyt.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.pl_classification_data_module import CLDataModule


tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
SAMPLES = 10
BATCH_SIZE = 2
INPUT_SHAPE = 600
OUTPUT_SHAPE = 224
DATASET = 'CLDataset'
NUM_CLASSES = 10

@pytest.fixture
def _test_dir():
    # set this as dataset folder name
    splits = ['train', 'val', 'test']
    img_paths = []

    if not os.path.exists(tmp_top_dir):
        os.makedirs(tmp_top_dir)
    tmp_img_dir = os.path.join(tmp_top_dir)
    check_and_create(tmp_img_dir)

    # write the class.txt to tmp_img_dir, which consists of class names
    class_file = os.path.join(tmp_img_dir, 'classes.txt')
    with open(class_file, 'w') as f:
        for i in range(NUM_CLASSES):
            f.write(str(i) + '\n')

    for split in splits:
        tmp_split_img_dir = os.path.join(tmp_img_dir, split)
        check_and_create(tmp_split_img_dir)
        img_paths.append(tmp_split_img_dir)

    #Input images
    test_data = np.random.rand(INPUT_SHAPE, INPUT_SHAPE, 3) * 255
    test_data = test_data.astype(np.uint8)
    im = Image.fromarray(test_data)

    total_samples = SAMPLES
    for sample in range(total_samples):
        for img_path in img_paths:
            if 'test' in img_path:
                im.save(os.path.join(img_path, str(sample)+'.png'))
            else:
                for class_id in range(NUM_CLASSES):
                    class_dir = os.path.join(img_path, str(class_id))
                    check_and_create(class_dir)
                    # randomly scale the images
                    scale1 = np.random.uniform(0.5, 1.5)
                    scale2 = np.random.uniform(0.5, 1.5)
                    im_resized = im.resize((int(INPUT_SHAPE*scale1), int(INPUT_SHAPE*scale2)))
                    im_resized.save(os.path.join(class_dir, str(sample)+'.png'))

@pytest.fixture
def _test_exp_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config["dataset"]["root_dir"] = tmp_top_dir
    experiment_config["dataset"]["train_dataset"]["images_dir"] = os.path.join(tmp_top_dir, "train")
    experiment_config["dataset"]["val_dataset"]["images_dir"] = os.path.join(tmp_top_dir, "val")
    experiment_config["dataset"]["test_dataset"]["images_dir"] = os.path.join(tmp_top_dir, "test")
    experiment_config["dataset"]["dataset"] = DATASET
    experiment_config["dataset"]["img_size"] = OUTPUT_SHAPE
    experiment_config["dataset"]["batch_size"] = BATCH_SIZE
    experiment_config["dataset"]["num_classes"] = NUM_CLASSES

    experiment_config["results_dir"] = tmp_top_dir


    yield experiment_config


@pytest.mark.parametrize("stage", ['fit', 'test', 'predict'])
@pytest.mark.cv_unit
def test_build_dataloader(_test_exp_spec, stage, _test_dir):

    dm = CLDataModule(_test_exp_spec.dataset)
    dm.setup(stage)
    if stage == 'fit':
        loader = dm.train_dataloader()
    elif stage == 'test':
        loader = dm.test_dataloader()
    elif stage == 'predict':
        loader = dm.predict_dataloader()
    for _, batch in enumerate(loader):
        img = batch['img']
        assert img.shape[0] == BATCH_SIZE, "Incorrect image batch size"
        assert img.shape[2] == _test_exp_spec["dataset"]["img_size"], "Incorrect image height"
        assert img.shape[3] == _test_exp_spec["dataset"]["img_size"], "Incorrect image width"

        if stage == 'fit':
            assert 'class' in batch, "class not present in batch"
            assert batch['class'].shape[0] == BATCH_SIZE, "Incorrect class batch size"
    tmp_top_obj.cleanup()
