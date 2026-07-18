# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Visual ChangeNet-Segmentation Trainer Unit Tests
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
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.build_data_loader import build_dataloader
from nvidia_tao_pytorch.cv.optical_inspection.model.build_nn_model import AOIMetrics
from nvidia_tao_pytorch.config.visual_changenet.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.dataloader.pl_changenet_data_module import CNDataModule
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models.cn_pl_model import ChangeNetPlModel as ChangeNetPlSegment
from pytorch_lightning import Trainer


FAST_DEV_RUN = 2
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
SAMPLES = 20
BATCH_SIZE = 2
OUTPUT_SHAPE = 224
LABEL_TRANSFORM = 'norm'
DATASET = 'CNDataset'
IMG_FOLDER_NAME = 'A'
CHANGE_FOLDER_NAME = 'B'
LABEL_FOLDER_NAME = 'label'
LIST_FOLDER_NAME = 'list'
NUM_CLASSES = 2


@pytest.fixture
def _test_dir():

    if not os.path.exists(tmp_top_dir):
        os.makedirs(tmp_top_dir)
    tmp_test_dir = os.path.join(tmp_top_dir, IMG_FOLDER_NAME)
    tmp_golden_dir = os.path.join(tmp_top_dir, CHANGE_FOLDER_NAME)
    tmp_list_dir = os.path.join(tmp_top_dir, LIST_FOLDER_NAME)
    tmp_label_dir = os.path.join(tmp_top_dir, LABEL_FOLDER_NAME)

    check_and_create(tmp_test_dir)
    check_and_create(tmp_golden_dir)
    check_and_create(tmp_list_dir)
    check_and_create(tmp_label_dir)

    #Input images
    test_data = np.random.rand(OUTPUT_SHAPE, OUTPUT_SHAPE, 3) * 255
    test_data = test_data.astype(np.uint8)
    im = Image.fromarray(test_data)
    #GT Label image
    label_data = np.zeros((OUTPUT_SHAPE, OUTPUT_SHAPE))
    label_data = label_data.astype(np.uint8)
    im_label = Image.fromarray(label_data)

    total_samples = SAMPLES
    txt_data = []
    for sample in range(total_samples):
        im.save(os.path.join(tmp_test_dir, str(sample)+'.png'))
        im.save(os.path.join(tmp_golden_dir, str(sample)+'.png'))
        im_label.save(os.path.join(tmp_label_dir, str(sample)+'.png'))
        txt_data.append(str(sample)+'.png')

    splits = ['train', 'val', 'test']
    for split in splits:
        txt_file_name = os.path.join(tmp_list_dir, split+'.txt')
        with open (txt_file_name ,'w') as f:
            for data in txt_data:
                f.write(f"{data}\n")


@pytest.fixture
def _train_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset.segment.root_dir = tmp_top_dir
    experiment_config.dataset.segment.label_transform = LABEL_TRANSFORM
    experiment_config.dataset.segment.img_size = OUTPUT_SHAPE
    experiment_config.dataset.segment.dataset = DATASET
    experiment_config.dataset.segment.image_folder_name = IMG_FOLDER_NAME
    experiment_config.dataset.segment.change_image_folder_name = CHANGE_FOLDER_NAME
    experiment_config.dataset.segment.list_folder_name = LIST_FOLDER_NAME
    experiment_config.dataset.segment.annotation_folder_name = LABEL_FOLDER_NAME
    experiment_config.dataset.segment.label_suffix = '.png'
    experiment_config.dataset.segment.batch_size = BATCH_SIZE
    experiment_config.dataset.segment.num_classes = NUM_CLASSES
    experiment_config.results_dir = tmp_top_dir

    experiment_config.train.num_epochs = 1
    experiment_config.train.num_gpus = 1
    experiment_config.train.num_nodes = 1

    yield experiment_config


TEST_TOPOLOGIES = [
    ("fan_tiny_8_p4_hybrid"),
    ("vit_large_nvdinov2"),
    ("c_radio_v2_vit_base_patch16_224"),
]
if not os.getenv("CI_PROJECT_DIR", None):
    TEST_TOPOLOGIES.extend([
        ("c_radio_p3_vit_huge_patch16_224_mlpnorm"),
    ])


@pytest.mark.cv_unit
@pytest.mark.visual_changenet_segment
@pytest.mark.train
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
@pytest.mark.parametrize("task", ['segment'])
def test_trainer_fit(_test_dir, _train_spec, backbone, task):

    _train_spec.model.backbone.type = backbone
    _train_spec.task = task
    if 'vit' in backbone:
        _train_spec.model.decode_head.feature_strides = [4, 8, 16, 32]
        _train_spec.dataset.segment.batch_size = 1

    dm = CNDataModule(_train_spec.dataset.segment)
    dm.setup(stage="fit")
    model = ChangeNetPlSegment(_train_spec)

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
@pytest.mark.visual_changenet_segment
@pytest.mark.evaluate
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
@pytest.mark.parametrize("task", ['segment'])
def test_trainer_evaluate(_test_dir, _train_spec, backbone, task):

    _train_spec.model.backbone.type = backbone
    _train_spec.task = task
    if 'vit' in backbone:
        _train_spec.model.decode_head.feature_strides = [4, 8, 16, 32]
        _train_spec.dataset.segment.batch_size = 1

    dm = CNDataModule(_train_spec.dataset.segment)
    dm.setup(stage="test")
    model = ChangeNetPlSegment(_train_spec)

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                      default_root_dir=_train_spec.results_dir,
                      accelerator='auto',
                      fast_dev_run=FAST_DEV_RUN)

    trainer.test(model, datamodule=dm)


@pytest.mark.cv_unit
@pytest.mark.visual_changenet_segment
@pytest.mark.inference
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
@pytest.mark.parametrize("task", ['segment'])
def test_trainer_infer(_test_dir, _train_spec, backbone, task):

    _train_spec.model.backbone.type = backbone
    _train_spec.task = task
    if 'vit' in backbone:
        _train_spec.model.decode_head.feature_strides = [4, 8, 16, 32]
        _train_spec.dataset.segment.batch_size = 1

    dm = CNDataModule(_train_spec.dataset.segment)
    dm.setup(stage="predict")
    model = ChangeNetPlSegment(_train_spec)

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                      default_root_dir=_train_spec.results_dir,
                      accelerator='auto',
                      fast_dev_run=FAST_DEV_RUN)

    trainer.predict(model, datamodule=dm)

    tmp_top_obj.cleanup()
