# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pytest
import numpy as np
import tempfile
from PIL import Image
import cv2
from omegaconf import OmegaConf
from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.config.depth_net.dataset import DepthNetAugmentationConfig, DepthNetDatasetConfig
from nvidia_tao_pytorch.cv.depth_net.dataloader import build_pl_data_module
from nvidia_tao_pytorch.cv.depth_net.dataloader.mono_transforms import build_mono_transforms


TRAIN_BATCH_SIZE = 2
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 2
TEST_WIDTH = 960
TEST_HEIGHT = 540
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
mono_txt_file = os.path.join(tmp_top_dir, "mono_train.txt")


@pytest.fixture
def _mono_test_sample_txt():
    check_and_create(tmp_top_dir)
    with open(mono_txt_file, 'w') as fout:

        for image_id in range(0, TRAIN_BATCH_SIZE*5):
            sample_w = int(np.random.randint(low=TEST_WIDTH-20, high=TEST_WIDTH+20, size=1)[0])
            sample_h = int(np.random.randint(low=TEST_HEIGHT-20, high=TEST_HEIGHT+20, size=1)[0])
            img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_w, sample_h, 3), dtype=np.uint8))
            img_file = os.path.join(tmp_top_dir, f"image_{str(image_id)}.jpg")

            depth = np.random.randint(low=0, high=255, size=(sample_w, sample_h, 1), dtype=np.uint8)
            depth_file = os.path.join(tmp_top_dir, f"depth_{str(image_id)}.png")

            img.save(img_file)
            cv2.imwrite(depth_file, depth)

            print("{} {}".format(img_file, depth_file), file=fout)

    fout.close()

@pytest.fixture
def _aug_spec():
    aug_config = OmegaConf.structured(DepthNetAugmentationConfig())
    yield aug_config

@pytest.fixture
def _dataset_spec(_aug_spec):
    data_config = OmegaConf.structured(DepthNetDatasetConfig())

    data_config.train_dataset.batch_size = TRAIN_BATCH_SIZE
    data_config.train_dataset.workers = 0
    data_config.train_dataset.augmentation = _aug_spec

    data_config.val_dataset.batch_size = VAL_BATCH_SIZE
    data_config.val_dataset.workers = 0
    data_config.val_dataset.augmentation = _aug_spec
   
    data_config.infer_dataset.batch_size = TEST_BATCH_SIZE
    data_config.infer_dataset.workers = 0
    data_config.infer_dataset.augmentation = _aug_spec

    data_config.test_dataset.batch_size = TEST_BATCH_SIZE
    data_config.test_dataset.workers = 0
    data_config.test_dataset.augmentation = _aug_spec
    yield data_config


@pytest.mark.cv_unit
def test_build_transforms(_dataset_spec):
    build_mono_transforms(_dataset_spec.train_dataset.augmentation, split='train', resize_target=True)
    build_mono_transforms(_dataset_spec.val_dataset.augmentation, split='val', resize_target=False)
    build_mono_transforms(_dataset_spec.infer_dataset.augmentation, split='infer', resize_target=False)


@pytest.mark.cv_unit
@pytest.mark.parametrize("dataset_type", ["MonoDataset"])
@pytest.mark.parametrize("model_type", ["RelativeDepthAnything", "MetricDepthAnything"])
def test_train_dataloader(_mono_test_sample_txt, _dataset_spec, dataset_type, model_type):
    _dataset_spec.dataset_name = dataset_type
    if dataset_type == "MonoDataset":
        if model_type == "RelativeDepthAnything":
            _dataset_spec.train_dataset.data_sources = [
                {
                    "dataset_name": "RelativeMonoDataset",
                    "data_file": mono_txt_file
                }
            ]
            _dataset_spec.val_dataset.data_sources = [
                {
                    "dataset_name": "RelativeMonoDataset",
                    "data_file": mono_txt_file
                }
            ]
        elif model_type == "MetricDepthAnything":
            _dataset_spec.train_dataset.data_sources = [
                {
                    "dataset_name": "MetricMonoDataset",
                    "data_file": mono_txt_file
                }
            ]
            _dataset_spec.val_dataset.data_sources = [
                {
                    "dataset_name": "MetricMonoDataset",
                    "data_file": mono_txt_file
                }
            ]
            _dataset_spec.min_depth = 0.001
            _dataset_spec.max_depth = 10
    dm = build_pl_data_module(_dataset_spec)
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    for i, batch in enumerate(train_loader):
        sample = batch
        b, c, height, width = sample['image'].shape
        assert b == TRAIN_BATCH_SIZE, "Incorrect batch size"
        assert c == 3, "Incorrect image channel size"

        if model_type == "RelativeDepthAnything":
            assert "disparity" in sample, "Disparity not in sample"

        if model_type == "MetricDepthAnything":
            assert "depth" in sample, "Depth not in sample"

        if 'disparity' in sample:
            b, c, disp_height, disp_width = sample['disparity'].shape
            assert b == TRAIN_BATCH_SIZE, "Incorrect disparity batch size"
            assert c == 1, "Incorrect disparity channel size"
            assert height == disp_height, "Incorrect disparity height"
            assert width == disp_width, "Incorrect disparity width"

        if 'depth' in sample:
            b, c, depth_height, depth_width = sample['depth'].shape
            assert b == TRAIN_BATCH_SIZE, "Incorrect depth batch size"
            assert c == 1, "Incorrect depth channel size"
            assert height == depth_height, "Incorrect depth height"
            assert width == depth_width, "Incorrect depth width"

    tmp_top_obj.cleanup()

@pytest.mark.cv_unit
@pytest.mark.parametrize("dataset_type", ["MonoDataset"])
@pytest.mark.parametrize("model_type", ["RelativeDepthAnything", "MetricDepthAnything"])
def test_pred_dataloader(_mono_test_sample_txt, _dataset_spec, dataset_type, model_type):
    _dataset_spec.dataset_name = dataset_type
    if dataset_type == "MonoDataset":
        if model_type == "RelativeDepthAnything":
            _dataset_spec.infer_dataset.data_sources = [
                {
                    "dataset_name": "RelativeMonoDataset",
                    "data_file": mono_txt_file
                }
            ]
        elif model_type == "MetricDepthAnything":
            _dataset_spec.infer_dataset.data_sources = [
                {
                    "dataset_name": "MetricMonoDataset",
                    "data_file": mono_txt_file
                }
            ]
            _dataset_spec.min_depth = 0.001
            _dataset_spec.max_depth = 10
    dm = build_pl_data_module(_dataset_spec)
    dm.setup(stage="predict")
    pred_loader = dm.predict_dataloader()
    for i, batch in enumerate(pred_loader):
        sample = batch
        b = len(sample['image'])
        c, height, width = sample['image'][0].shape
        assert c == 3, "Incorrect image channel size"

        if model_type == "RelativeDepthAnything":
            assert "disparity" in sample, "Disparity not in sample"

        if model_type == "MetricDepthAnything":
            assert "depth" in sample, "Depth not in sample"

        if 'disparity' in sample:
            b = len(sample['disparity'])
            c, disp_height, disp_width = sample['disparity'][0].shape
            assert b == TEST_BATCH_SIZE, "Incorrect disparity batch size"
            assert c == 1, "Incorrect disparity channel size"

        if 'depth' in sample:
            b = len(sample['depth'])
            c, depth_height, depth_width = sample['depth'][0].shape
            assert b == TEST_BATCH_SIZE, "Incorrect depth batch size"
            assert c == 1, "Incorrect depth channel size"

    tmp_top_obj.cleanup()
