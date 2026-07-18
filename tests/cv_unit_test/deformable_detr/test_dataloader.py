# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pytest
import numpy as np
import tempfile
from PIL import Image
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.deformable_detr.dataset import DDAugmentationConfig, DDDatasetConfig
from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.cv.deformable_detr.dataloader.pl_od_data_module import ODDataModule
from nvidia_tao_pytorch.cv.deformable_detr.dataloader.transforms import build_transforms


TEST_BATCH_SIZE = 4
TEST_WIDTH = 544
TEST_HEIGHT = 960
TEST_OBJ_WIDTH = 80
TEST_OBJ_HEIGHT = 80
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
json_file = os.path.join(tmp_top_dir, "sample_json.json")
classmap_file = os.path.join(tmp_top_dir, "classmap.txt")

@pytest.fixture
def _test_sample_json():
    json_output = {
                    "images": [],
                    "annotations": [],
                    "categories": [ {"supercategory": "person","id": 1, "name": "person"},
                                    {"supercategory": "face","id": 2, "name": "face"},
                                    {"supercategory": "bag","id": 3, "name": "bag"}]
                   }
    check_and_create(tmp_top_dir)
    for image_id in range(0, 10):
        sample_w = int(np.random.randint(low=TEST_WIDTH-20, high=TEST_WIDTH+20, size=1)[0])
        sample_h = int(np.random.randint(low=TEST_HEIGHT-20, high=TEST_HEIGHT+20, size=1)[0])
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_w, sample_h, 3), dtype=np.uint8))
        img_file = os.path.join(tmp_top_dir, f"test_{str(image_id)}.jpg")

        img.save(img_file)
        images_info = {
            "id": image_id,
            "file_name": f"test_{str(image_id)}.jpg",
            "height": sample_w,
            "width": sample_w
        }
        json_output["images"].append(images_info)

        for cat_id in range(0, 5):
            annotation_id = image_id + cat_id
            category_id = int(np.random.randint(low=1, high=3, size=1)[0])
            x1 = int(np.random.randint(low=0, high=sample_w-TEST_OBJ_WIDTH, size=1)[0])
            y1 = int(np.random.randint(low=0, high=sample_h-TEST_OBJ_HEIGHT, size=1)[0])
            w = int(np.random.randint(low=1, high=TEST_OBJ_WIDTH, size=1)[0])
            h = int(np.random.randint(low=1, high=TEST_OBJ_HEIGHT, size=1)[0])
            bbox= [x1, y1, w, h]
            area  = bbox[2] * bbox[3]
            annotation_info = {
                            'image_id': image_id,
                            'category_id': category_id,
                            'id': annotation_id,
                            'bbox': bbox,
                            'area': area,
                            'iscrowd': 0
                        }

            json_output["annotations"].append(annotation_info)

    with open(json_file, 'w+') as outfile:
        json.dump(json_output, outfile)

    with open(classmap_file, "w") as f:
        for classname in ["person", "face", "bag"]:
            f.write(classname + "\n")


@pytest.fixture
def _dataset_spec():
    data_config = OmegaConf.structured(DDDatasetConfig())
    data_config.train_data_sources = [{"image_dir": tmp_top_dir, "json_file": json_file},
                                      {"image_dir": tmp_top_dir, "json_file": json_file}]
    data_config.val_data_sources = [{"image_dir": tmp_top_dir, "json_file": json_file}]
    data_config.infer_data_sources = {"image_dir": tmp_top_dir, "classmap": classmap_file}
    data_config.batch_size = TEST_BATCH_SIZE
    data_config.workers = 0

    yield data_config

@pytest.fixture
def _aug_spec():
    aug_config = OmegaConf.structured(DDAugmentationConfig())
    yield aug_config

@pytest.mark.cv_unit
@pytest.mark.parametrize("fixed_padding", [True, False])
def test_train_dataloader(_test_sample_json, _dataset_spec, fixed_padding):
    _dataset_spec.augmentation.fixed_padding = fixed_padding
    dm = ODDataModule(_dataset_spec)
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    max_size, min_size = _dataset_spec.augmentation.random_resize_max_size, _dataset_spec.augmentation.test_random_resize
    for _, batch in enumerate(train_loader):
        data, targets, _ = batch
        b, c, height, width = data.shape
        assert b == TEST_BATCH_SIZE, "Incorrect batch size"
        assert c == 4, "Incorrect channel size"

        if fixed_padding:
            assert height in (min_size, max_size), "Incorrect image height"
            assert width in (min_size, max_size), "Incorrect image width"
        else:
            assert height <= max_size, "Incorrect image height"
            assert width <= max_size, "Incorrect image width"

        for target in targets:
            h, w = target["size"]
            assert h <= max_size, "Incorrect label height"
            assert w <= max_size, "Incorrect label width"

@pytest.mark.cv_unit
@pytest.mark.parametrize("fixed_padding", [True, False])
@pytest.mark.parametrize("dataset_type", ["shm", "default"])
def test_pred_dataloader(_test_sample_json, _dataset_spec, fixed_padding, dataset_type):
    _dataset_spec.augmentation.fixed_padding = fixed_padding
    _dataset_spec.dataset_type = dataset_type
    dm = ODDataModule(_dataset_spec)
    dm.setup("predict")
    pred_loader = dm.predict_dataloader()
    total_iter = len(pred_loader)
    total_size = len(dm.pred_dataset)
    max_size, min_size = _dataset_spec.augmentation.random_resize_max_size, _dataset_spec.augmentation.test_random_resize
    for i, batch in enumerate(pred_loader):
        data, _, _ = batch
        b, _, height, width = data.shape
        if i + 1 == total_iter:
            assert b == (total_size % TEST_BATCH_SIZE), "Incorrect handling at the last batch"
        else:
            assert b == TEST_BATCH_SIZE, "Incorrect batch size"
        if fixed_padding:
            assert height <= max_size and height >= min_size, "Incorrect image height"
            assert width <= max_size and width >= min_size, "Incorrect image width"
        else:
            assert height <= max_size, "Incorrect image height"
            assert width <= max_size, "Incorrect image width"

    tmp_top_obj.cleanup()

@pytest.mark.cv_unit
def test_build_transforms(_aug_spec):
    build_transforms(_aug_spec, subtask_config=None, dataset_mode='train')
    build_transforms(_aug_spec, subtask_config=None, dataset_mode='val')
    build_transforms(_aug_spec, subtask_config=None, dataset_mode='eval')
    build_transforms(_aug_spec, subtask_config=None, dataset_mode='infer')
