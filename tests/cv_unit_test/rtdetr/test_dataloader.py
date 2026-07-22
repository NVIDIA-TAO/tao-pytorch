# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pytest
import numpy as np
import tempfile
from PIL import Image
from omegaconf import OmegaConf
from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.config.rtdetr.dataset import RTAugmentationConfig, RTDatasetConfig
from nvidia_tao_pytorch.cv.rtdetr.dataloader.pl_od_data_module import ODDataModule
from nvidia_tao_pytorch.cv.rtdetr.dataloader.transforms import build_transforms


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
    data_config = OmegaConf.structured(RTDatasetConfig())
    data_config.train_data_sources = [{"image_dir": tmp_top_dir, "json_file": json_file},
                                      {"image_dir": tmp_top_dir, "json_file": json_file}]
    data_config.val_data_sources = {"image_dir": tmp_top_dir, "json_file": json_file}
    data_config.infer_data_sources = {"image_dir": tmp_top_dir, "classmap": classmap_file}
    data_config.batch_size = TEST_BATCH_SIZE
    data_config.workers = 0

    yield data_config

@pytest.fixture
def _aug_spec():
    aug_config = OmegaConf.structured(RTAugmentationConfig())
    yield aug_config

@pytest.mark.cv_unit
@pytest.mark.parametrize("remap_mscoco_category", [True, False])
def test_train_dataloader(_test_sample_json, _dataset_spec, remap_mscoco_category):
    _dataset_spec.remap_mscoco_category = remap_mscoco_category
    dm = ODDataModule(_dataset_spec)
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    height_target, width_target = tuple(_dataset_spec.augmentation.train_spatial_size)
    for _, batch in enumerate(train_loader):
        data, targets, _ = batch
        b, c, height, width = data.shape
        assert b == TEST_BATCH_SIZE, "Incorrect batch size"
        assert c == 3, "Incorrect channel size"

        assert height <= height_target, f"Incorrect image height {height} {height_target}"
        assert width <= width_target, f"Incorrect image width {width} {width_target}"

        # Check
        if remap_mscoco_category:
            for target in targets:
                if len(target['labels']) == 0:
                    # Random augmentation can crop out all objects; an empty
                    # labels tensor trivially satisfies the MSCOCO ID range.
                    continue
                assert max(target['labels']) <= 81, f"{target['labels']}"



@pytest.mark.cv_unit
@pytest.mark.parametrize("dataset_type", ["shm", "default"])
@pytest.mark.parametrize("preserve_aspect_ratio", [False, True])
def test_pred_dataloader(_test_sample_json, _dataset_spec, dataset_type, preserve_aspect_ratio):
    _dataset_spec.dataset_type = dataset_type
    _dataset_spec.augmentation.preserve_aspect_ratio = preserve_aspect_ratio
    dm = ODDataModule(_dataset_spec)
    dm.setup("predict")
    pred_loader = dm.predict_dataloader()
    total_iter = len(pred_loader)
    total_size = len(dm.pred_dataset)
    height_target, width_target = tuple(_dataset_spec.augmentation.eval_spatial_size)
    for i, batch in enumerate(pred_loader):
        data, targets, _ = batch
        b, _, height, width = data.shape
        if i + 1 == total_iter:
            assert b == (total_size % TEST_BATCH_SIZE), "Incorrect handling at the last batch"
        else:
            assert b == TEST_BATCH_SIZE, "Incorrect batch size"

        assert height <= height_target, "Incorrect image height"
        assert width <= width_target, "Incorrect image width"

        for target in targets:
            if preserve_aspect_ratio:
                assert target["orig_size"][0] == height_target, f"{target['orig_size'][0]} {height_target}"
                assert target["orig_size"][1] == width_target, f"{target['orig_size'][1]} {width_target}"

    tmp_top_obj.cleanup()

@pytest.mark.cv_unit
def test_build_transforms(_aug_spec):
    build_transforms(_aug_spec, subtask_config=None, dataset_mode='train')
    build_transforms(_aug_spec, subtask_config=None, dataset_mode='val')
    build_transforms(_aug_spec, subtask_config=None, dataset_mode='eval')
    build_transforms(_aug_spec, subtask_config=None, dataset_mode='infer')
