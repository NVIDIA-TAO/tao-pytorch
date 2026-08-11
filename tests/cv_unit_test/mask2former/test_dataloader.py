# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pytest
import numpy as np
import tempfile
from PIL import Image
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.mask2former.dataset import AugmentationConfig
from nvidia_tao_pytorch.config.mask2former.default_config import Mask2FormerDatasetConfig
from nvidia_tao_pytorch.cv.mask2former.dataloader.pl_data_module import SemSegmDataModule


TEST_BATCH_SIZE = 4
TEST_WIDTH = 544
TEST_HEIGHT = 960
TEST_OBJ_WIDTH = 80
TEST_OBJ_HEIGHT = 80
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
json_file = os.path.join(tmp_top_dir, "sample_json.json")


@pytest.fixture
def _test_sample_json():
    json_output = {
                    "images": [],
                    "annotations": [],
                    "categories": [ {"supercategory": "person","id": 1, "name": "person"},
                                    {"supercategory": "face","id": 2, "name": "face"},
                                    {"supercategory": "bag","id": 3, "name": "bag"}]
                   }
    os.makedirs(tmp_top_dir, exist_ok=True)
    jsonl_path = os.path.join(tmp_top_dir, "test_ade.jsonl")
    jsonl_records = []
    for image_id in range(0, 10):
        sample_w = int(np.random.randint(low=TEST_WIDTH-20, high=TEST_WIDTH+20, size=1)[0])
        sample_h = int(np.random.randint(low=TEST_HEIGHT-20, high=TEST_HEIGHT+20, size=1)[0])
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_w, sample_h, 3), dtype=np.uint8))
        segm = Image.fromarray(np.random.randint(low=1, high=3, size=(sample_w, sample_h), dtype=np.uint8))
        img_file = os.path.join(tmp_top_dir, f"test_{str(image_id)}.jpg")
        segm_file = os.path.join(tmp_top_dir, f"segm_{str(image_id)}.png")
        img.save(img_file)
        segm.save(segm_file)

        jsonl_records.append({"img": img_file, "segm": segm_file})

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
            x1 = int(np.random.randint(low=100, high=sample_w-TEST_OBJ_WIDTH-100, size=1)[0])
            y1 = int(np.random.randint(low=100, high=sample_h-TEST_OBJ_HEIGHT-100, size=1)[0])
            w = int(np.random.randint(low=10, high=TEST_OBJ_WIDTH, size=1)[0])
            h = int(np.random.randint(low=10, high=TEST_OBJ_HEIGHT, size=1)[0])
            bbox= [x1, y1, w, h]
            x2, y2 = x1 + w, y1 + h
            area  = bbox[2] * bbox[3]
            annotation_info = {
                            'image_id': image_id,
                            'category_id': category_id,
                            'id': annotation_id,
                            'bbox': bbox,
                            'area': area,
                            'iscrowd': 0,
                            'segmentation': [[x1, y1, x1, y2, x2, y2, x2, y1]]
                        }

            json_output["annotations"].append(annotation_info)

    with open(jsonl_path, mode="w", encoding="utf-8") as writer:
        for line in jsonl_records:
            json.dump(line, writer)
            writer.write("\n")

    with open(json_file, 'w+') as outfile:
        json.dump(json_output, outfile)


@pytest.fixture
def _cfg():
    cfg = OmegaConf.structured(Mask2FormerDatasetConfig())
    cfg.contiguous_id = False
    cfg.train.type = 'coco'
    cfg.train.instance_json = os.path.join(tmp_top_dir, "sample_json.json")
    cfg.train.img_dir = tmp_top_dir
    cfg.train.batch_size = 4
    cfg.val.type = 'ade'
    cfg.val.annot_file = os.path.join(tmp_top_dir, "test_ade.jsonl")
    cfg.augmentation = OmegaConf.structured(AugmentationConfig())
    yield cfg


@pytest.mark.cv_unit
@pytest.mark.parametrize("train_min_size", [[320, 640], [200]])
@pytest.mark.parametrize("train_crop_size", [[320, 320], [128, 128]])
@pytest.mark.parametrize("val_batch_size", [2, 1])
@pytest.mark.parametrize("val_target_size", [[640, 640], [321, 321]])
def test_dataloader(_test_sample_json, _cfg, train_min_size, train_crop_size, val_batch_size, val_target_size):
    _cfg.augmentation.train_min_size = train_min_size
    _cfg.augmentation.train_crop_size = train_crop_size
    _cfg.augmentation.test_min_size = 640
    _cfg.val.batch_size = val_batch_size
    _cfg.val.target_size = val_target_size
    dm = SemSegmDataModule(_cfg)
    train_loader = dm.train_dataloader()
    for _, batch in enumerate(train_loader):
        b, c, h, w = batch['images'].shape
        assert b == 4
        assert c == 3
        assert [h, w] == train_crop_size

    val_loader = dm.val_dataloader()
    for _, batch in enumerate(val_loader):
        b, c, h, w = batch['images'].shape
        assert b == val_batch_size
        assert c == 3
        assert [h, w] == [640, 640]
