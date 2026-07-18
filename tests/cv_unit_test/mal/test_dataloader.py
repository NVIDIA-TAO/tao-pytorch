# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pytest
import numpy as np
import tempfile
from PIL import Image
import json
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.mal.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.mal.datasets.pl_wsi_data_module import WSISDataModule

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
            x1 = int(np.random.randint(low=100, high=sample_w-TEST_OBJ_WIDTH-100, size=1)[0])
            y1 = int(np.random.randint(low=100, high=sample_h-TEST_OBJ_HEIGHT-100, size=1)[0])
            w = int(np.random.randint(low=10, high=TEST_OBJ_WIDTH, size=1)[0])
            h = int(np.random.randint(low=10, high=TEST_OBJ_HEIGHT, size=1)[0])
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


@pytest.fixture
def _cfg():
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.dataset.train_ann_path = os.path.join(tmp_top_dir, "sample_json.json") 
    cfg.dataset.train_img_dir = tmp_top_dir
    cfg.dataset.val_ann_path = os.path.join(tmp_top_dir, "sample_json.json") 
    cfg.dataset.val_img_dir = tmp_top_dir
    cfg.dataset.load_mask = False
    yield cfg


@pytest.mark.cv_unit
def test_dataloader(_test_sample_json, _cfg):
    dm = WSISDataModule(
        num_workers=1,
        cfg=_cfg)
    dm.setup(stage='fit')
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()

    for batch in val_dl:
        assert len(batch) == 13
        assert list(batch['image'].shape)[1:] == [3, 512, 512]
    for batch in train_dl:
        assert len(batch) == 14
        assert list(batch['image'].shape)[1:] == [3, 512, 512]
    tmp_top_obj.cleanup()
