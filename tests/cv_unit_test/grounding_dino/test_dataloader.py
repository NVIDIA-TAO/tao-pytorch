# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import jsonlines
import os
import random
import pytest
import numpy as np
import tempfile
from PIL import Image
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.grounding_dino.dataset import GDINOAugmentationConfig, GDINODatasetConfig
from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.cv.grounding_dino.dataloader.pl_odvg_data_module import ODVGDataModule
from nvidia_tao_pytorch.cv.deformable_detr.dataloader.transforms import build_transforms


TEST_BATCH_SIZE = 4
TEST_WIDTH = 544
TEST_HEIGHT = 960
TEST_OBJ_WIDTH = 80
TEST_OBJ_HEIGHT = 80
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
detection_json_file = os.path.join(tmp_top_dir, "detection_json.jsonl")
grounding_json_file = os.path.join(tmp_top_dir, "grounding_json.jsonl")
json_file = os.path.join(tmp_top_dir, "sample_json.json")
classmap_file = os.path.join(tmp_top_dir, "classmap.json")
classtxt_file = os.path.join(tmp_top_dir, "classmap.txt")
classlist = ["person", "face", "bag"]
classmap = {i: c for i, c in enumerate(classlist)}


@pytest.fixture
def _test_sample_json():
    json_output = {
                    "images": [],
                    "annotations": [],
                    "categories": [ {"supercategory": "person","id": 1, "name": "person"},
                                    {"supercategory": "face","id": 2, "name": "face"},
                                    {"supercategory": "bag","id": 3, "name": "bag"}]
                   }
    coco_dir = os.path.join(tmp_top_dir, "coco")
    check_and_create(tmp_top_dir)
    check_and_create(coco_dir)
    for image_id in range(0, 10):
        sample_w = int(np.random.randint(low=TEST_WIDTH-20, high=TEST_WIDTH+20, size=1)[0])
        sample_h = int(np.random.randint(low=TEST_HEIGHT-20, high=TEST_HEIGHT+20, size=1)[0])
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_w, sample_h, 3), dtype=np.uint8))
        img_file = os.path.join(coco_dir, f"test_{str(image_id)}.jpg")

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

    with open(classtxt_file, "w") as f:
        for classname in classlist:
            f.write(classname + "\n")


@pytest.fixture
def _test_detection_jsonl():

    jsonl_outputs = []
    detection_dir = os.path.join(tmp_top_dir, "detection")
    check_and_create(tmp_top_dir)
    check_and_create(detection_dir)

    for image_id in range(0, 10):
        sample_w = int(np.random.randint(low=TEST_WIDTH-20, high=TEST_WIDTH+20, size=1)[0])
        sample_h = int(np.random.randint(low=TEST_HEIGHT-20, high=TEST_HEIGHT+20, size=1)[0])
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_w, sample_h, 3), dtype=np.uint8))
        img_file = os.path.join(detection_dir, f"test_{str(image_id)}.jpg")

        img.save(img_file)
        json_output = {
            "file_name": f"test_{str(image_id)}.jpg",
            "height": sample_w,
            "width": sample_w
        }
        instances = []
        for _ in range(0, 5):
            x1 = int(np.random.randint(low=0, high=sample_w-TEST_OBJ_WIDTH, size=1)[0])
            y1 = int(np.random.randint(low=0, high=sample_h-TEST_OBJ_HEIGHT, size=1)[0])
            w = int(np.random.randint(low=1, high=TEST_OBJ_WIDTH, size=1)[0])
            h = int(np.random.randint(low=1, high=TEST_OBJ_HEIGHT, size=1)[0])
            x2, y2 = x1 + w, y1 + h
            categoy_id = random.choice(list(classmap.keys()))
            instances.append({
                "bbox": [x1, y1, x2, y2],
                "label": categoy_id,
                "category": classmap[categoy_id]
            })
        json_output["detection"] = {"instances": instances}
        jsonl_outputs.append(json_output)

    with jsonlines.open(detection_json_file, 'w') as outfile:
        outfile.write_all(jsonl_outputs)

    with open(classmap_file, "w") as f:
        json.dump(classmap, f)


@pytest.fixture
def _test_grounding_jsonl():

    sentence = "Two people are talking outside of the video game shop next door to the mobile phone store."
    phrases = ["Two people", "the mobile phone store", "the video game shop"]
    jsonl_outputs = []
    grounding_dir = os.path.join(tmp_top_dir, "grounding")
    check_and_create(tmp_top_dir)
    check_and_create(grounding_dir)

    for image_id in range(0, 10):
        sample_w = int(np.random.randint(low=TEST_WIDTH-20, high=TEST_WIDTH+20, size=1)[0])
        sample_h = int(np.random.randint(low=TEST_HEIGHT-20, high=TEST_HEIGHT+20, size=1)[0])
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_w, sample_h, 3), dtype=np.uint8))
        img_file = os.path.join(grounding_dir, f"test_{str(image_id)}.jpg")

        img.save(img_file)
        json_output = {
            "file_name": f"test_{str(image_id)}.jpg",
            "height": sample_w,
            "width": sample_w
        }
        regions = []
        for _ in range(0, 5):
            x1 = int(np.random.randint(low=0, high=sample_w-TEST_OBJ_WIDTH, size=1)[0])
            y1 = int(np.random.randint(low=0, high=sample_h-TEST_OBJ_HEIGHT, size=1)[0])
            w = int(np.random.randint(low=1, high=TEST_OBJ_WIDTH, size=1)[0])
            h = int(np.random.randint(low=1, high=TEST_OBJ_HEIGHT, size=1)[0])
            x2, y2 = x1 + w, y1 + h
            phrase = random.choice(phrases)
            regions.append({
                "bbox": [x1, y1, x2, y2],
                "phrase": phrase,
            })
        json_output["grounding"] = {"caption": sentence, "regions": regions}
        jsonl_outputs.append(json_output)

    with jsonlines.open(grounding_json_file, 'w') as outfile:
        outfile.write_all(jsonl_outputs)


@pytest.fixture
def _dataset_spec():
    data_config = OmegaConf.structured(GDINODatasetConfig())
    data_config.train_data_sources = [{"image_dir": os.path.join(tmp_top_dir, "detection"), 
                                       "json_file": detection_json_file,
                                       "label_map": classmap_file},
                                      {"image_dir": os.path.join(tmp_top_dir, "grounding"),
                                       "json_file": grounding_json_file},
                                      ]
    data_config.val_data_sources = {"image_dir": os.path.join(tmp_top_dir, "coco"), "json_file": json_file}
    data_config.infer_data_sources = {"image_dir": os.path.join(tmp_top_dir, "coco"), "captions": str(classlist)}
    data_config.batch_size = TEST_BATCH_SIZE
    data_config.workers = 0

    yield data_config

@pytest.fixture
def _aug_spec():
    aug_config = OmegaConf.structured(GDINOAugmentationConfig())
    yield aug_config

@pytest.mark.cv_unit
@pytest.mark.parametrize("fixed_padding", [True, False])
def test_train_dataloader(_test_detection_jsonl, _test_grounding_jsonl, _test_sample_json, _dataset_spec, fixed_padding):
    _dataset_spec.augmentation.fixed_padding = fixed_padding
    dm = ODVGDataModule(_dataset_spec)
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    max_size, min_size = _dataset_spec.augmentation.random_resize_max_size, _dataset_spec.augmentation.test_random_resize
    for _, batch in enumerate(train_loader):
        data, targets = batch
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
def test_pred_dataloader(_test_sample_json, _dataset_spec, fixed_padding):
    _dataset_spec.augmentation.fixed_padding = fixed_padding
    dm = ODVGDataModule(_dataset_spec)
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
