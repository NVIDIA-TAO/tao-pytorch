# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import jsonlines
import os
import random
import pytest
import tempfile
import numpy as np
from PIL import Image
from pytorch_lightning import Trainer

from omegaconf import OmegaConf
from nvidia_tao_pytorch.config.mask_grounding_dino.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.mask_grounding_dino.dataloader.od_data_module import ODVGDataModule
from nvidia_tao_pytorch.cv.mask_grounding_dino.model.pl_gdino_model import MaskGDINOPlModel
from nvidia_tao_pytorch.core.utilities import check_and_create


TEST_SIZE_MIN = 960
TEST_SIZE_MAX = 1024
TEST_OBJ_WIDTH = 80
TEST_OBJ_HEIGHT = 80
FAST_DEV_RUN = 2  # Run dry run 2 times
AUGMENT_SIZE_CROP = None
AUGMENT_TEST_SIZE = 800
AUGMENT_MAX_SIZE = 1333
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
tmp_img = Image.fromarray(np.random.randint(low=0, high=255, size=(TEST_SIZE_MIN, TEST_SIZE_MAX, 3), dtype=np.uint8))
img_file = os.path.join(tmp_top_dir, f"test.jpg")
tmp_img.save(img_file)
detection_json_file = os.path.join(tmp_top_dir, "detection_json.jsonl")
grounding_expression_json_file = os.path.join(tmp_top_dir, "grounding_expression_json.jsonl")
grounding_phrase_json_file = os.path.join(tmp_top_dir, "grounding_phrase_json.jsonl")
pred_grounding_json_file = os.path.join(tmp_top_dir, "pred_grounding_json.jsonl")
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
        min_size = int(np.random.randint(low=TEST_SIZE_MIN-20, high=TEST_SIZE_MIN+20, size=1)[0])
        max_size = int(np.random.randint(low=TEST_SIZE_MAX-20, high=TEST_SIZE_MAX+20, size=1)[0])
        if random.random() < 0.5:
            sample_w = min_size
            sample_h = max_size
        else:
            sample_w = max_size
            sample_h = min_size
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_h, sample_w, 3), dtype=np.uint8))
        img_file = os.path.join(coco_dir, f"test_{str(image_id)}.jpg")

        img.save(img_file)
        images_info = {
            "id": image_id,
            "file_name": f"test_{str(image_id)}.jpg",
            "height": sample_h,
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
            segmentation = [[
                x1, y1,
                x1, y1 + h,
                x1 + w, y1 + h,
                x1 + w, y1
            ]]
            area  = bbox[2] * bbox[3]
            annotation_info = {
                            'image_id': image_id,
                            'category_id': category_id,
                            'id': annotation_id,
                            'bbox': bbox,
                            'area': area,
                            'iscrowd': 0,
                            'segmentation': segmentation
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
        min_size = int(np.random.randint(low=TEST_SIZE_MIN-20, high=TEST_SIZE_MIN+20, size=1)[0])
        max_size = int(np.random.randint(low=TEST_SIZE_MAX-20, high=TEST_SIZE_MAX+20, size=1)[0])
        if random.random() < 0.5:
            sample_w = min_size
            sample_h = max_size
        else:
            sample_w = max_size
            sample_h = min_size
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_h, sample_w, 3), dtype=np.uint8))
        img_file = os.path.join(detection_dir, f"test_{str(image_id)}.jpg")

        img.save(img_file)
        json_output = {
            "file_name": f"test_{str(image_id)}.jpg",
            "height": sample_h,
            "width": sample_w,
            "image_id": image_id
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
                "category": classmap[categoy_id],
                "mask": [[x1, y1, x1, y2, x2, y2, x2, y1]]
            })
        json_output["detection"] = {"instances": instances}
        jsonl_outputs.append(json_output)

    with jsonlines.open(detection_json_file, 'w') as outfile:
        outfile.write_all(jsonl_outputs)

    with open(classmap_file, "w") as f:
        json.dump(classmap, f)


@pytest.fixture
def _test_grounding_expression_jsonl():

    sentence = "Two people are talking outside of the video game shop next door to the mobile phone store."
    phrases = ["Two people", "the mobile phone store", "the video game shop"]
    jsonl_outputs = []
    grounding_dir = os.path.join(tmp_top_dir, "grounding_expression")
    check_and_create(tmp_top_dir)
    check_and_create(grounding_dir)

    for image_id in range(0, 10):
        min_size = int(np.random.randint(low=TEST_SIZE_MIN-20, high=TEST_SIZE_MIN+20, size=1)[0])
        max_size = int(np.random.randint(low=TEST_SIZE_MAX-20, high=TEST_SIZE_MAX+20, size=1)[0])
        if random.random() < 0.5:
            sample_w = min_size
            sample_h = max_size
        else:
            sample_w = max_size
            sample_h = min_size
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_h, sample_w, 3), dtype=np.uint8))
        img_file = os.path.join(grounding_dir, f"test_expression_{str(image_id)}.jpg")

        img.save(img_file)
        json_output = {
            "file_name": f"test_expression_{str(image_id)}.jpg",
            "image_id": image_id,
            "height": sample_h,
            "width": sample_w
        }
        # 20% chance of creating an "empty" sample
        if random.random() < 0.4:
            json_output["grounding"] = {
                "expression": sentence,
                "regions": [],
                "sent_id": image_id,
                "expression_id": image_id,
                "empty": True
            }
        else:
            regions = []
            for _ in range(0, 5):
                x1 = int(np.random.randint(low=0, high=sample_w-TEST_OBJ_WIDTH, size=1)[0])
                y1 = int(np.random.randint(low=0, high=sample_h-TEST_OBJ_HEIGHT, size=1)[0])
                w = int(np.random.randint(low=1, high=TEST_OBJ_WIDTH, size=1)[0])
                h = int(np.random.randint(low=1, high=TEST_OBJ_HEIGHT, size=1)[0])
                x2, y2 = x1 + w, y1 + h
                phrase = random.choice(phrases).lower().strip()
                regions.append({
                    "bbox": [x1, y1, x2, y2],
                    "phrase": phrase,
                    "mask": [[x1, y1, x1, y2, x2, y2, x2, y1]]
                })
            json_output["grounding"] = {
                "expression": sentence,
                "regions": regions,
                "sent_id": image_id,
                "expression_id": image_id,
                "empty": False,
            }
        jsonl_outputs.append(json_output)

    with jsonlines.open(grounding_expression_json_file, 'w') as outfile:
        outfile.write_all(jsonl_outputs)



@pytest.fixture
def _test_grounding_phrase_jsonl():

    sentence = "Two people are talking outside of the video game shop next door to the mobile phone store."
    phrases = ["Two people", "the mobile phone store", "the video game shop"]
    jsonl_outputs = []
    grounding_dir = os.path.join(tmp_top_dir, "grounding_phrase")
    check_and_create(tmp_top_dir)
    check_and_create(grounding_dir)

    for image_id in range(0, 10):
        min_size = int(np.random.randint(low=TEST_SIZE_MIN-20, high=TEST_SIZE_MIN+20, size=1)[0])
        max_size = int(np.random.randint(low=TEST_SIZE_MAX-20, high=TEST_SIZE_MAX+20, size=1)[0])
        if random.random() < 0.5:
            sample_w = min_size
            sample_h = max_size
        else:
            sample_w = max_size
            sample_h = min_size
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_h, sample_w, 3), dtype=np.uint8))
        img_file = os.path.join(grounding_dir, f"test_phrase_{str(image_id)}.jpg")

        img.save(img_file)
        json_output = {
            "file_name": f"test_phrase_{str(image_id)}.jpg",
            "image_id": image_id,
            "height": sample_h,
            "width": sample_w
        }
        # 20% chance of creating an "empty" sample
        if random.random() < 0.4:
            json_output["grounding"] = {
                "caption": sentence,
                "regions": [],
                "sent_id": image_id,
                "expression_id": image_id,
                "empty": True
            }
        else:
            regions = []
            for _ in range(0, 5):
                x1 = int(np.random.randint(low=0, high=sample_w-TEST_OBJ_WIDTH, size=1)[0])
                y1 = int(np.random.randint(low=0, high=sample_h-TEST_OBJ_HEIGHT, size=1)[0])
                w = int(np.random.randint(low=1, high=TEST_OBJ_WIDTH, size=1)[0])
                h = int(np.random.randint(low=1, high=TEST_OBJ_HEIGHT, size=1)[0])
                x2, y2 = x1 + w, y1 + h
                phrase = random.choice(phrases).lower().strip()
                regions.append({
                    "bbox": [x1, y1, x2, y2],
                    "phrase": phrase,
                    "mask": [[x1, y1, x1, y2, x2, y2, x2, y1]]
                })
            json_output["grounding"] = {
                "caption": sentence,
                "regions": regions,
                "sent_id": image_id,
                "expression_id": image_id,
                "empty": False,
            }
        jsonl_outputs.append(json_output)

    with jsonlines.open(grounding_phrase_json_file, 'w') as outfile:
        outfile.write_all(jsonl_outputs)
        
@pytest.fixture
def _test_pred_grounding_jsonl():
    jsonl_outputs = []
    pred_grounding_dir = os.path.join(tmp_top_dir, "pred_grounding")
    check_and_create(tmp_top_dir)
    check_and_create(pred_grounding_dir)
    expression_list = ["PERSON LEFT WITH BACK SHOWING", "Two persons and a flying disk.", "Person on left", "the women wearing white and blue sweater standing in the middle washing dishes and the white clock on the wall", "right girl"]
    
    for image_id in range(0, 10):
        min_size = int(np.random.randint(low=TEST_SIZE_MIN-20, high=TEST_SIZE_MIN+20, size=1)[0])
        max_size = int(np.random.randint(low=TEST_SIZE_MAX-20, high=TEST_SIZE_MAX+20, size=1)[0])
        if random.random() < 0.5:
            sample_w = min_size
            sample_h = max_size
        else:
            sample_w = max_size
            sample_h = min_size
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_h, sample_w, 3), dtype=np.uint8))
        img_file = os.path.join(pred_grounding_dir, f"test_pred_grounding_{str(image_id)}.jpg")

        img.save(img_file)
        json_output = {
            "image_path": f"test_pred_grounding_{str(image_id)}.jpg",
            "expression": expression_list[image_id % len(expression_list)]
        }
        
        jsonl_outputs.append(json_output)

    with jsonlines.open(pred_grounding_json_file, 'w') as outfile:
        outfile.write_all(jsonl_outputs)


@pytest.fixture
def _train_spec(request):
    data_type, folder, json_file = request.param
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.train.num_gpus = 1
    experiment_config.train.num_nodes = 1
    experiment_config.train.num_epochs = 1
    experiment_config.model.loss_types = ['labels', 'boxes', 'masks']
    experiment_config.dataset.train_data_sources = [
                                    {"image_dir": os.path.join(tmp_top_dir, "detection"), 
                                       "json_file": detection_json_file,
                                       "label_map": classmap_file},
                                      {"image_dir": os.path.join(tmp_top_dir, "grounding_expression"),
                                       "json_file": grounding_expression_json_file},
                                      {"image_dir": os.path.join(tmp_top_dir, "grounding_phrase"),
                                       "json_file": grounding_phrase_json_file},
                                      ]
    
    experiment_config.dataset.val_data_sources = {"image_dir": os.path.join(tmp_top_dir, folder), "json_file": json_file, "data_type": data_type}

    experiment_config.dataset.batch_size = 2
    experiment_config.dataset.workers = 0
    experiment_config.dataset.augmentation.fixed_random_crop=AUGMENT_SIZE_CROP
    experiment_config.dataset.augmentation.random_resize_max_size=AUGMENT_MAX_SIZE
    experiment_config.dataset.augmentation.test_random_resize=AUGMENT_TEST_SIZE
    experiment_config = OmegaConf.to_container(experiment_config, resolve=True)
    experiment_config = OmegaConf.create(experiment_config)
    yield experiment_config


@pytest.fixture
def _eval_spec(request):
    data_type, folder, json_file = request.param
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.evaluate.num_gpus = 1
    
    experiment_config.dataset.test_data_sources = {"image_dir": os.path.join(tmp_top_dir, folder), "json_file": json_file, "data_type": data_type}
    experiment_config.dataset.batch_size = 2
    experiment_config.dataset.workers = 0
    experiment_config.dataset.augmentation.fixed_random_crop=AUGMENT_SIZE_CROP
    experiment_config.dataset.augmentation.random_resize_max_size=AUGMENT_MAX_SIZE
    experiment_config.dataset.augmentation.test_random_resize=AUGMENT_TEST_SIZE
    experiment_config = OmegaConf.to_container(experiment_config, resolve=True)
    experiment_config = OmegaConf.create(experiment_config)
    yield experiment_config


@pytest.fixture
def _infer_spec(request):
    data_type = request.param
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.inference.num_gpus = 1
    if data_type == "VG":
        infer_folder = "pred_grounding"
        infer_extra_key = "json_file"
        infer_extra_value = pred_grounding_json_file
    else:
        infer_folder = "coco"
        infer_extra_key = "captions"
        infer_extra_value = "person"
        experiment_config.inference.color_map = {"person": "green", "face": "red", "bag": "blue"}

    experiment_config.dataset.infer_data_sources = {"image_dir": os.path.join(tmp_top_dir, infer_folder), "data_type": data_type, infer_extra_key: infer_extra_value}
    if data_type == "OD":
        experiment_config.dataset.infer_data_sources["label_map"] = classmap_file
    experiment_config.dataset.batch_size = 1
    experiment_config.dataset.workers = 0
    experiment_config.dataset.augmentation.fixed_random_crop=AUGMENT_SIZE_CROP
    experiment_config.dataset.augmentation.random_resize_max_size=AUGMENT_MAX_SIZE
    experiment_config.dataset.augmentation.test_random_resize=AUGMENT_TEST_SIZE
    experiment_config = OmegaConf.to_container(experiment_config, resolve=True)
    experiment_config = OmegaConf.create(experiment_config)
    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.mask_grounding_dino
@pytest.mark.train
@pytest.mark.parametrize("precision", ["32-true", "16-mixed", "bf16-mixed"])
@pytest.mark.parametrize("region_queries", [0, 100, 144, 200])
@pytest.mark.parametrize("_train_spec", [
    ("OD", "coco", json_file),
    ("VG", "grounding_expression", grounding_expression_json_file)
], indirect=True)
@pytest.mark.parametrize("freeze", [[], ["backbone.0", "bert"]])
def test_trainer_fit(_test_detection_jsonl, _test_grounding_expression_jsonl, _test_grounding_phrase_jsonl, _test_sample_json, _train_spec, precision, freeze, region_queries):

    _train_spec.train.freeze = freeze
    _train_spec.model.num_region_queries = region_queries
    if region_queries > 0:
        _train_spec.model.loss_types.append('rela')

    dm = ODVGDataModule(_train_spec.dataset)
    dm.setup(stage="fit")
    cap_lists = dm.val_dataset.cap_lists
    pt_model = MaskGDINOPlModel(_train_spec, cap_lists=cap_lists)

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                      num_nodes=_train_spec.train.num_nodes,
                      default_root_dir=_train_spec.results_dir,
                      accelerator='auto',
                      precision=precision,
                      gradient_clip_val=_train_spec.train.clip_grad_norm,
                      use_distributed_sampler=False,
                      fast_dev_run=FAST_DEV_RUN)

    # Test train
    trainer.fit(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.mask_grounding_dino
@pytest.mark.evaluate
@pytest.mark.parametrize("_eval_spec", [
    ("OD", "coco", json_file),
    ("VG", "grounding_expression", grounding_expression_json_file)
], indirect=True)
def test_trainer_evaluate(_test_detection_jsonl, _test_grounding_expression_jsonl, _test_grounding_phrase_jsonl, _test_sample_json, _eval_spec):

    dm = ODVGDataModule(_eval_spec.dataset)
    dm.setup(stage="test")
    cap_lists = dm.test_dataset.cap_lists
    pt_model = MaskGDINOPlModel(_eval_spec, cap_lists=cap_lists)

    trainer = Trainer(devices=_eval_spec.evaluate.num_gpus,
                      default_root_dir=_eval_spec.results_dir,
                      accelerator='auto',
                      fast_dev_run=FAST_DEV_RUN)
    # Test evaluate
    trainer.test(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.mask_grounding_dino
@pytest.mark.inference
@pytest.mark.parametrize("_infer_spec", ["OD", "VG"], indirect=True)
def test_trainer_inference(_test_sample_json, _test_pred_grounding_jsonl, _infer_spec):

    dm = ODVGDataModule(_infer_spec.dataset)
    dm.setup(stage="predict")
    cap_lists = dm.pred_dataset.cap_lists
    pt_model = MaskGDINOPlModel(_infer_spec, cap_lists=cap_lists)

    trainer = Trainer(devices=_infer_spec.inference.num_gpus,
                      default_root_dir=_infer_spec.results_dir,
                      accelerator='auto',
                      fast_dev_run=FAST_DEV_RUN)
    # Test predict
    trainer.predict(pt_model, dm)

    tmp_top_obj.cleanup()
