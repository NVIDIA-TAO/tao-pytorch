# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pytest
import tempfile
from PIL import Image
import numpy as np
from pytorch_lightning import Trainer
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.mal.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.mal.datasets.pl_wsi_data_module import WSISDataModule
from nvidia_tao_pytorch.cv.mal.models.mal import MAL, MALPseudoLabels
from nvidia_tao_pytorch.cv.mal.utils.config_utils import update_config

TEST_WIDTH = 544
TEST_HEIGHT = 960
TEST_OBJ_WIDTH = 80
TEST_OBJ_HEIGHT = 80
FAST_DEV_RUN = 2  # Run dry run 2 times
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
def _train_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.train.num_gpus = 1
    experiment_config.train.batch_size = 2

    experiment_config.dataset.train_ann_path = json_file
    experiment_config.dataset.train_img_dir = tmp_top_dir
    experiment_config.dataset.val_ann_path = json_file
    experiment_config.dataset.val_img_dir = tmp_top_dir
    experiment_config.dataset.load_mask = False

    experiment_config = update_config(experiment_config, 'train')

    yield experiment_config


@pytest.fixture
def _eval_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.evaluate.num_gpus = 1
    experiment_config.evaluate.batch_size = 2

    experiment_config.dataset.val_ann_path = json_file
    experiment_config.dataset.val_img_dir = tmp_top_dir
    experiment_config.dataset.load_mask = False

    experiment_config = update_config(experiment_config, 'evaluate')

    yield experiment_config


@pytest.fixture
def _infer_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.inference.num_gpus = 1
    experiment_config.inference.batch_size = 2
    experiment_config.inference.ann_path = json_file
    experiment_config.inference.img_dir = tmp_top_dir
    experiment_config.inference.label_dump_path = os.path.join(tmp_top_dir, 'sample_json_out.json')

    experiment_config.dataset.val_ann_path = json_file
    experiment_config.dataset.val_img_dir = tmp_top_dir
    experiment_config.dataset.load_mask = False

    experiment_config = update_config(experiment_config, 'inference')

    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.mal
@pytest.mark.train
@pytest.mark.parametrize("precision", ["32-true", "16-mixed"])
def test_trainer_fit(_test_sample_json, _train_spec, precision):

    if precision == '32-true':
        _train_spec.train.use_amp = False

    dm = WSISDataModule(num_workers=1, cfg=_train_spec)
    dm.setup(stage="fit")
    num_iter_per_epoch = len(dm.train_dataloader())

    with open(json_file, 'r') as outfile:
        categories = json.load(outfile)["categories"]

    model = MAL(
        cfg=_train_spec, num_iter_per_epoch=num_iter_per_epoch,
        categories=categories)

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                      max_epochs=_train_spec.train.num_epochs,
                      default_root_dir=_train_spec.results_dir,
                      precision=precision,
                      fast_dev_run=FAST_DEV_RUN)

    # Test train
    trainer.fit(model, dm)


@pytest.mark.cv_unit
@pytest.mark.mal
@pytest.mark.evaluate
def test_trainer_evaluate(_test_sample_json, _eval_spec):

    dm = WSISDataModule(num_workers=1, cfg=_eval_spec)
    dm.setup(stage='test')

    with open(json_file, 'r') as outfile:
        categories = json.load(outfile)["categories"]

    pt_model = MAL(
        cfg=_eval_spec, num_iter_per_epoch=1,
        categories=categories)

    trainer = Trainer(devices=_eval_spec.evaluate.num_gpus,
                      default_root_dir=_eval_spec.results_dir,
                      precision='16-mixed',
                      fast_dev_run=FAST_DEV_RUN)

    # Test evaluate
    trainer.test(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.mal
@pytest.mark.inference
def test_trainer_inference(_test_sample_json, _infer_spec):

    dm = WSISDataModule(num_workers=1, cfg=_infer_spec)
    dm.setup(stage='predict')

    with open(json_file, 'r') as outfile:
        categories = json.load(outfile)["categories"]

    # Phase 2: Generating pseudo-labels
    pt_model = MALPseudoLabels(
        cfg=_infer_spec,
        categories=categories)

    trainer = Trainer(devices=_infer_spec.inference.num_gpus,
                      default_root_dir=_infer_spec.results_dir,
                      precision='16-mixed',
                      fast_dev_run=FAST_DEV_RUN)

    # Test predict
    trainer.predict(pt_model, dm)

    tmp_top_obj.cleanup()
