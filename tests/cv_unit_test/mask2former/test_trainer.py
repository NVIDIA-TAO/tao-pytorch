# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pytest
import tempfile
import numpy as np
from PIL import Image
from pytorch_lightning import Trainer

from omegaconf import OmegaConf
from nvidia_tao_pytorch.config.mask2former.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.mask2former.dataloader.pl_data_module import SemSegmDataModule
from nvidia_tao_pytorch.cv.mask2former.model.pl_model import Mask2formerPlModule


FAST_DEV_RUN = 2
TEST_BATCH_SIZE = 4
TEST_WIDTH = 544
TEST_HEIGHT = 960
TEST_OBJ_WIDTH = 80
TEST_OBJ_HEIGHT = 80
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
json_file = os.path.join(tmp_top_dir, "sample_json.json")
jsonl_path = os.path.join(tmp_top_dir, "test_ade.jsonl")
classlist = ["person", "face", "bag"]
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
labelmap = [{"color": color, "isthing": 1, "id": i, "name": label} for i, (label, color) in enumerate(zip(classlist, colors))]
labelmap_file = os.path.join(tmp_top_dir, "labelmap.json")


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
        segm = Image.fromarray(np.random.randint(low=1, high=3, size=(sample_w, sample_h), dtype=np.uint8))
        img_file = os.path.join(tmp_top_dir, f"test_{str(image_id)}.jpg")
        segm_file = os.path.join(tmp_top_dir, f"segm_{str(image_id)}.png")
        img.save(img_file)
        segm.save(segm_file)

        with open(jsonl_path, mode="w", encoding="utf-8") as writer:
            line = {"img": img_file, "segm": segm_file}
            json.dump(line, writer)
            writer.write("\n")

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

    with open(json_file, 'w+') as outfile:
        json.dump(json_output, outfile)

    with open(labelmap_file, "w") as f:
        json.dump(labelmap, f)


@pytest.fixture
def _train_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.train.results_dir = results_dir
    experiment_config.train.num_gpus = 1
    experiment_config.train.num_nodes = 1
    experiment_config.train.num_epochs = 1

    experiment_config.dataset.label_map = labelmap_file
    experiment_config.dataset.train.type = 'coco'
    experiment_config.dataset.train.instance_json = json_file
    experiment_config.dataset.train.img_dir = tmp_top_dir
    experiment_config.dataset.train.batch_size = 4
    experiment_config.dataset.val.type = 'ade'
    experiment_config.dataset.val.annot_file = jsonl_path

    yield experiment_config


@pytest.fixture
def _eval_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.evaluate.results_dir = results_dir
    experiment_config.evaluate.num_gpus = 1

    experiment_config.dataset.label_map = labelmap_file
    experiment_config.dataset.val.type = 'ade'
    experiment_config.dataset.val.annot_file = jsonl_path
    experiment_config.dataset.test.type = 'ade'
    experiment_config.dataset.test.annot_file = jsonl_path

    yield experiment_config


@pytest.fixture
def _infer_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.inference.results_dir = results_dir

    experiment_config.dataset.label_map = labelmap_file
    experiment_config.dataset.test.img_dir = tmp_top_dir

    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.mask2former
@pytest.mark.train
@pytest.mark.parametrize("precision", ["32-true", "16-mixed"])
@pytest.mark.parametrize("name", ['tiny', 'large'])
def test_trainer_fit(_test_sample_json, _train_spec, precision, name):

    _train_spec.model.backbone.swin.type = name

    dm = SemSegmDataModule(_train_spec.dataset)
    dm.setup(stage="fit")
    pt_model = Mask2formerPlModule(_train_spec)

    if not _train_spec.train.iters_per_epoch:
        _train_spec.train.iters_per_epoch = len(dm.train_dataloader())

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                      num_nodes=_train_spec.train.num_nodes,
                      max_epochs=_train_spec.train.num_epochs,
                      check_val_every_n_epoch=1,
                      default_root_dir=_train_spec.results_dir,
                      accelerator='auto',
                      precision=precision,
                      gradient_clip_val=_train_spec.train.clip_grad_norm,
                      use_distributed_sampler=False,
                      fast_dev_run=FAST_DEV_RUN)

    # Test train
    trainer.fit(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.mask2former
@pytest.mark.evaluate
@pytest.mark.parametrize("precision", ["32-true", "16-mixed"])
@pytest.mark.parametrize("name", ['tiny', 'large'])
def test_trainer_evaluate(_test_sample_json, _eval_spec, precision, name):

    _eval_spec.model.backbone.swin.type = name

    dm = SemSegmDataModule(_eval_spec.dataset)
    dm.setup(stage="test")
    pt_model = Mask2formerPlModule(_eval_spec)

    trainer = Trainer(devices=_eval_spec.evaluate.num_gpus,
                      default_root_dir=_eval_spec.results_dir,
                      accelerator='auto',
                      precision=precision,
                      fast_dev_run=FAST_DEV_RUN)
    # Test evaluate
    trainer.test(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.mask2former
@pytest.mark.inference
@pytest.mark.parametrize("precision", ["32-true", "16-mixed"])
@pytest.mark.parametrize("name", ['tiny', 'large'])
def test_trainer_inference(_test_sample_json, _infer_spec, precision, name):

    _infer_spec.model.backbone.swin.type = name

    dm = SemSegmDataModule(_infer_spec.dataset)
    dm.setup(stage="predict")
    pt_model = Mask2formerPlModule(_infer_spec)

    trainer = Trainer(devices=_infer_spec.inference.num_gpus,
                      default_root_dir=_infer_spec.results_dir,
                      accelerator='auto',
                      precision=precision,
                      fast_dev_run=FAST_DEV_RUN)
    # Test predict
    trainer.predict(pt_model, dm)

    tmp_top_obj.cleanup()
