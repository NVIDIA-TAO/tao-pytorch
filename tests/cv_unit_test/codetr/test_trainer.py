# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simple trainer tests for CoDETR."""

import json
import os
import tempfile

import numpy as np
import pytest
from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.codetr.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.cv.codetr.model.pl_codetr_model import CoDETRPlModel
from nvidia_tao_pytorch.cv.deformable_detr.dataloader.pl_od_data_module import ODDataModule


TEST_WIDTH = 544
TEST_HEIGHT = 960
TEST_OBJ_WIDTH = 80
TEST_OBJ_HEIGHT = 80
FAST_DEV_RUN = 2

tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
json_file = os.path.join(tmp_top_dir, "sample_json.json")
classmap_file = os.path.join(tmp_top_dir, "classmap.txt")


@pytest.fixture
def _test_sample_json():
    json_output = {
        "images": [],
        "annotations": [],
        "categories": [
            {"supercategory": "person", "id": 1, "name": "person"},
            {"supercategory": "face", "id": 2, "name": "face"},
            {"supercategory": "bag", "id": 3, "name": "bag"},
        ],
    }
    check_and_create(tmp_top_dir)
    for image_id in range(0, 10):
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(TEST_WIDTH, TEST_HEIGHT, 3), dtype=np.uint8))
        img_file = os.path.join(tmp_top_dir, f"test_{str(image_id)}.jpg")
        img.save(img_file)
        json_output["images"].append({
            "id": image_id,
            "file_name": f"test_{str(image_id)}.jpg",
            "height": TEST_HEIGHT,
            "width": TEST_WIDTH,
        })

        for cat_id in range(0, 5):
            annotation_id = image_id + cat_id
            category_id = int(np.random.randint(low=1, high=3, size=1)[0])
            x1 = int(np.random.randint(low=0, high=TEST_WIDTH - TEST_OBJ_WIDTH, size=1)[0])
            y1 = int(np.random.randint(low=0, high=TEST_HEIGHT - TEST_OBJ_HEIGHT, size=1)[0])
            w = int(np.random.randint(low=1, high=TEST_OBJ_WIDTH, size=1)[0])
            h = int(np.random.randint(low=1, high=TEST_OBJ_HEIGHT, size=1)[0])
            bbox = [x1, y1, w, h]
            json_output["annotations"].append({
                "image_id": image_id,
                "category_id": category_id,
                "id": annotation_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            })

    with open(json_file, "w+") as f:
        json.dump(json_output, f)

    with open(classmap_file, "w") as f:
        for classname in ["person", "face", "bag"]:
            f.write(classname + "\n")


def _base_spec():
    cfg = OmegaConf.structured(ExperimentConfig())
    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    cfg.results_dir = results_dir

    # Light-weight model setup so the trainer step is quick.
    cfg.model.backbone = "resnet_50"
    cfg.model.num_feature_levels = 2
    cfg.model.return_interm_indices = [1, 2]
    cfg.model.num_queries = 100
    cfg.model.num_co_heads = 1
    cfg.model.co_head_num_convs = 1

    cfg.dataset.train_data_sources = [{"image_dir": tmp_top_dir, "json_file": json_file}]
    cfg.dataset.val_data_sources = [{"image_dir": tmp_top_dir, "json_file": json_file}]
    cfg.dataset.test_data_sources = {"image_dir": tmp_top_dir, "json_file": json_file}
    cfg.dataset.infer_data_sources = {"image_dir": tmp_top_dir, "classmap": classmap_file}
    cfg.dataset.num_classes = 4
    cfg.dataset.batch_size = 2
    cfg.dataset.workers = 0
    return cfg


@pytest.fixture
def _train_spec():
    cfg = _base_spec()
    cfg.train.num_gpus = 1
    cfg.train.num_nodes = 1
    yield cfg


@pytest.fixture
def _eval_spec():
    cfg = _base_spec()
    cfg.evaluate.num_gpus = 1
    yield cfg


@pytest.fixture
def _infer_spec():
    cfg = _base_spec()
    cfg.inference.num_gpus = 1
    cfg.inference.color_map = {"person": "green", "face": "red", "bag": "blue"}
    yield cfg


@pytest.mark.cv_unit
@pytest.mark.train
def test_codetr_trainer_fit(_test_sample_json, _train_spec):
    """Smoke test: CoDETR PL model runs a fast_dev_run train loop."""
    dm = ODDataModule(_train_spec.dataset)
    dm.setup(stage="fit")
    pt_model = CoDETRPlModel(_train_spec)

    trainer = Trainer(
        devices=_train_spec.train.num_gpus,
        num_nodes=_train_spec.train.num_nodes,
        default_root_dir=_train_spec.results_dir,
        accelerator="auto",
        gradient_clip_val=_train_spec.train.clip_grad_norm,
        use_distributed_sampler=False,
        fast_dev_run=FAST_DEV_RUN,
    )

    trainer.fit(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.evaluate
def test_codetr_trainer_evaluate(_test_sample_json, _eval_spec):
    """Smoke test: CoDETR PL model runs a fast_dev_run test/evaluate loop."""
    dm = ODDataModule(_eval_spec.dataset)
    dm.setup(stage="test")
    pt_model = CoDETRPlModel(_eval_spec)

    trainer = Trainer(
        devices=_eval_spec.evaluate.num_gpus,
        default_root_dir=_eval_spec.results_dir,
        accelerator="auto",
        fast_dev_run=FAST_DEV_RUN,
    )
    trainer.test(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.inference
def test_codetr_trainer_inference(_test_sample_json, _infer_spec):
    """Smoke test: CoDETR PL model runs a fast_dev_run predict loop."""
    dm = ODDataModule(_infer_spec.dataset)
    dm.setup(stage="predict")
    pt_model = CoDETRPlModel(_infer_spec)

    trainer = Trainer(
        devices=_infer_spec.inference.num_gpus,
        default_root_dir=_infer_spec.results_dir,
        accelerator="auto",
        fast_dev_run=FAST_DEV_RUN,
    )
    trainer.predict(pt_model, dm)

    tmp_top_obj.cleanup()
