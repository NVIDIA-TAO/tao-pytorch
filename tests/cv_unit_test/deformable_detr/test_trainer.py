# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pytest
import tempfile
import numpy as np
from PIL import Image
from pytorch_lightning import Trainer

from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.deformable_detr.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.cv.deformable_detr.dataloader.pl_od_data_module import ODDataModule
from nvidia_tao_pytorch.cv.deformable_detr.model.pl_dd_model import DeformableDETRModel


TEST_WIDTH = 544
TEST_HEIGHT = 960
TEST_OBJ_WIDTH = 80
TEST_OBJ_HEIGHT = 80
FAST_DEV_RUN = 2  # Run dry run 2 times
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
json_file = os.path.join(tmp_top_dir, "sample_json.json")
classmap_file = os.path.join(tmp_top_dir, "classmap.txt")


@pytest.fixture
def _test_sample_json():
    json_output = {
                    "images": [],
                    "annotations": [],
                    "categories": [{"supercategory": "person","id": 1, "name": "person"},
                                   {"supercategory": "face","id": 2, "name": "face"},
                                   {"supercategory": "bag","id": 3, "name": "bag"}]
                   }
    check_and_create(tmp_top_dir)
    for image_id in range(0, 10):
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(TEST_WIDTH, TEST_HEIGHT, 3), dtype=np.uint8))
        img_file = os.path.join(tmp_top_dir, f"test_{str(image_id)}.jpg")

        img.save(img_file)
        images_info = {
            "id": image_id,
            "file_name": f"test_{str(image_id)}.jpg",
            "height": TEST_HEIGHT,
            "width": TEST_WIDTH
        }
        json_output["images"].append(images_info)

        for cat_id in range(0, 5):
            annotation_id = image_id + cat_id
            category_id = int(np.random.randint(low=1, high=3, size=1)[0])
            x1 = int(np.random.randint(low=0, high=TEST_WIDTH-TEST_OBJ_WIDTH, size=1)[0])
            y1 = int(np.random.randint(low=0, high=TEST_HEIGHT-TEST_OBJ_HEIGHT, size=1)[0])
            w = int(np.random.randint(low=1, high=TEST_OBJ_WIDTH, size=1)[0])
            h = int(np.random.randint(low=1, high=TEST_OBJ_HEIGHT, size=1)[0])
            bbox = [x1, y1, w, h]
            area = bbox[2] * bbox[3]
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
def _train_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.train.num_gpus = 1
    experiment_config.train.num_nodes = 1

    experiment_config.dataset.train_data_sources = [{"image_dir": tmp_top_dir, "json_file": json_file}]
    experiment_config.dataset.val_data_sources = [{"image_dir": tmp_top_dir, "json_file": json_file}]
    experiment_config.dataset.test_data_sources = {"image_dir": tmp_top_dir, "json_file": json_file}
    experiment_config.dataset.infer_data_sources = {"image_dir": tmp_top_dir, "classmap": classmap_file}
    experiment_config.dataset.num_classes = 4
    experiment_config.dataset.batch_size = 2
    experiment_config.dataset.workers = 0

    yield experiment_config


@pytest.fixture
def _eval_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.evaluate.num_gpus = 1

    experiment_config.dataset.test_data_sources = {"image_dir": tmp_top_dir, "json_file": json_file}
    experiment_config.dataset.num_classes = 4
    experiment_config.dataset.batch_size = 2
    experiment_config.dataset.workers = 0

    yield experiment_config


@pytest.fixture
def _infer_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.inference.num_gpus = 1
    experiment_config.inference.color_map = {"person": "green", "face": "red", "bag": "blue"}

    experiment_config.dataset.infer_data_sources = {"image_dir": tmp_top_dir, "classmap": classmap_file}
    experiment_config.dataset.num_classes = 4
    experiment_config.dataset.batch_size = 2
    experiment_config.dataset.workers = 0

    yield experiment_config


@pytest.mark.parametrize("precision", ['32-true', '16-mixed'])
@pytest.mark.parametrize("distributed_strategy", ["ddp", "fsdp"])
@pytest.mark.cv_unit
@pytest.mark.deformable_detr
@pytest.mark.train
def test_trainer_fit(_test_sample_json, _train_spec, precision, distributed_strategy):

    # Replicating this from
    # nvidia_tao_pytorch/cv/deformable_detr/scripts/train.py
    # to handle the find_unused_parameters
    if _train_spec.train.activation_checkpoint and \
        len(_train_spec.model.return_interm_indices) < 4 and \
            _train_spec.train.num_gpus > 1:
        _train_spec.train.activation_checkpoint = False
        print("Disabling  activation checkpointing since model is smaller")

    strategy = 'auto'
    activation_checkpoint = _train_spec.train.activation_checkpoint
    if _train_spec.train.num_gpus > 1:
        if distributed_strategy.lower() == "ddp" and activation_checkpoint:
            strategy = 'ddp'
        else:
            strategy = 'ddp_find_unused_parameters_true'

    dm = ODDataModule(_train_spec.dataset)
    dm.setup(stage="fit")
    pt_model = DeformableDETRModel(_train_spec)

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                      num_nodes=_train_spec.train.num_nodes,
                      default_root_dir=_train_spec.results_dir,
                      accelerator='gpu',
                      precision=precision,
                      strategy=strategy,
                      gradient_clip_val=_train_spec.train.clip_grad_norm,
                      use_distributed_sampler=False,
                      fast_dev_run=FAST_DEV_RUN)

    # Test train
    trainer.fit(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.deformable_detr
@pytest.mark.evaluate
def test_trainer_evaluate(_test_sample_json, _eval_spec):

    dm = ODDataModule(_eval_spec.dataset)
    dm.setup(stage="test")
    pt_model = DeformableDETRModel(_eval_spec)

    trainer = Trainer(devices=_eval_spec.evaluate.num_gpus,
                      default_root_dir=_eval_spec.results_dir,
                      accelerator='gpu',
                      strategy='auto',
                      fast_dev_run=FAST_DEV_RUN)
    # Test evaluate
    trainer.test(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.deformable_detr
@pytest.mark.inference
def test_trainer_inference(_test_sample_json, _infer_spec):

    dm = ODDataModule(_infer_spec.dataset)
    dm.setup(stage="predict")
    pt_model = DeformableDETRModel(_infer_spec)

    trainer = Trainer(devices=_infer_spec.inference.num_gpus,
                      default_root_dir=_infer_spec.results_dir,
                      accelerator='gpu',
                      strategy='auto',
                      fast_dev_run=FAST_DEV_RUN)
    # Test predict
    trainer.predict(pt_model, dm)

    tmp_top_obj.cleanup()
