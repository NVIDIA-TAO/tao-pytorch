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

from nvidia_tao_pytorch.config.dino.default_config import ExperimentConfig, DINODistillationConfig, DINOModelDistillationBindingConfig
from nvidia_tao_pytorch.cv.deformable_detr.dataloader.pl_od_data_module import ODDataModule
from nvidia_tao_pytorch.cv.dino.distillation.distiller import DINODistiller
from nvidia_tao_pytorch.core.utilities import check_and_create


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
                    "categories": [ {"supercategory": "person","id": 1, "name": "person"},
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
def _train_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.train.num_gpus = 1
    experiment_config.train.num_nodes = 1
    experiment_config.train.num_epochs = 20

    experiment_config.dataset.train_data_sources = [{"image_dir": tmp_top_dir, "json_file": json_file}]
    experiment_config.dataset.val_data_sources = [{"image_dir": tmp_top_dir, "json_file": json_file}]
    experiment_config.dataset.test_data_sources = {"image_dir": tmp_top_dir, "json_file": json_file}
    experiment_config.dataset.infer_data_sources = {"image_dir": tmp_top_dir, "classmap": classmap_file}
    experiment_config.dataset.num_classes = 4
    experiment_config.dataset.batch_size = 2
    experiment_config.dataset.workers = 0
    experiment_config.dataset.augmentation.random_resize_max_size = 1344

    experiment_config.distill = DINODistillationConfig()
    experiment_config.distill.teacher.backbone = "fan_small"
    experiment_config.distill.teacher.train_backbone = False
    experiment_config.distill.pretrained_teacher_model_path = None
    experiment_config.distill.bindings = []
    experiment_config.distill.bindings.append(DINOModelDistillationBindingConfig())
    experiment_config.distill.bindings[0].student_module_name = "model.backbone.0.body"
    experiment_config.distill.bindings[0].teacher_module_name = "model.backbone.0.body"
    experiment_config.distill.bindings[0].criterion = 'L2'
    experiment_config.distill.bindings[0].weight = 1.0

    experiment_config.model.backbone = "resnet_50"
    experiment_config.model.distillation_loss_coef = 0.5

    yield experiment_config

@pytest.mark.cv_unit
@pytest.mark.parametrize("student_arch, student, teacher, criterion, weight", 
                        [("resnet_50", "model.backbone.0.body", "model.backbone.0.body", 'L2', 1.0),
                        ("resnet_50", "pred_logits", "pred_logits", "KL", 1.0),
                        ("resnet_50","pred_boxes", "pred_boxes", "L2", 1.0),
                        ("efficientvit_b1","pred_boxes", "pred_boxes", "L2", 1.0)])
def test_distiller_fit(_test_sample_json, _train_spec, student_arch, student, teacher, criterion, weight):
    _train_spec.model.backbone = student_arch
    _train_spec.distill.bindings[0].student_module_name = student
    _train_spec.distill.bindings[0].teacher_module_name = teacher
    _train_spec.distill.bindings[0].criterion = criterion
    _train_spec.distill.bindings[0].weight = weight

    acc_flag = 'auto'

    dm = ODDataModule(_train_spec.dataset)
    pt_model = DINODistiller(_train_spec)

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                      num_nodes=_train_spec.train.num_nodes,
                      max_epochs=_train_spec.train.num_epochs,
                      check_val_every_n_epoch=1,
                      default_root_dir=_train_spec.results_dir,
                      accelerator=acc_flag,
                      gradient_clip_val=_train_spec.train.clip_grad_norm,
                      use_distributed_sampler=False,
                      fast_dev_run=FAST_DEV_RUN)

    # Test train
    trainer.fit(pt_model, dm)
