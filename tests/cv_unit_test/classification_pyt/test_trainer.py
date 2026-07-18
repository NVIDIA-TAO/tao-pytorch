# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SegFormer_PL Trainer Unit Tests
"""
import os
import torch
import pytest
import tempfile
import numpy as np
import pandas as pd
from PIL import Image

from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.config.classification_pyt.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.pl_classification_data_module import CLDataModule
from nvidia_tao_pytorch.cv.classification_pyt.model.classifier_pl_model import ClassifierPlModel

FAST_DEV_RUN = 2
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
SAMPLES = 10
BATCH_SIZE = 2
NUM_CLASSES = 10
INPUT_SHAPE = 512
OUTPUT_SHAPE = 224
DATASET = 'CLDataset'
TEST_TOPOLOGIES = [
    # ConvNeXtV2.
    ("convnextv2_atto"),
    # DINOV2.
    ("vit_large_patch14_dinov2_swiglu"),
    # FAN.
    ("fan_small_12_p16_224"),
    ("fan_small_12_p4_hybrid"),
    ("fan_small_12_p16_224_se_attn"),
    # FasterViT.
    ("faster_vit_1_224"),
    # GCViT.
    ("gc_vit_xxtiny"),
    # OpenCLIP.
    ("vit_l_14_siglip_clipa_336"),
    # RADIO.
    ("c_radio_v2_vit_base_patch16"),
]
LARGE_BACKBONE_TOPOLOGIES = [
    # DINOV2.
    ("vit_giant_patch14_reg4_dinov2_swiglu"),
    # RADIO.
    ("c_radio_p3_vit_huge_patch16_mlpnorm"),
]
if not os.getenv("CI_PROJECT_DIR", None):
    TEST_TOPOLOGIES.extend(LARGE_BACKBONE_TOPOLOGIES)



@pytest.fixture
def _test_dir():
    # set this as dataset folder name
    splits = ['train', 'val', 'test']
    img_paths = []

    if not os.path.exists(tmp_top_dir):
        os.makedirs(tmp_top_dir)
    tmp_img_dir = os.path.join(tmp_top_dir)
    check_and_create(tmp_img_dir)

    # write the class.txt to tmp_img_dir, which consists of class names
    class_file = os.path.join(tmp_img_dir, 'classes.txt')
    with open(class_file, 'w') as f:
        for i in range(NUM_CLASSES):
            f.write(str(i) + '\n')

    for split in splits:
        tmp_split_img_dir = os.path.join(tmp_img_dir, split)
        check_and_create(tmp_split_img_dir)
        img_paths.append(tmp_split_img_dir)

    #Input images
    test_data = np.random.rand(INPUT_SHAPE, INPUT_SHAPE, 3) * 255
    test_data = test_data.astype(np.uint8)
    im = Image.fromarray(test_data)

    total_samples = SAMPLES
    for sample in range(total_samples):
        for img_path in img_paths:
            if 'test' in img_path:
                im.save(os.path.join(img_path, str(sample)+'.png'))
            else:
                for class_id in range(NUM_CLASSES):
                    class_dir = os.path.join(img_path, str(class_id))
                    check_and_create(class_dir)
                    # randomly scale the images
                    scale1 = np.random.uniform(0.5, 1.5)
                    scale2 = np.random.uniform(0.5, 1.5)
                    im_resized = im.resize((int(INPUT_SHAPE*scale1), int(INPUT_SHAPE*scale2)))
                    im_resized.save(os.path.join(class_dir, str(sample)+'.png'))


@pytest.fixture
def _train_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config["dataset"]["root_dir"] = tmp_top_dir
    experiment_config["dataset"]["train_dataset"]["images_dir"] = os.path.join(tmp_top_dir, "train")
    experiment_config["dataset"]["val_dataset"]["images_dir"] = os.path.join(tmp_top_dir, "val")
    experiment_config["dataset"]["test_dataset"]["images_dir"] = os.path.join(tmp_top_dir, "test")
    experiment_config["dataset"]["dataset"] = DATASET
    experiment_config["dataset"]["img_size"] = OUTPUT_SHAPE
    experiment_config["dataset"]["batch_size"] = BATCH_SIZE
    experiment_config["dataset"]["num_classes"] = NUM_CLASSES
    experiment_config["dataset"]["classes_file"] = os.path.join(tmp_top_dir, "classes.txt")

    experiment_config["results_dir"] = tmp_top_dir

    experiment_config.train.num_epochs = 1
    experiment_config.train.num_gpus = 1
    experiment_config.train.num_nodes = 1

    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.classification_pyt
@pytest.mark.train
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
def test_trainer_fit(_test_dir, _train_spec, backbone):

    _train_spec.model.backbone.type = backbone

    dm = CLDataModule(_train_spec.dataset)
    dm.setup(stage="fit")
    model = ClassifierPlModel(_train_spec)

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                    num_nodes=_train_spec.train.num_nodes,
                    default_root_dir=_train_spec.results_dir,
                    accelerator='gpu',
                    strategy='auto',
                    precision='32-true',
                    fast_dev_run=FAST_DEV_RUN)

    # Test train
    trainer.fit(model, dm)


@pytest.mark.cv_unit
@pytest.mark.classification_pyt
@pytest.mark.evaluate
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
def test_trainer_evaluate(_test_dir, _train_spec, backbone):

    _train_spec.model.backbone.type = backbone

    dm = CLDataModule(_train_spec.dataset)
    dm.setup(stage="test")
    model = ClassifierPlModel(_train_spec)

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                      default_root_dir=_train_spec.results_dir,
                      accelerator='auto',
                      fast_dev_run=FAST_DEV_RUN)

    trainer.test(model, datamodule=dm)


@pytest.mark.cv_unit
@pytest.mark.classification_pyt
@pytest.mark.inference
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
def test_trainer_infer(_test_dir, _train_spec, backbone):

    _train_spec.model.backbone.type = backbone

    dm = CLDataModule(_train_spec.dataset)
    dm.setup(stage="predict")
    model = ClassifierPlModel(_train_spec)

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                      default_root_dir=_train_spec.results_dir,
                      accelerator='auto',
                      fast_dev_run=FAST_DEV_RUN)

    trainer.predict(model, datamodule=dm)

    tmp_top_obj.cleanup()
