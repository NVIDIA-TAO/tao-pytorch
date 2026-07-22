# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
from PIL import Image
import pytest
import tempfile
from pytorch_lightning import Trainer
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.ocdnet.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.ocdnet.data_loader.pl_ocd_data_module import OCDDataModule
from nvidia_tao_pytorch.cv.ocdnet.model.pl_ocd_model import OCDnetModel

TEST_SAMPEL = 10
TEST_IMG_WIDTH  = 1280
TEST_IMG_HEIGHT = 720
TEST_DEFAULT_LABEL = '100,300,800,300,800,500,100,500,NVIDIA'
FAST_DEV_RUN = 2  # Run dry run 2 times
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name


@pytest.fixture
def _test_icdar_dir():
    if not os.path.isdir(tmp_top_dir):
        os.makedirs(tmp_top_dir, exist_ok=True)
    tmp_img_dir = os.path.join(tmp_top_dir, "img")
    tmp_gt_dir = os.path.join(tmp_top_dir, "gt")
    os.makedirs(tmp_img_dir, exist_ok=True)
    os.makedirs(tmp_gt_dir, exist_ok=True)
    for img_idx in range(0, TEST_SAMPEL):
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(TEST_IMG_HEIGHT, TEST_IMG_WIDTH, 3), dtype=np.uint8))
        img_file = os.path.join(tmp_img_dir, f"test_{str(img_idx)}.jpg")
        img.save(img_file)
        gt_file = os.path.join(tmp_gt_dir, f"gt_test_{str(img_idx)}.txt")
        with open(gt_file, 'w') as f:
            f.write(f'{TEST_DEFAULT_LABEL}\n')

    yield tmp_top_dir


@pytest.fixture
def _train_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.train.results_dir = results_dir
    experiment_config.train.num_gpus = 1
    experiment_config.train.trainer.clip_grad_norm = 5
    experiment_config.train.post_processing.args.thresh = 0.3
    experiment_config.train.post_processing.args.box_thresh = 0.55
    experiment_config.train.post_processing.args.max_candidates = 1000
    experiment_config.train.post_processing.args.unclip_ratio = 1.5
    experiment_config.train.lr_scheduler.args.warmup_epoch = 0
    experiment_config.train.metric.args.is_output_polygon = False

    experiment_config.model.load_pruned_graph = False

    experiment_config.dataset.train_dataset.data_path = [tmp_top_dir]
    experiment_config.dataset.train_dataset.loader.batch_size = 2
    experiment_config.dataset.train_dataset.args.pre_processes = []
    experiment_config.dataset.train_dataset.args.pre_processes.append({'type': 'EastRandomCropData',
                                                                      'args': {'size': [640, 640],
                                                                               'max_tries': 50,
                                                                               'keep_ratio': True}})
    experiment_config.dataset.train_dataset.args.pre_processes.append({'type': 'MakeBorderMap',
                                                                      'args': {'shrink_ratio': 0.4,
                                                                               'thresh_min': 0.3,
                                                                               'thresh_max': 0.7}})
    experiment_config.dataset.train_dataset.args.pre_processes.append({'type': 'MakeShrinkMap',
                                                                      'args': {'shrink_ratio': 0.4,
                                                                               'min_text_size': 8}})

    experiment_config.dataset.validate_dataset.data_path = [tmp_top_dir]
    experiment_config.dataset.validate_dataset.loader.batch_size = 2
    experiment_config.dataset.validate_dataset.args.pre_processes = [{'type': 'Resize2D',
                                                                     'args': {'short_size': [1280, 736],
                                                                              'resize_text_polys': True}}]

    yield experiment_config


@pytest.fixture
def _eval_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.evaluate.num_gpus = 1
    experiment_config.evaluate.results_dir = tmp_top_dir
    experiment_config.evaluate.post_processing.args.thresh = 0.3
    experiment_config.evaluate.post_processing.args.box_thresh = 0.55
    experiment_config.evaluate.post_processing.args.max_candidates = 1000
    experiment_config.evaluate.post_processing.args.unclip_ratio = 1.5
    experiment_config.evaluate.metric.args.is_output_polygon = False

    experiment_config.model.load_pruned_graph = False

    experiment_config.dataset.validate_dataset.data_path = [tmp_top_dir]
    experiment_config.dataset.validate_dataset.loader.batch_size = 2
    experiment_config.dataset.validate_dataset.args.pre_processes = [{'type': 'Resize2D',
                                                                     'args': {'short_size': [1280, 736],
                                                                              'resize_text_polys': True}}]

    yield experiment_config


@pytest.fixture
def _infer_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.inference.num_gpus = 1
    experiment_config.inference.input_folder = tmp_top_dir
    experiment_config.inference.width = TEST_IMG_WIDTH
    experiment_config.inference.height = TEST_IMG_HEIGHT
    experiment_config.inference.img_mode = 'BGR'
    experiment_config.inference.polygon = False
    experiment_config.inference.post_processing.args.thresh = 0.3
    experiment_config.inference.post_processing.args.box_thresh = 0.55
    experiment_config.inference.post_processing.args.max_candidates = 1000
    experiment_config.inference.post_processing.args.unclip_ratio = 1.5

    experiment_config.model.load_pruned_graph = False

    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.ocdnet
@pytest.mark.train
@pytest.mark.parametrize("precision", ["32-true", "16-mixed"])
@pytest.mark.parametrize("backbone", ["deformable_resnet18", "fan_tiny_8_p4_hybrid"])
def test_trainer_fit(_test_icdar_dir, _train_spec, precision, backbone):

    _train_spec.model.backbone = backbone
    sync_batchnorm = False
    if 'fan' in backbone:
        _train_spec.model.enlarge_feature_map_size = True
        _train_spec.model.activation_checkpoint = True
        sync_batchnorm = True

    _train_spec = OmegaConf.to_container(_train_spec)

    dm = OCDDataModule(_train_spec)
    dm.setup('fit')
    model = OCDnetModel(_train_spec, dm, 'fit')

    trainer = Trainer(devices=_train_spec['train']['num_gpus'],
                      default_root_dir=_train_spec['results_dir'],
                      gradient_clip_val=_train_spec['train']['trainer']['clip_grad_norm'],
                      precision=precision,
                      sync_batchnorm=sync_batchnorm,
                      fast_dev_run=FAST_DEV_RUN)

    # Test train
    trainer.fit(model, dm)


@pytest.mark.cv_unit
@pytest.mark.ocdnet
@pytest.mark.evaluate
@pytest.mark.parametrize("backbone", ["deformable_resnet18", "fan_tiny_8_p4_hybrid"])
def test_trainer_evaluate(_test_icdar_dir, _eval_spec, backbone):

    _eval_spec.model.backbone = backbone
    if 'fan' in backbone:
        _eval_spec.model.enlarge_feature_map_size = True
        _eval_spec.model.activation_checkpoint = True

    _eval_spec = OmegaConf.to_container(_eval_spec)

    dm = OCDDataModule(_eval_spec)
    dm.setup(stage='test')
    model = OCDnetModel(_eval_spec, dm, 'test')

    trainer = Trainer(devices=_eval_spec['evaluate']['num_gpus'],
                      default_root_dir=_eval_spec['results_dir'],
                      fast_dev_run=FAST_DEV_RUN)

    # Test evaluate
    trainer.test(model, dm)


@pytest.mark.cv_unit
@pytest.mark.ocdnet
@pytest.mark.inference
@pytest.mark.parametrize("backbone", ["deformable_resnet18", "fan_tiny_8_p4_hybrid"])
def test_trainer_inference(_test_icdar_dir, _infer_spec, backbone):

    _infer_spec.model.backbone = backbone
    if 'fan' in backbone:
        _infer_spec.model.enlarge_feature_map_size = True
        _infer_spec.model.activation_checkpoint = True

    _infer_spec = OmegaConf.to_container(_infer_spec)

    dm = OCDDataModule(_infer_spec)
    dm.setup(stage='predict')
    model = OCDnetModel(_infer_spec, dm, 'predict')

    trainer = Trainer(devices=_infer_spec['inference']['num_gpus'],
                      default_root_dir=_infer_spec['results_dir'],
                      fast_dev_run=FAST_DEV_RUN)

    # Test predict
    trainer.predict(model, dm)

    tmp_top_obj.cleanup()
