# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pytest
import tempfile
import numpy as np
from PIL import Image
import cv2
from pytorch_lightning import Trainer

from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.depth_net.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.cv.depth_net.dataloader import build_pl_data_module
from nvidia_tao_pytorch.cv.depth_net.model.build_pl_model import build_pl_model


TEST_WIDTH = 924
TEST_HEIGHT = 512
TRAIN_BATCH_SIZE = 2
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 2
FAST_DEV_RUN = 2  # Run dry run 2 times
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
mono_txt_file = os.path.join(tmp_top_dir, "mono_train.txt")


@pytest.fixture
def _mono_test_sample_txt():
    check_and_create(tmp_top_dir)
    with open(mono_txt_file, 'w') as fout:

        for image_id in range(0, 10):
            sample_w = int(np.random.randint(low=TEST_WIDTH-20, high=TEST_WIDTH+20, size=1)[0])
            sample_h = int(np.random.randint(low=TEST_HEIGHT-20, high=TEST_HEIGHT+20, size=1)[0])
            img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_w, sample_h, 3), dtype=np.uint8))
            img_file = os.path.join(tmp_top_dir, f"image_{str(image_id)}.jpg")

            depth = np.random.randint(low=0, high=255, size=(sample_w, sample_h, 1), dtype=np.uint8)
            depth_file = os.path.join(tmp_top_dir, f"depth_{str(image_id)}.png")

            img.save(img_file)
            cv2.imwrite(depth_file, depth)

            print("{} {}".format(img_file, depth_file), file=fout)

    fout.close()


@pytest.fixture
def _train_relative_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.train.num_gpus = 1
    experiment_config.train.num_nodes = 1
    experiment_config.train.num_epochs = 1
    experiment_config.dataset.dataset_name = "MonoDataset"
    experiment_config.model.model_type = "RelativeDepthAnything"
    experiment_config.dataset.train_dataset.data_sources = [{"dataset_name": "RelativeMonoDataset", "data_file": mono_txt_file}]
    experiment_config.dataset.val_dataset.data_sources = [{"dataset_name": "RelativeMonoDataset", "data_file": mono_txt_file}]
    experiment_config.dataset.train_dataset.batch_size = TRAIN_BATCH_SIZE
    experiment_config.dataset.val_dataset.batch_size = VAL_BATCH_SIZE
    experiment_config.train.optim.lr_scheduler = "LambdaLR"
    experiment_config.train.optim.lr = 0.000006

    yield experiment_config

@pytest.fixture
def _eval_relative_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir
    experiment_config.dataset.dataset_name = "MonoDataset"
    experiment_config.model.model_type = "RelativeDepthAnything"

    experiment_config.evaluate.num_gpus = 1

    experiment_config.dataset.test_dataset.data_sources = [{"dataset_name": "RelativeMonoDataset", "data_file": mono_txt_file}]
    experiment_config.dataset.test_dataset.batch_size = TEST_BATCH_SIZE
    experiment_config.dataset.test_dataset.workers = 0
    yield experiment_config


@pytest.fixture
def _infer_relative_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir
    experiment_config.dataset.dataset_name = "MonoDataset"
    experiment_config.model.model_type = "RelativeDepthAnything"

    experiment_config.inference.num_gpus = 1

    experiment_config.dataset.infer_dataset.data_sources = [{"dataset_name": "RelativeMonoDataset", "data_file": mono_txt_file}]
    experiment_config.dataset.infer_dataset.batch_size = TEST_BATCH_SIZE
    experiment_config.dataset.infer_dataset.workers = 0

    yield experiment_config


@pytest.fixture
def _train_metric_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.train.num_gpus = 1
    experiment_config.train.num_nodes = 1
    experiment_config.train.num_epochs = 1
    experiment_config.dataset.dataset_name = "MonoDataset"
    experiment_config.model.model_type = "MetricDepthAnything"
    experiment_config.dataset.train_dataset.data_sources = [{"dataset_name": "MetricMonoDataset", "data_file": mono_txt_file}]
    experiment_config.dataset.val_dataset.data_sources = [{"dataset_name": "MetricMonoDataset", "data_file": mono_txt_file}]
    experiment_config.dataset.min_depth = 0.001
    experiment_config.dataset.max_depth = 10
    experiment_config.dataset.train_dataset.batch_size = TRAIN_BATCH_SIZE
    experiment_config.dataset.val_dataset.batch_size = VAL_BATCH_SIZE
    experiment_config.train.optim.lr_scheduler = "LambdaLR"
    experiment_config.train.optim.lr = 0.000006

    yield experiment_config


@pytest.fixture
def _infer_metric_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.dataset.dataset_name = "MonoDataset"
    experiment_config.model.model_type = "MetricDepthAnything"

    experiment_config.inference.num_gpus = 1
    experiment_config.dataset.min_depth = 0.001
    experiment_config.dataset.max_depth = 10
    experiment_config.dataset.infer_dataset.data_sources = [{"dataset_name": "MetricMonoDataset", "data_file": mono_txt_file}]
    experiment_config.dataset.infer_dataset.batch_size = TEST_BATCH_SIZE
    experiment_config.dataset.infer_dataset.workers = 0

    yield experiment_config

@pytest.fixture
def _eval_metric_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())
    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir
    experiment_config.dataset.dataset_name = "MonoDataset"
    experiment_config.model.model_type = "MetricDepthAnything"
    experiment_config.evaluate.num_gpus = 1
    experiment_config.dataset.min_depth = 0.001
    experiment_config.dataset.max_depth = 10
    experiment_config.dataset.test_dataset.data_sources = [{"dataset_name": "MetricMonoDataset", "data_file": mono_txt_file}]
    experiment_config.dataset.test_dataset.batch_size = TEST_BATCH_SIZE
    experiment_config.dataset.test_dataset.workers = 0
    yield experiment_config


@pytest.mark.parametrize("precision", ['32-true'])#, '16-mixed'])
@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.train
def test_relative_trainer_fit(_mono_test_sample_txt, _train_relative_spec, precision):

    strategy = 'auto'
    dm = build_pl_data_module(_train_relative_spec.dataset)
    dm.setup(stage="fit")
    
    pt_model = build_pl_model(_train_relative_spec)

    trainer = Trainer(devices=_train_relative_spec.train.num_gpus,
                      num_nodes=_train_relative_spec.train.num_nodes,
                      default_root_dir=_train_relative_spec.results_dir,
                      accelerator='gpu',
                      precision=precision,
                      strategy=strategy,
                      gradient_clip_val=_train_relative_spec.train.clip_grad_norm,
                      use_distributed_sampler=False,
                      fast_dev_run=FAST_DEV_RUN)

    # Test train
    trainer.fit(pt_model, dm)
    tmp_top_obj.cleanup()


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.evaluate
def test_relative_trainer_evaluate(_mono_test_sample_txt, _eval_relative_spec):

    dm = build_pl_data_module(_eval_relative_spec.dataset)
    dm.setup(stage="test")
    pt_model = build_pl_model(_eval_relative_spec)

    trainer = Trainer(devices=_eval_relative_spec.evaluate.num_gpus,
                      default_root_dir=_eval_relative_spec.results_dir,
                      accelerator='gpu',
                      strategy='auto',
                      fast_dev_run=FAST_DEV_RUN)
    # Test evaluate
    trainer.test(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.inference
def test_relative_trainer_inference(_mono_test_sample_txt, _infer_relative_spec):

    dm = build_pl_data_module(_infer_relative_spec.dataset)
    dm.setup(stage="predict")
    pt_model = build_pl_model(_infer_relative_spec)

    trainer = Trainer(devices=_infer_relative_spec.inference.num_gpus,
                      default_root_dir=_infer_relative_spec.results_dir,
                      accelerator='gpu',
                      strategy='auto',
                      fast_dev_run=FAST_DEV_RUN)
    # Test predict
    trainer.predict(pt_model, dm)

    tmp_top_obj.cleanup()


@pytest.mark.parametrize("precision", ['32-true', '16-mixed'])
@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.train
def test_metric_trainer_fit(_mono_test_sample_txt, _train_metric_spec, precision):

    strategy = 'auto'
    dm = build_pl_data_module(_train_metric_spec.dataset)
    dm.setup(stage="fit")
    
    pt_model = build_pl_model(_train_metric_spec)

    trainer = Trainer(devices=_train_metric_spec.train.num_gpus,
                      num_nodes=_train_metric_spec.train.num_nodes,
                      default_root_dir=_train_metric_spec.results_dir,
                      accelerator='gpu',
                      precision=precision,
                      strategy=strategy,
                      gradient_clip_val=_train_metric_spec.train.clip_grad_norm,
                      use_distributed_sampler=False,
                      fast_dev_run=FAST_DEV_RUN)
    # Test train
    trainer.fit(pt_model, dm)
    tmp_top_obj.cleanup()


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.evaluate
def test_metric_trainer_evaluate(_mono_test_sample_txt, _eval_metric_spec):

    dm = build_pl_data_module(_eval_metric_spec.dataset)
    dm.setup(stage="test")
    pt_model = build_pl_model(_eval_metric_spec)

    trainer = Trainer(devices=_eval_metric_spec.evaluate.num_gpus,
                      default_root_dir=_eval_metric_spec.results_dir,
                      accelerator='gpu',
                      strategy='auto',
                      fast_dev_run=FAST_DEV_RUN)
    # Test evaluate
    trainer.test(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.depth_net
@pytest.mark.inference
def test_metric_trainer_inference(_mono_test_sample_txt, _infer_metric_spec):
    dm = build_pl_data_module(_infer_metric_spec.dataset)
    dm.setup(stage="predict")
    pt_model = build_pl_model(_infer_metric_spec)

    trainer = Trainer(devices=_infer_metric_spec.inference.num_gpus,
                      default_root_dir=_infer_metric_spec.results_dir,
                      accelerator='gpu',
                      strategy='auto',
                      fast_dev_run=FAST_DEV_RUN)
    # Test predict
    trainer.predict(pt_model, dm)

    tmp_top_obj.cleanup()
