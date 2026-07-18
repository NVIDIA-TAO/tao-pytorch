# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import tempfile
from pytorch_lightning import Trainer

from omegaconf import OmegaConf
from nvidia_tao_pytorch.config.ml_recog.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.ml_recog.dataloader.pl_ml_data_module import MLDataModule
from nvidia_tao_pytorch.cv.ml_recog.model.pl_ml_recog_model import MLRecogModel


FAST_DEV_RUN = 2  # Run dry run 2 times
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
TEST_DATA_DIR = "/home/scratch.metropolis2/tao_ci/tao_pytorch/data/metric_learning_recognition"


@pytest.fixture
def _train_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.train.results_dir = results_dir
    experiment_config.train.num_gpus = 1
    experiment_config.train.num_nodes = 1
    experiment_config.train.batch_size = 4
    experiment_config.train.val_batch_size = 2

    experiment_config.dataset.train_dataset = f"{TEST_DATA_DIR}/train/"
    experiment_config.dataset.val_dataset = {"reference": f"{TEST_DATA_DIR}/val/", "query": f"{TEST_DATA_DIR}/test/"}

    yield experiment_config


@pytest.fixture
def _eval_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.evaluate.results_dir = results_dir
    experiment_config.evaluate.num_gpus = 1
    experiment_config.evaluate.batch_size = 2
    experiment_config.evaluate.checkpoint = ""

    experiment_config.dataset.val_dataset = {"reference": f"{TEST_DATA_DIR}/val/", "query": f"{TEST_DATA_DIR}/test/"}

    yield experiment_config


@pytest.fixture
def _infer_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.inference.results_dir = results_dir
    experiment_config.inference.num_gpus = 1
    experiment_config.inference.batch_size = 2
    experiment_config.inference.checkpoint = ""
    experiment_config.inference.input_path = f"{TEST_DATA_DIR}/test/"

    experiment_config.dataset.val_dataset = {"reference": f"{TEST_DATA_DIR}/val/"}

    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.metric_learning
@pytest.mark.train
def test_trainer_fit(_train_spec):

    dm = MLDataModule(_train_spec)
    dm.setup('fit')
    model = MLRecogModel(_train_spec, dm, subtask='train')

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                      num_nodes=_train_spec.train.num_nodes,
                      default_root_dir=_train_spec.results_dir,
                      accelerator='auto',
                      gradient_clip_val=_train_spec.train.clip_grad_norm,
                      fast_dev_run=FAST_DEV_RUN)

    # Test train
    trainer.fit(model, dm)


@pytest.mark.cv_unit
@pytest.mark.metric_learning
@pytest.mark.evaluate
def test_trainer_evaluate(_eval_spec):

    dm = MLDataModule(_eval_spec)
    dm.setup('test')
    model = MLRecogModel(_eval_spec, dm, subtask='evaluate')

    trainer = Trainer(devices=_eval_spec.evaluate.num_gpus,
                      default_root_dir=_eval_spec.results_dir,
                      accelerator='auto',
                      fast_dev_run=FAST_DEV_RUN)
    # Test evaluate
    trainer.test(model, dm)


@pytest.mark.cv_unit
@pytest.mark.metric_learning
@pytest.mark.inference
def test_trainer_inference(_infer_spec):

    dm = MLDataModule(_infer_spec)
    dm.setup('predict')
    model = MLRecogModel(_infer_spec, dm, subtask='inference')

    trainer = Trainer(devices=_infer_spec.inference.num_gpus,
                      default_root_dir=_infer_spec.results_dir,
                      accelerator='auto',
                      fast_dev_run=FAST_DEV_RUN)
    # Test predict
    trainer.predict(model, dm)

    tmp_top_obj.cleanup()
