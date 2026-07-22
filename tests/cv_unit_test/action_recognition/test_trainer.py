# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import tempfile
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.action_recognition.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.action_recognition.dataloader.pl_ar_data_module import ARDataModule
from nvidia_tao_pytorch.cv.action_recognition.model.pl_ar_model import ActionRecognitionModel

TEST_WIDTH = 224
TEST_HEIGHT = 224
FAST_DEV_RUN = 2  # Run dry run 2 times
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
tmp_top_data_dir = os.path.join(tmp_top_dir, "data")

# TODO @seanf: for now, these test an rgb model, eventually they should do OF and joint as well

@pytest.fixture
def _test_dir():
    img = np.random.randint(low=0, high=255, size=(240, 320, 3), dtype=np.uint8)
    seq_len = 32
    tmp_im = [Image.fromarray(img) for _ in range(seq_len)]
    a_video_dir = os.path.join(tmp_top_data_dir, "a", "001", "rgb")
    b_video_dir = os.path.join(tmp_top_data_dir, "b", "002", "rgb")
    if not os.path.exists(a_video_dir):
        os.makedirs(a_video_dir)
    if not os.path.exists(b_video_dir):
        os.makedirs(b_video_dir)
    for idx, img in enumerate(tmp_im):
        img.save(os.path.join(a_video_dir, str(idx) + ".png"))
    for idx, img in enumerate(tmp_im):
        img.save(os.path.join(b_video_dir, str(idx) + ".png"))


@pytest.fixture
def _train_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "train", "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.train.num_gpus = 1

    experiment_config.dataset.train_dataset_dir = tmp_top_data_dir
    experiment_config.dataset.val_dataset_dir = tmp_top_data_dir
    experiment_config.dataset.batch_size = 2
    experiment_config.dataset.label_map = {'a': 0, 'b': 1}

    experiment_config.model.model_type = "rgb"

    yield experiment_config


@pytest.fixture
def _eval_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "test", "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.evaluate.num_gpus = 1
    experiment_config.evaluate.results_dir = tmp_top_dir
    experiment_config.evaluate.test_dataset_dir = tmp_top_data_dir
    experiment_config.evaluate.batch_size = 2

    experiment_config.dataset.label_map = {'a': 0, 'b': 1}

    experiment_config.model.model_type = "rgb"

    yield experiment_config


@pytest.fixture
def _infer_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "infer", "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.inference.num_gpus = 1
    experiment_config.inference.inference_dataset_dir = os.path.join(tmp_top_data_dir, "a")
    experiment_config.inference.batch_size = 1

    experiment_config.dataset.label_map = {'a': 0, 'b': 1}

    experiment_config.model.model_type = "rgb"

    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.action_recognition
@pytest.mark.train
def test_trainer_fit(_test_dir, _train_spec):

    dm = ARDataModule(_train_spec)
    dm.setup(stage="fit")
    model = ActionRecognitionModel(_train_spec, dm)

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                      max_epochs=_train_spec.train.num_epochs,
                      check_val_every_n_epoch=1,
                      default_root_dir=_train_spec.results_dir,
                      gradient_clip_val=_train_spec.train.clip_grad_norm,
                      fast_dev_run=FAST_DEV_RUN)

    # Test train
    trainer.fit(model, dm)


@pytest.mark.cv_unit
@pytest.mark.action_recognition
@pytest.mark.evaluate
def test_trainer_evaluate(_test_dir, _eval_spec):

    dm = ARDataModule(_eval_spec)
    dm.setup(stage="test")
    pt_model = ActionRecognitionModel(_eval_spec, dm)

    trainer = Trainer(devices=_eval_spec.evaluate.num_gpus,
                      default_root_dir=_eval_spec.results_dir,
                      fast_dev_run=FAST_DEV_RUN)

    # Test evaluate
    trainer.test(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.action_recognition
@pytest.mark.inference
def test_trainer_inference(_test_dir, _infer_spec):

    dm = ARDataModule(_infer_spec)
    dm.setup(stage="predict")
    pt_model = ActionRecognitionModel(_infer_spec, dm)

    trainer = Trainer(devices=_infer_spec.inference.num_gpus,
                      default_root_dir=_infer_spec.results_dir,
                      fast_dev_run=FAST_DEV_RUN)

    # Test predict
    trainer.predict(pt_model, dm)

    tmp_top_obj.cleanup()
