# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import pytest
from omegaconf import OmegaConf
import tempfile

from nvidia_tao_pytorch.config.ml_recog.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.ml_recog.dataloader.pl_ml_data_module import MLDataModule
from nvidia_tao_pytorch.cv.ml_recog.model.build_nn_model import build_model
from nvidia_tao_pytorch.cv.ml_recog.model.pl_ml_recog_model import MLRecogModel

TEST_DATA_DIR = "/home/scratch.metropolis2/tao_ci/tao_pytorch/data/metric_learning_recognition"
tmp_top_obj = tempfile.TemporaryDirectory()
TEST_OUTPUT_DIR = tmp_top_obj.name


@pytest.fixture
def _test_dir():
    output_dir = TEST_OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


@pytest.fixture
def _test_experiment_spec(_test_dir):
    experiment_config = OmegaConf.structured(ExperimentConfig())
    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone, feat_dim, input_width, input_height",
                         [("fan_tiny", 2048, 224, 224),
                          ("fan_small", 256, 224, 256),
                          ("resnet_50", 128, 960, 544)])
def test_MLRecog_model(_test_experiment_spec, backbone, feat_dim, input_width, input_height):
    _test_experiment_spec.model.backbone = backbone
    _test_experiment_spec.model.feat_dim = feat_dim
    _test_experiment_spec.model.input_width = input_width
    _test_experiment_spec.model.input_height = input_height
    build_model(_test_experiment_spec)


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone, feat_dim, batch_size, subtask",
                         [("resnet_101", 2048, 4, "train"),
                          ("fan_large", 256, 4, "evaluate"),
                          ("nvdinov2_vit_large_legacy", 128, 1, "inference")])
def test_MLRecog_pl_model(_test_experiment_spec, backbone, feat_dim, batch_size, subtask):

    _test_experiment_spec.model.backbone = backbone
    _test_experiment_spec.model.feat_dim = feat_dim
    _test_experiment_spec.train.batch_size = batch_size
    _test_experiment_spec.dataset.train_dataset = f"{TEST_DATA_DIR}/train/"
    _test_experiment_spec.dataset.val_dataset = \
        {"reference": f"{TEST_DATA_DIR}/train/", "query": f"{TEST_DATA_DIR}/val/"}
    _test_experiment_spec.results_dir = TEST_OUTPUT_DIR
    if subtask != "train":
        _test_experiment_spec[subtask]["checkpoint"] = "placeholder"
    dm = MLDataModule(_test_experiment_spec)
    MLRecogModel(_test_experiment_spec, dm, subtask)
    shutil.rmtree(TEST_OUTPUT_DIR)
