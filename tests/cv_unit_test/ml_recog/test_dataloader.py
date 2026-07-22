# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.ml_recog.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.ml_recog.dataloader.build_data_loader import build_dataloader, build_inference_dataloader
from nvidia_tao_pytorch.cv.ml_recog.dataloader.datasets.image_datasets import MetricLearnImageFolder
from nvidia_tao_pytorch.cv.ml_recog.dataloader.transforms import build_transforms

TEST_DATA_DIR = "/home/scratch.metropolis2/tao_ci/tao_pytorch/data/metric_learning_recognition"


@pytest.fixture
def _test_spec():
    spec = OmegaConf.structured(ExperimentConfig())
    yield spec


@pytest.mark.cv_unit
def test_dataset_dict(_test_spec):

    _test_spec["dataset"]["train_dataset"] = f"{TEST_DATA_DIR}/train/"
    _test_spec["dataset"]["val_dataset"] = \
        {"reference": f"{TEST_DATA_DIR}/val/", "query": f"{TEST_DATA_DIR}/test/"}
    # batch_size % num_instance == 0 and number of examples / batch_size <= number of classes
    _test_spec["train"]["batch_size"] = 4
    _, _, _, dataset_dict = build_dataloader(_test_spec, mode="train")
    assert len(dataset_dict["train"]) == 50, "The train folder should have 50 images."
    assert len(dataset_dict["gallery"]) == 25, "The gallery folder should have 25 images."
    assert len(dataset_dict["query"]) == 25, "The query folder should have 25 images."


@pytest.mark.cv_unit
@pytest.mark.parametrize("mode, batch_size, workers",
                         [("eval", 1, 8),
                          ("inference", 2, 16),
                          ("eval", 2, 16),
                          ("train", 4, 12),
                          ("train", 8, 16)])
def test_build_dataloader(_test_spec, mode, batch_size, workers):
    _test_spec["dataset"]["train_dataset"] = f"{TEST_DATA_DIR}/train/"
    _test_spec["dataset"]["val_dataset"] = \
        {"reference": f"{TEST_DATA_DIR}/train/", "query": f"{TEST_DATA_DIR}/val/"}
    if mode == "train":
        _test_spec["train"]["batch_size"] = batch_size
    elif mode == "inference":
        _test_spec["inference"]["batch_size"] = batch_size
    elif mode == "eval":
        _test_spec["evaluate"]["batch_size"] = batch_size
    _test_spec["dataset"]["workers"] = workers
    build_dataloader(_test_spec, mode)


@pytest.mark.cv_unit
@pytest.mark.parametrize("batch_size, workers",
                         [(1, 8),
                          (4, 8),
                          (8, 16)])
def test_build_inference_dataloader_cls_folder(_test_spec, batch_size, workers):
    _test_spec["dataset"]["val_dataset"] = \
        {"reference": f"{TEST_DATA_DIR}/train/", "query": ""}
    _test_spec["inference"]["input_path"] = f"{TEST_DATA_DIR}/test/"
    _test_spec["inference"]["batch_size"] = batch_size
    _test_spec["dataset"]["workers"] = workers
    _test_spec["inference"]["inference_input_type"] = 'classification_folder'
    build_inference_dataloader(_test_spec)


@pytest.mark.cv_unit
@pytest.mark.parametrize("batch_size, workers",
                         [(1, 8),
                          (4, 8),
                          (8, 16)])
def test_build_inference_dataloader_image_folder(_test_spec, batch_size, workers):
    _test_spec["dataset"]["val_dataset"] = \
        {"reference": f"{TEST_DATA_DIR}/train/", "query": ""}
    _test_spec["inference"]["input_path"] = f"{TEST_DATA_DIR}/test/c000001"
    _test_spec["inference"]["batch_size"] = batch_size
    _test_spec["dataset"]["workers"] = workers
    _test_spec["inference"]["inference_input_type"] = 'image_folder'
    build_inference_dataloader(_test_spec)


@pytest.mark.cv_unit
def test_build_transforms_train(_test_spec):
    _test_spec["model"]["input_width"] = 50
    _test_spec["model"]["input_height"] = 50
    train_dataset_dir = f"{TEST_DATA_DIR}/train/"

    transforms = build_transforms(_test_spec, is_train=True)

    data_set = MetricLearnImageFolder(
        train_dataset_dir,
        transforms)
    assert data_set[0][0].shape == (
        3,
        _test_spec["model"]["input_width"],
        _test_spec["model"]["input_height"]), \
        "Incorrect transform image dimensions."


@pytest.mark.cv_unit
def test_build_transforms_val(_test_spec):
    _test_spec["model"]["input_width"] = 224
    _test_spec["model"]["input_height"] = 256
    test_dataset_dir = f"{TEST_DATA_DIR}/test/"
    transforms = build_transforms(_test_spec, is_train=False)

    data_set = MetricLearnImageFolder(
        test_dataset_dir,
        transforms)
    assert data_set[0][0].shape == (
        3,
        _test_spec["model"]["input_width"],
        _test_spec["model"]["input_height"]), \
        "Incorrect transform image dimensions."
