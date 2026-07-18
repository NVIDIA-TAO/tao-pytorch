# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import pytest
import torchvision.transforms as transforms
from random import randint
from PIL import Image
from omegaconf import OmegaConf
from nvidia_tao_pytorch.cv.re_identification.dataloader.build_data_loader import train_collate_fn, val_collate_fn, list_dataset, build_dataloader
from nvidia_tao_pytorch.config.re_identification.default_config import ReIDModelConfig, ReIDTrainExpConfig, ReIDDatasetConfig, ReIDInferenceExpConfig

@pytest.fixture
def _test_dir():
    os.system('tar -xvf /home/scratch.metropolis2/tao_ci/tao_pytorch/data/re_identification/test_data.tar.xz --directory tests/cv_unit_test/re_identification/')
    tmp_top_dir = "tests/cv_unit_test/re_identification/test_data"
    yield tmp_top_dir
    shutil.rmtree(tmp_top_dir)

@pytest.fixture
def _test_sample_dict(_test_dir):
    bbox_test_dir = os.path.join(_test_dir, "bounding_box_test")
    sample_dict = list_dataset(bbox_test_dir)
    yield sample_dict

@pytest.fixture
def _test_spec(_test_dir):
    model_config = OmegaConf.structured(ReIDModelConfig())
    train_config = OmegaConf.structured(ReIDTrainExpConfig())
    dataset_config = OmegaConf.structured(ReIDDatasetConfig())
    inference_config = OmegaConf.structured(ReIDInferenceExpConfig())
    spec = {"model": model_config,
            "train": train_config,
            "dataset": dataset_config,
            "inference": inference_config}
    yield spec

@pytest.fixture
def _test_batch(_test_dir):
    pids = [randint(0, 10) for _ in range(17)]
    camids = [randint(0, 10) for _ in range(17)]
    batch = []
    bbox_test_dir = os.path.join(_test_dir, "bounding_box_test")

    for index, file_img in enumerate(os.listdir(bbox_test_dir)):
        img = Image.open(os.path.join(bbox_test_dir, file_img))
        img = img.resize((128,256))
        transform = transforms.Compose([transforms.PILToTensor()])
        img  = transform(img)
        batch.append([img, pids[index], camids[index], file_img])
    yield batch

@pytest.mark.cv_unit
def test_list_dataset(_test_dir):
    bbox_test_dir = os.path.join(_test_dir, "bounding_box_test")
    sample_dict = list_dataset(bbox_test_dir)
    assert len(sample_dict.keys()) == 16, "The test folder should have 16 images."

@pytest.mark.cv_unit
def test_train_collate_fn(_test_batch):
    assert train_collate_fn(_test_batch)[0].shape == (16, 3, 256, 128), "Incorrect batch size."
    assert len(train_collate_fn(_test_batch)) == 2, "Incorrect length size."

@pytest.mark.cv_unit
def test_val_collate_fn(_test_batch):
    assert val_collate_fn(_test_batch)[0].shape == (16, 3, 256, 128), "Incorrect batch size."
    assert len(val_collate_fn(_test_batch)) == 4, "Incorrect length size."

@pytest.mark.cv_unit
@pytest.mark.parametrize("is_train, batch_size, num_workers",
                         [(False, 4, 8),
                          (False ,8, 16),
                          (True, 16, 8)])
def test_build_dataloader(_test_dir, _test_spec, is_train, batch_size, num_workers):
    _test_spec["dataset"].train_dataset_dir = os.path.join(_test_dir, "bounding_box_train")
    _test_spec["dataset"].test_dataset_dir = os.path.join(_test_dir, "bounding_box_test")
    _test_spec["dataset"].query_dataset_dir = os.path.join(_test_dir, "query")
    _test_spec["dataset"].batch_size = batch_size
    _test_spec["dataset"].num_workers = num_workers
    _test_spec["inference"]["test_dataset"] = os.path.join(_test_dir, "gt_query")
    _test_spec["inference"]["query_dataset"] = os.path.join(_test_dir, "bounding_box_test")
    build_dataloader(_test_spec, is_train)
