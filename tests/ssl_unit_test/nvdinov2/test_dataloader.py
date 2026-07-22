# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NVDINOv2 Dataloader Unit Tests
"""
import os
import pytest
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import tempfile

from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.config.nvdinov2.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.nvdinov2.dataloader.pl_dinov2_data_module import DinoV2DataModule

BATCH_SIZE = 2
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
SAMPLES = 10

@pytest.fixture()
def _test_dir_obj():
    tmp_obj = tempfile.TemporaryDirectory()
    check_and_create(tmp_obj.name)
    for sample in range(SAMPLES):
        test_data = np.random.rand(IMAGE_HEIGHT, IMAGE_WIDTH, 3) * 255
        test_data = test_data.astype(np.uint8)
        im = Image.fromarray(test_data)
        save_name = f'test_{sample}.jpg'
        im.save(os.path.join(tmp_obj.name, save_name))

    yield tmp_obj
    
@pytest.fixture
def _test_exp_spec(_test_dir_obj):
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset.train_dataset.images_dir = _test_dir_obj.name
    experiment_config.dataset.batch_size = BATCH_SIZE
    experiment_config.results_dir = _test_dir_obj.name
    yield experiment_config

@pytest.mark.ssl_unit
def test_nvdionv2_dataloader(_test_dir_obj, _test_exp_spec):
    data_module = DinoV2DataModule(experiment_config=_test_exp_spec)
    data_module.setup('fit')
    loader = data_module.train_dataloader()
    
    for batch in loader:
        assert batch['global_crops'].shape[0] == BATCH_SIZE * _test_exp_spec["dataset"]["transform"]["n_global_crops"], "Incorrect batch size of global crops"
        assert batch['global_crops'].shape[2] == _test_exp_spec["dataset"]["transform"]["global_crops_size"], "Incorrect height of global crops"
        assert batch['global_crops'].shape[3] == _test_exp_spec["dataset"]["transform"]["global_crops_size"], "Incorrect width of global crops"
        assert batch['local_crops'].shape[0] == BATCH_SIZE * _test_exp_spec["dataset"]["transform"]["n_local_crops"], "Incorrect batch size of local crops"
        assert batch['local_crops'].shape[2] == _test_exp_spec["dataset"]["transform"]["local_crops_size"], "Incorrect height of local crops"
        assert batch['local_crops'].shape[3] == _test_exp_spec["dataset"]["transform"]["local_crops_size"], "Incorrect width of local crops"
    
    _test_dir_obj.cleanup()
