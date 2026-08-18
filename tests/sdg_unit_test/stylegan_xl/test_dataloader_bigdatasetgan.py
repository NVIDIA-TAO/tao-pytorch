# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
BigDatasetGAN Dataloader Unit Tests
"""
import os
import pytest
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import tempfile

from nvidia_tao_pytorch.config.stylegan_xl.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.sdg.stylegan_xl.dataloader.pl_bg_data_module import BGDataModule


BATCH_SIZE = 5
IMAGE_WIDTH = 16
IMAGE_HEIGHT = 16
IMAGE_CHANNEL = 1  # binary mask
SAMPLES = 10
assert SAMPLES % BATCH_SIZE == 0


@pytest.fixture()
def _test_dir_obj():
    # Create a temporary root directory
    tmp_obj = tempfile.TemporaryDirectory()
    root_dir = tmp_obj.name

    # Generate dummy images
    for sample in range(SAMPLES):
        test_data = (np.random.rand(IMAGE_HEIGHT, IMAGE_WIDTH) > 0.5).astype(np.uint8) * 255  # binary mask, only have pixel value 0 and 255
        test_data = test_data.astype(np.uint8)
        im = Image.fromarray(test_data)
        
        # Save image with the naming format 'seed.png'
        save_name = f'seed{sample:d}.png'  # Format: seed0.png, seed2.png, etc.
        save_path = os.path.join(root_dir, save_name)
        im.save(save_path)

    # Return the root directory object
    yield tmp_obj


@pytest.fixture
def _test_exp_spec(_test_dir_obj):
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset.bigdatasetgan.train_dataset.images_dir = _test_dir_obj.name
    experiment_config.dataset.bigdatasetgan.validation_dataset.images_dir = _test_dir_obj.name
    experiment_config.dataset.bigdatasetgan.test_dataset.images_dir = _test_dir_obj.name
    experiment_config.dataset.batch_size = BATCH_SIZE
    experiment_config.dataset.common.img_resolution = IMAGE_WIDTH
    experiment_config.results_dir = _test_dir_obj.name
    yield experiment_config


@pytest.mark.parametrize("stage", ['fit', 'test', 'predict'])
@pytest.mark.sdg_unit
def test_build_dataloader(_test_dir_obj, _test_exp_spec, stage):
    data_module = BGDataModule(dataset_config=_test_exp_spec.dataset)
    class Trainer():
        def __init__(self):
            self.world_size=1
    data_module.trainer = Trainer()
    data_module.setup(stage)

    if (stage == 'fit'):
        loader = data_module.train_dataloader()
    elif (stage == 'test'):
        loader = data_module.test_dataloader()
    elif (stage == 'predict'):
        loader = data_module.predict_dataloader()
    else:
        raise NotImplementedError

    for batch in loader:
        if (stage != 'predict'):
            assert len(batch) == 2, "incorrect batch, batch should only contain 'image' and 'label'"
            assert batch[0].shape[0] == BATCH_SIZE, "Incorrect batch size"
            assert batch[1].shape[0] == BATCH_SIZE, "Incorrect batch size"
            assert len(batch[1].shape) == 3, "Current only support binary segmentation/mask. The mask size should be (B, H, W)"
            assert batch[1].shape[1] == batch[1].shape[2] == _test_exp_spec["dataset"]['common']['img_resolution'], "Incorrect image resolution"
        else:
            assert batch.shape[0] == BATCH_SIZE, "Incorrect batch size"
    _test_dir_obj.cleanup()
