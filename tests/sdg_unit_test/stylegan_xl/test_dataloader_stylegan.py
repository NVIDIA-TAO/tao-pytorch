# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
StyleGAN-XL Dataloader Unit Tests
"""
import os
import pytest
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import tempfile
import json

from nvidia_tao_pytorch.config.stylegan_xl.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.sdg.stylegan_xl.dataloader.pl_sx_data_module import SXDataModule


BATCH_SIZE = 5
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNEL = 3
SAMPLES = 10
assert IMAGE_WIDTH == IMAGE_HEIGHT
assert SAMPLES % BATCH_SIZE == 0


@pytest.fixture()
def _test_dir_obj():
    # Create a temporary root directory
    tmp_obj = tempfile.TemporaryDirectory()
    root_dir = tmp_obj.name
    # Create folder '00000' inside the root directory for images
    image_folder_name = '00000'
    images_dir = os.path.join(root_dir, image_folder_name)
    check_and_create(images_dir)

    # Generate dummy images and dataset.json entries
    image_list = []
    for sample in range(SAMPLES):
        test_data = np.random.rand(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL) * 255
        test_data = test_data.astype(np.uint8)
        im = Image.fromarray(test_data)
        
        # Save image with the naming format 'img00000000.png'
        save_name = f'img{sample:08d}.png'  # Format: img00000000.png, img00000001.png, etc.
        save_path = os.path.join(images_dir, save_name)
        im.save(save_path)
        # Append the image path and label (e.g., 0 as a dummy label) to the dataset
        image_list.append([f"{image_folder_name}/{save_name}", sample])

    # Create 'dataset.json' in the root directory
    dataset = {
        'labels': image_list
    }

    json_path = os.path.join(root_dir, 'dataset.json')
    with open(json_path, 'w') as json_file:
        json.dump(dataset, json_file, indent=4)

    # Return the root directory object
    yield tmp_obj


@pytest.fixture
def _test_exp_spec(_test_dir_obj):
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset.stylegan.train_dataset.images_dir = _test_dir_obj.name
    experiment_config.dataset.stylegan.validation_dataset.images_dir = _test_dir_obj.name
    experiment_config.dataset.stylegan.test_dataset.images_dir = _test_dir_obj.name
    experiment_config.dataset.stylegan.infer_dataset.start_seed = 0
    experiment_config.dataset.stylegan.infer_dataset.end_seed = SAMPLES
    experiment_config.dataset.common.cond = (SAMPLES > 1)
    experiment_config.dataset.common.num_classes = SAMPLES
    experiment_config.dataset.common.img_channels = IMAGE_CHANNEL
    experiment_config.dataset.common.img_resolution = IMAGE_WIDTH
    experiment_config.dataset.batch_size = BATCH_SIZE
    experiment_config.results_dir = _test_dir_obj.name
    yield experiment_config


@pytest.mark.parametrize("stage", ['fit', 'test', 'predict'])
@pytest.mark.sdg_unit
def test_build_dataloader(_test_dir_obj, _test_exp_spec, stage):
    data_module = SXDataModule(dataset_config=_test_exp_spec.dataset)
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
            assert batch[0].shape[1] == _test_exp_spec["dataset"]['common']['img_channels'], "Incorrect image channels"
            assert batch[0].shape[2] == batch[0].shape[3] == _test_exp_spec["dataset"]['common']['img_resolution'], "Incorrect image resolution"
            assert batch[1].shape[0] == BATCH_SIZE, "Incorrect batch size"
            assert batch[1].shape[1] == _test_exp_spec["dataset"]['common']['num_classes'], "Incorrect number of classes"
        else:
            assert batch.shape[0] == BATCH_SIZE, "Incorrect batch size"

    _test_dir_obj.cleanup()
