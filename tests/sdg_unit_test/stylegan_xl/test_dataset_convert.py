# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
StyleGAN-XL Dataset Convert Unit Tests
"""

import os
import pytest
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import tempfile
import json

from nvidia_tao_pytorch.config.stylegan_xl.dataset import DataConvertExpConfig
from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.sdg.stylegan_xl.scripts.dataset_convert import run_experiment


# Constants
NUM_FOLDERS = 5          # Number of folders to create
SAMPLES = 10             # Number of images per folder
IMAGE_HEIGHT = 128       # Image height
IMAGE_WIDTH = 128        # Image width
IMAGE_CHANNEL = 3        # Image channels (e.g., RGB)

def check_and_create(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


@pytest.fixture()
def _test_dir_obj():
    # Create a temporary root directory
    tmp_obj = tempfile.TemporaryDirectory()
    root_dir = tmp_obj.name

    # Create 'source' subfolder under root
    source_dir = os.path.join(root_dir, "source")
    check_and_create(source_dir)
    # Loop to create multiple folders inside 'source'
    for folder_index in range(NUM_FOLDERS):
        # Create folder (e.g., '00000', '00001', ...)
        folder_name = f'{folder_index:05d}'  # Format: '00000', '00001', etc.
        images_dir = os.path.join(source_dir, folder_name)
        check_and_create(images_dir)
        
        # Generate dummy images for this folder
        for sample in range(SAMPLES):
            test_data = np.random.rand(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL) * 255
            test_data = test_data.astype(np.uint8)
            im = Image.fromarray(test_data)
            
            # Save image with the naming format 'img00000000.png'
            save_name = f'img{sample:08d}.png'
            save_path = os.path.join(images_dir, save_name)
            im.save(save_path)

    # Return the root directory object
    yield tmp_obj


@pytest.fixture
def _test_exp_spec(_test_dir_obj):
    experiment_config = OmegaConf.structured(DataConvertExpConfig())
    experiment_config.results_dir = _test_dir_obj.name
    experiment_config.source = os.path.join(_test_dir_obj.name, "source")
    experiment_config.dest_file_name = "dest.zip"
    experiment_config.resolution = (512, 512)
    yield experiment_config


@pytest.mark.parametrize("transform", [None, 'center-crop'])
@pytest.mark.sdg_unit
def test_dataset_convert(_test_dir_obj, _test_exp_spec, transform):
    _test_exp_spec.transform = transform
    run_experiment(_test_exp_spec, _test_exp_spec.results_dir)

    _test_dir_obj.cleanup()
