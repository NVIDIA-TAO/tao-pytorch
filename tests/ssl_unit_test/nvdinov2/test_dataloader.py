# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NVDINOv2 Dataloader Unit Tests
"""
import os
import pytest
import tarfile
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import tempfile

from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.config.nvdinov2.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.nvdinov2.dataloader.dataset import DinoV2Dataset
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


def _dummy_transform(image):
    return image


def _save_test_image(path):
    test_data = (np.random.rand(1, 1, 3) * 255).astype(np.uint8)
    Image.fromarray(test_data).save(path)


@pytest.mark.ssl_unit
def test_nvdinov2_dataset_rejects_archive_images_dir(tmp_path):
    """Regression test for bug 6460966: images_dir pointing at a .tar.gz archive."""
    image_path = tmp_path / 'test_0.jpg'
    _save_test_image(image_path)
    archive_path = tmp_path / 'images.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(image_path, arcname='test_0.jpg')

    with pytest.raises(ValueError, match='archive'):
        DinoV2Dataset(root=archive_path, transform=_dummy_transform)


@pytest.mark.ssl_unit
def test_nvdinov2_dataset_nonexistent_images_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        DinoV2Dataset(root=tmp_path / 'does_not_exist', transform=_dummy_transform)


@pytest.mark.ssl_unit
def test_nvdinov2_dataset_empty_images_dir(tmp_path):
    with pytest.raises(ValueError, match='No images found'):
        DinoV2Dataset(root=tmp_path, transform=_dummy_transform)


@pytest.mark.ssl_unit
def test_nvdinov2_dataset_extracted_images_dir(tmp_path):
    _save_test_image(tmp_path / 'test_0.jpg')
    dataset = DinoV2Dataset(root=tmp_path, transform=_dummy_transform)
    assert len(dataset) == 1
