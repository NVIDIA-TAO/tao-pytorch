# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
StyleGAN-XL Trainer Unit Tests
"""
import os
import torch
import pytest
import tempfile
import numpy as np
from PIL import Image
import json

from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.stylegan_xl.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.sdg.stylegan_xl.model.sx_pl_model import StyleganPlModel
from nvidia_tao_pytorch.sdg.stylegan_xl.dataloader.pl_sx_data_module import SXDataModule


BATCH_SIZE = 5
IMAGE_WIDTH = 16
IMAGE_HEIGHT = 16
IMAGE_CHANNEL = 3
SAMPLES = 10
assert SAMPLES % BATCH_SIZE == 0

UP_FACTOR_LIST = [2, 4]
HEAD_LAYERS_LIST = [3, 7]
STEM_RESOLUTION = 16
SUPERRES_RESOLUTION = 16
for up_factor in UP_FACTOR_LIST:
    SUPERRES_RESOLUTION *= up_factor


@pytest.fixture()
def _test_dir_obj():
    # Create a temporary root directory
    tmp_obj = tempfile.TemporaryDirectory()
    root_dir = tmp_obj.name

    resolution_for_stem_superres = [STEM_RESOLUTION, SUPERRES_RESOLUTION]
    for cur_resolution in resolution_for_stem_superres:
        # Create a folder representing the resolution (e.g., '16')
        resolution_folder_name = str(cur_resolution)
        resolution_dir = os.path.join(root_dir, resolution_folder_name)
        check_and_create(resolution_dir)

        # Create folder '00000' inside the resolution folder for images
        image_folder_name = '00000'
        images_dir = os.path.join(resolution_dir, image_folder_name)
        check_and_create(images_dir)

        # Generate dummy images and dataset.json entries
        image_list = []
        for sample in range(SAMPLES):
            # Generate random image data
            test_data = np.random.rand(cur_resolution, cur_resolution, IMAGE_CHANNEL) * 255
            test_data = test_data.astype(np.uint8)
            im = Image.fromarray(test_data)

            # Save image with the naming format 'img00000000.png'
            save_name = f'img{sample:08d}.png'
            save_path = os.path.join(images_dir, save_name)
            im.save(save_path)

            # Append the image path and label (e.g., 0 as a dummy label) to the dataset
            image_list.append([f"{image_folder_name}/{save_name}", sample])

        # Create 'dataset.json' in the root directory
        dataset = {
            'labels': image_list
        }

        json_path = os.path.join(resolution_dir, 'dataset.json')
        with open(json_path, 'w') as json_file:
            json.dump(dataset, json_file, indent=4)

    # Return the root directory object
    yield tmp_obj


@pytest.fixture
def _test_exp_spec(_test_dir_obj):
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset.stylegan.train_dataset.images_dir = os.path.join(_test_dir_obj.name, str(STEM_RESOLUTION))
    experiment_config.dataset.stylegan.validation_dataset.images_dir = os.path.join(_test_dir_obj.name, str(STEM_RESOLUTION))
    experiment_config.dataset.stylegan.test_dataset.images_dir = os.path.join(_test_dir_obj.name, str(STEM_RESOLUTION))
    experiment_config.dataset.stylegan.infer_dataset.start_seed = 0
    experiment_config.dataset.stylegan.infer_dataset.end_seed = SAMPLES
    experiment_config.dataset.batch_size = BATCH_SIZE
    experiment_config.results_dir = _test_dir_obj.name
    experiment_config.train.num_gpus = 1
    experiment_config.train.num_nodes = 1
    experiment_config.train.num_epochs = 1
    experiment_config.dataset.common.cond = (SAMPLES > 1)
    experiment_config.dataset.common.num_classes = SAMPLES
    experiment_config.dataset.common.img_channels = IMAGE_CHANNEL
    experiment_config.model.stylegan.metrics.num_fake_imgs = SAMPLES
    experiment_config.dataset.common.img_resolution = STEM_RESOLUTION
    experiment_config.model.generator.stem.resolution = STEM_RESOLUTION
    yield experiment_config


def save_tmp_stem_checkpoint(model_save_path, model):
    stem_checkpoint = {'state_dict': model.state_dict()}
    torch.save(stem_checkpoint, model_save_path)

@pytest.mark.skipif(
    os.getenv("CI_PROJECT_DIR", None) is not None,
    reason='Skipping running on CI.'
)
@pytest.mark.parametrize("superres", [False, True])
@pytest.mark.sdg_unit
@pytest.mark.train
def test_trainer_fit(_test_dir_obj, _test_exp_spec, superres):
    precision = '32-true'
    strategy='auto'

    dm = SXDataModule(dataset_config=_test_exp_spec.dataset)

    model = StyleganPlModel(experiment_spec=_test_exp_spec, dm=dm)
    if (superres==True):
        # Using superres dataset
        _test_exp_spec.dataset.stylegan.train_dataset.images_dir = os.path.join(_test_dir_obj.name, str(SUPERRES_RESOLUTION))
        _test_exp_spec.dataset.stylegan.validation_dataset.images_dir = os.path.join(_test_dir_obj.name, str(SUPERRES_RESOLUTION))
        _test_exp_spec.dataset.stylegan.test_dataset.images_dir = os.path.join(_test_dir_obj.name, str(SUPERRES_RESOLUTION))
        dm = SXDataModule(dataset_config=_test_exp_spec.dataset)
        
        for i in range(len(UP_FACTOR_LIST)):
            # Save temp checkpoint
            model_save_path = os.path.join(_test_dir_obj.name, "stem.ckpt")
            save_tmp_stem_checkpoint(model_save_path=model_save_path, model=model)
            # Load a stem model to create supperres model
            _test_exp_spec.model.generator.superres=True
            _test_exp_spec.model.generator.added_head_superres.up_factor=UP_FACTOR_LIST[:i+1]
            _test_exp_spec.model.generator.added_head_superres.head_layers=HEAD_LAYERS_LIST[:i+1]
            _test_exp_spec.dataset.common.img_resolution *= UP_FACTOR_LIST[i]
            _test_exp_spec.model.generator.added_head_superres.pretrained_stem_path=model_save_path
            
            model = StyleganPlModel(experiment_spec=_test_exp_spec, dm=dm)

    trainer = Trainer(devices=_test_exp_spec.train.num_gpus,
                      num_nodes=_test_exp_spec.train.num_nodes,
                      num_sanity_val_steps=0,
                      max_epochs=1,
                      check_val_every_n_epoch=_test_exp_spec.train.validation_interval,
                      default_root_dir=_test_exp_spec.results_dir,
                      accelerator='gpu',
                      strategy=strategy,
                      precision=precision,
                      )

    trainer.fit(model, dm, ckpt_path=None)

@pytest.mark.skipif(
    os.getenv("CI_PROJECT_DIR", None) is not None,
    reason='Skipping running on CI.'
)
@pytest.mark.parametrize("superres", [False, True])
@pytest.mark.sdg_unit
@pytest.mark.evaluate
def test_trainer_evaluate(_test_dir_obj, _test_exp_spec, superres):
    precision = '32-true'
    strategy='auto'

    dm = SXDataModule(dataset_config=_test_exp_spec.dataset)

    model = StyleganPlModel(experiment_spec=_test_exp_spec, dm=dm)
    if (superres==True):
        # Using superres dataset
        _test_exp_spec.dataset.stylegan.train_dataset.images_dir = os.path.join(_test_dir_obj.name, str(SUPERRES_RESOLUTION))
        _test_exp_spec.dataset.stylegan.validation_dataset.images_dir = os.path.join(_test_dir_obj.name, str(SUPERRES_RESOLUTION))
        _test_exp_spec.dataset.stylegan.test_dataset.images_dir = os.path.join(_test_dir_obj.name, str(SUPERRES_RESOLUTION))
        dm = SXDataModule(dataset_config=_test_exp_spec.dataset)
        
        for i in range(len(UP_FACTOR_LIST)):
            # Save temp checkpoint
            model_save_path = os.path.join(_test_dir_obj.name, "stem.ckpt")
            save_tmp_stem_checkpoint(model_save_path=model_save_path, model=model)
            # Load a stem model to create supperres model
            _test_exp_spec.model.generator.superres=True
            _test_exp_spec.model.generator.added_head_superres.up_factor=UP_FACTOR_LIST[:i+1]
            _test_exp_spec.model.generator.added_head_superres.head_layers=HEAD_LAYERS_LIST[:i+1]
            _test_exp_spec.dataset.common.img_resolution *= UP_FACTOR_LIST[i]
            _test_exp_spec.model.generator.added_head_superres.pretrained_stem_path=model_save_path
            
            model = StyleganPlModel(experiment_spec=_test_exp_spec, dm=dm)

    trainer = Trainer(devices=_test_exp_spec.train.num_gpus,
                      num_nodes=_test_exp_spec.train.num_nodes,
                      num_sanity_val_steps=0,
                      max_epochs=1,
                      check_val_every_n_epoch=_test_exp_spec.train.validation_interval,
                      default_root_dir=_test_exp_spec.results_dir,
                      accelerator='gpu',
                      strategy=strategy,
                      precision=precision,
                      )

    trainer.test(model, dm, ckpt_path=None)

@pytest.mark.skipif(
    os.getenv("CI_PROJECT_DIR", None) is not None,
    reason='Skipping running on CI.'
)
@pytest.mark.parametrize("superres", [False, True])
@pytest.mark.sdg_unit
@pytest.mark.inference
def test_trainer_infer(_test_dir_obj, _test_exp_spec, superres):
    precision = '32-true'
    strategy='auto'
    
    _test_exp_spec.inference.results_dir = _test_exp_spec.results_dir
    
    dm = SXDataModule(dataset_config=_test_exp_spec.dataset)

    model = StyleganPlModel(experiment_spec=_test_exp_spec, dm=dm)
    if (superres==True):
        # Using superres dataset
        _test_exp_spec.dataset.stylegan.train_dataset.images_dir = os.path.join(_test_dir_obj.name, str(SUPERRES_RESOLUTION))
        _test_exp_spec.dataset.stylegan.validation_dataset.images_dir = os.path.join(_test_dir_obj.name, str(SUPERRES_RESOLUTION))
        _test_exp_spec.dataset.stylegan.test_dataset.images_dir = os.path.join(_test_dir_obj.name, str(SUPERRES_RESOLUTION))
        dm = SXDataModule(dataset_config=_test_exp_spec.dataset)
        
        for i in range(len(UP_FACTOR_LIST)):
            # Save temp checkpoint
            model_save_path = os.path.join(_test_dir_obj.name, "stem.ckpt")
            save_tmp_stem_checkpoint(model_save_path=model_save_path, model=model)
            # Load a stem model to create supperres model
            _test_exp_spec.model.generator.superres=True
            _test_exp_spec.model.generator.added_head_superres.up_factor=UP_FACTOR_LIST[:i+1]
            _test_exp_spec.model.generator.added_head_superres.head_layers=HEAD_LAYERS_LIST[:i+1]
            _test_exp_spec.dataset.common.img_resolution *= UP_FACTOR_LIST[i]
            _test_exp_spec.model.generator.added_head_superres.pretrained_stem_path=model_save_path
            
            model = StyleganPlModel(experiment_spec=_test_exp_spec, dm=dm)

    trainer = Trainer(devices=_test_exp_spec.train.num_gpus,
                      num_nodes=_test_exp_spec.train.num_nodes,
                      num_sanity_val_steps=0,
                      max_epochs=1,
                      check_val_every_n_epoch=_test_exp_spec.train.validation_interval,
                      default_root_dir=_test_exp_spec.results_dir,
                      accelerator='gpu',
                      strategy=strategy,
                      precision=precision,
                      )

    trainer.predict(model, dm, ckpt_path=None)

    _test_dir_obj.cleanup()
