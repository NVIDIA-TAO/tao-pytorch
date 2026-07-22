# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
BigDatasetGAN Trainer Unit Tests
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
from nvidia_tao_pytorch.sdg.stylegan_xl.model.bg_pl_model import BigdatasetganPlModel
from nvidia_tao_pytorch.sdg.stylegan_xl.dataloader.pl_bg_data_module import BGDataModule


BATCH_SIZE = 5
IMAGE_WIDTH = 16
IMAGE_HEIGHT = 16
IMAGE_CHANNEL = 1  # binary mask
SAMPLES = 10
assert SAMPLES % BATCH_SIZE == 0
assert IMAGE_WIDTH == IMAGE_HEIGHT
image_folder_name = 'data'
# Currently, segmentation map generated from BigDatasetGAN is fixed to (512, 512)
SEG_WIDTH = 512
SEG_HEIGHT = 512
# Super resolution parameters for StyleGAN-XL
UP_FACTOR_LIST = [2, 4]
HEAD_LAYERS_LIST = [3, 7]
# Hyper-parameters for BigDatasetGAN's feature extractor
BIGDATASETGAN_FEATURE_BLOCKS = [1, 2, 3, 4]


@pytest.fixture()
def _test_dir_obj():
    # Create a temporary root directory
    tmp_obj = tempfile.TemporaryDirectory()
    root_dir = tmp_obj.name
    # Create folder 'data' inside the root directory for images
    images_dir = os.path.join(root_dir, image_folder_name)
    check_and_create(images_dir)

    # Generate dummy images
    for sample in range(SAMPLES):
        test_data = (np.random.rand(SEG_HEIGHT, SEG_WIDTH) > 0.5).astype(np.uint8) * 255  # binary mask, only have pixel value 0 and 255
        test_data = test_data.astype(np.uint8)
        im = Image.fromarray(test_data)
        
        # Save image with the naming format 'seed.png'
        save_name = f'seed{sample:d}.png'  # Format: seed0.png, seed2.png, etc.
        save_path = os.path.join(images_dir, save_name)
        im.save(save_path)

    # Return the root directory object
    yield tmp_obj


@pytest.fixture
def _test_exp_spec(_test_dir_obj):
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.results_dir = _test_dir_obj.name
    experiment_config.dataset.bigdatasetgan.train_dataset.images_dir = os.path.join(_test_dir_obj.name, image_folder_name)
    experiment_config.dataset.bigdatasetgan.validation_dataset.images_dir = os.path.join(_test_dir_obj.name, image_folder_name)
    experiment_config.dataset.bigdatasetgan.test_dataset.images_dir = os.path.join(_test_dir_obj.name, image_folder_name)
    experiment_config.dataset.bigdatasetgan.infer_dataset.start_seed = 0
    experiment_config.dataset.bigdatasetgan.infer_dataset.end_seed = SAMPLES
    experiment_config.dataset.batch_size = BATCH_SIZE
    experiment_config.dataset.common.img_resolution = IMAGE_WIDTH
    experiment_config.train.num_gpus = 1
    experiment_config.train.num_nodes = 1
    experiment_config.train.num_epochs = 1
    experiment_config.train.bigdatasetgan.optim_labeller.optim = 'AdamW'
    yield experiment_config


def save_tmp_stem_checkpoint(model_save_path, model):
    stem_checkpoint = {'state_dict': model.state_dict()}
    torch.save(stem_checkpoint, model_save_path)


@pytest.mark.parametrize("superres", [False, True])
@pytest.mark.sdg_unit
@pytest.mark.train
def test_trainer_fit(_test_dir_obj, _test_exp_spec, superres):
    precision = '32-true'
    strategy='auto'
    
    # Create dataloader
    dm = BGDataModule(dataset_config=_test_exp_spec.dataset)
    
    # Create model
    root_dir = _test_dir_obj.name
    # Create temp checkpoint
    _test_exp_spec.model.generator.stem.resolution = IMAGE_WIDTH
    model = StyleganPlModel(experiment_spec=_test_exp_spec, dm=None)
    # Save temp checkpoint
    model_save_path = os.path.join(root_dir, "stem.ckpt")
    save_tmp_stem_checkpoint(model_save_path=model_save_path, model=model)
    # Hyper-parameters for BigDatasetGAN's feature extractor
    _test_exp_spec.model.bigdatasetgan.feature_extractor.blocks=BIGDATASETGAN_FEATURE_BLOCKS
    if (superres==True):
        for i in range(len(UP_FACTOR_LIST)):
            # Load a stem model to create supperres model
            _test_exp_spec.model.generator.superres=True
            _test_exp_spec.model.generator.added_head_superres.up_factor=UP_FACTOR_LIST[:i+1]
            _test_exp_spec.model.generator.added_head_superres.head_layers=HEAD_LAYERS_LIST[:i+1]
            _test_exp_spec.dataset.common.img_resolution *= UP_FACTOR_LIST[i]
            # Load stylegan checkpoint as super resolution stylegan's stem
            _test_exp_spec.model.generator.added_head_superres.pretrained_stem_path=model_save_path
            model = StyleganPlModel(experiment_spec=_test_exp_spec, dm = None)
            # Save temp checkpoint
            model_save_path = os.path.join(root_dir, "stem.ckpt")
            save_tmp_stem_checkpoint(model_save_path=model_save_path, model=model)
            # Load stylegan checkpoint as bigdatasetgan's feature extractor
            _test_exp_spec.model.bigdatasetgan.feature_extractor.stylegan_checkpoint_path=model_save_path
            model = BigdatasetganPlModel(experiment_spec=_test_exp_spec, dm=None)
    else:
        # Load stylegan checkpoint as bigdatasetgan's feature extractor
        _test_exp_spec.model.bigdatasetgan.feature_extractor.stylegan_checkpoint_path=model_save_path
        model = BigdatasetganPlModel(experiment_spec=_test_exp_spec, dm=None)
    
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


@pytest.mark.parametrize("superres", [False, True])
@pytest.mark.sdg_unit
@pytest.mark.evaluate
def test_trainer_evaluate(_test_dir_obj, _test_exp_spec, superres):
    precision = '32-true'
    strategy='auto'
    
    # Create dataloader
    dm = BGDataModule(dataset_config=_test_exp_spec.dataset)
    
    # Create model
    root_dir = _test_dir_obj.name
    # Create temp checkpoint
    _test_exp_spec.model.generator.stem.resolution = IMAGE_WIDTH
    model = StyleganPlModel(experiment_spec=_test_exp_spec, dm=None)
    # Save temp checkpoint
    model_save_path = os.path.join(root_dir, "stem.ckpt")
    save_tmp_stem_checkpoint(model_save_path=model_save_path, model=model)
    # Hyper-parameters for BigDatasetGAN's feature extractor
    _test_exp_spec.model.bigdatasetgan.feature_extractor.blocks=BIGDATASETGAN_FEATURE_BLOCKS
    if (superres==True):
        for i in range(len(UP_FACTOR_LIST)):
            # Load a stem model to create supperres model
            _test_exp_spec.model.generator.superres=True
            _test_exp_spec.model.generator.added_head_superres.up_factor=UP_FACTOR_LIST[:i+1]
            _test_exp_spec.model.generator.added_head_superres.head_layers=HEAD_LAYERS_LIST[:i+1]
            _test_exp_spec.dataset.common.img_resolution *= UP_FACTOR_LIST[i]
            # Load stylegan checkpoint as super resolution stylegan's stem
            _test_exp_spec.model.generator.added_head_superres.pretrained_stem_path=model_save_path
            model = StyleganPlModel(experiment_spec=_test_exp_spec, dm = None)
            # Save temp checkpoint
            model_save_path = os.path.join(root_dir, "stem.ckpt")
            save_tmp_stem_checkpoint(model_save_path=model_save_path, model=model)
            # Load stylegan checkpoint as bigdatasetgan's feature extractor
            _test_exp_spec.model.bigdatasetgan.feature_extractor.stylegan_checkpoint_path=model_save_path
            model = BigdatasetganPlModel(experiment_spec=_test_exp_spec, dm=None)
    else:
        # Load stylegan checkpoint as bigdatasetgan's feature extractor
        _test_exp_spec.model.bigdatasetgan.feature_extractor.stylegan_checkpoint_path=model_save_path
        model = BigdatasetganPlModel(experiment_spec=_test_exp_spec, dm=None)
    
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


@pytest.mark.parametrize("superres", [False, True])
@pytest.mark.sdg_unit
@pytest.mark.inference
def test_trainer_infer(_test_dir_obj, _test_exp_spec, superres):
    _test_exp_spec.inference.results_dir = _test_exp_spec.results_dir

    precision = '32-true'
    strategy='auto'
    
    # Create dataloader
    dm = BGDataModule(dataset_config=_test_exp_spec.dataset)
    
    # Create model
    root_dir = _test_dir_obj.name
    # Create temp checkpoint
    _test_exp_spec.model.generator.stem.resolution = IMAGE_WIDTH
    model = StyleganPlModel(experiment_spec=_test_exp_spec, dm=None)
    # Save temp checkpoint
    model_save_path = os.path.join(root_dir, "stem.ckpt")
    save_tmp_stem_checkpoint(model_save_path=model_save_path, model=model)
    # Hyper-parameters for BigDatasetGAN's feature extractor
    _test_exp_spec.model.bigdatasetgan.feature_extractor.blocks=BIGDATASETGAN_FEATURE_BLOCKS
    if (superres==True):
        for i in range(len(UP_FACTOR_LIST)):
            # Load a stem model to create supperres model
            _test_exp_spec.model.generator.superres=True
            _test_exp_spec.model.generator.added_head_superres.up_factor=UP_FACTOR_LIST[:i+1]
            _test_exp_spec.model.generator.added_head_superres.head_layers=HEAD_LAYERS_LIST[:i+1]
            _test_exp_spec.dataset.common.img_resolution *= UP_FACTOR_LIST[i]
            # Load stylegan checkpoint as super resolution stylegan's stem
            _test_exp_spec.model.generator.added_head_superres.pretrained_stem_path=model_save_path
            model = StyleganPlModel(experiment_spec=_test_exp_spec, dm = None)
            # Save temp checkpoint
            model_save_path = os.path.join(root_dir, "stem.ckpt")
            save_tmp_stem_checkpoint(model_save_path=model_save_path, model=model)
            # Load stylegan checkpoint as bigdatasetgan's feature extractor
            _test_exp_spec.model.bigdatasetgan.feature_extractor.stylegan_checkpoint_path=model_save_path
            model = BigdatasetganPlModel(experiment_spec=_test_exp_spec, dm=None)
    else:
        # Load stylegan checkpoint as bigdatasetgan's feature extractor
        _test_exp_spec.model.bigdatasetgan.feature_extractor.stylegan_checkpoint_path=model_save_path
        model = BigdatasetganPlModel(experiment_spec=_test_exp_spec, dm=None)
    
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
