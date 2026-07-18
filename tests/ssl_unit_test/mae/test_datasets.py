# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from PIL import Image
import tempfile
import numpy as np
import pytest

from torchvision import datasets
import torch

from nvidia_tao_pytorch.ssl.mae.dataloader.datasets import PretrainDataset, FinetuneDataset, PredictDataset

# Mocking the configuration for testing
class Config:
    class dataset:
        train_data_sources = '/path/to/train/img/dir'
        val_data_sources = '/path/to/val/img/dir'
        test_data_sources = '/path/to/test/img/dir'
        class augmentation:
            input_size = 224
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            min_scale = 0.1
            max_scale = 2.
            min_ratio = 3/4
            max_ratio = 4/3
            hflip = 0.5
            re_prob = 0.0
            interpolation = "random"
            color_jitter = 1
            auto_aug = "rand-m9-mstd0.5-inc1"
            norm_pix_loss = True

@pytest.fixture
def image_dir(tmp_path_factory):

    tmp_dir = str(tmp_path_factory.mktemp('random'))
    for class_name in ['c1', 'c2', 'c3']:
        random_array = np.random.random((224, 224, 3)) * 255
        random_array = random_array.astype(np.uint8)
        img = Image.fromarray(random_array)
        class_dir = tempfile.mkdtemp(dir=tmp_dir)
        fn = class_dir + '/' + f"{class_name}.png"
        img.save(fn)

    return tmp_dir

@pytest.fixture
def cfg(image_dir):
    tmp_cfg = Config()
    tmp_cfg.dataset.train_data_sources = image_dir
    tmp_cfg.dataset.val_data_sources = image_dir
    tmp_cfg.dataset.test_data_sources = image_dir
    return tmp_cfg

def test_pretrain_dataset_init(cfg):
    dataset = PretrainDataset(cfg=cfg, is_training=True)
    assert dataset.cfg == cfg
    assert dataset.is_training == True
    assert dataset.img_dir == cfg.dataset.train_data_sources

def test_pretrain_dataset_len(cfg):
    dataset = PretrainDataset(cfg=cfg, is_training=True)
    assert len(dataset) > 0

def test_pretrain_dataset_getitem(cfg):
    dataset = PretrainDataset(cfg=cfg, is_training=True)
    item = dataset[0]
    assert 'image' in item
    assert isinstance(item['image'], torch.Tensor)

def test_pretrain_dataset_collate_fn(cfg):
    dataset = PretrainDataset(cfg=cfg, is_training=True)
    batch = [dataset[0], dataset[1]]
    collated_batch = dataset.collate_fn(batch)
    assert 'images' in collated_batch
    assert isinstance(collated_batch['images'], torch.Tensor)

# Test FinetuneDataset
def test_finetune_dataset_init(cfg):
    dataset = FinetuneDataset(cfg=cfg, is_training=True)
    assert dataset.cfg == cfg
    assert dataset.is_training == True

def test_finetune_dataset_build(cfg):
    dataset = FinetuneDataset(cfg=cfg, is_training=True)
    built_dataset = dataset.build()
    assert isinstance(built_dataset, datasets.ImageFolder)

# Test PredictDataset
def test_predict_dataset_init(cfg):
    dataset = PredictDataset(cfg=cfg)
    assert dataset.cfg == cfg

def test_predict_dataset_len(cfg):
    dataset = PredictDataset(cfg=cfg)
    assert len(dataset) > 0

def test_predict_dataset_getitem(cfg):
    dataset = PredictDataset(cfg=cfg)
    item = dataset[0]
    assert 'image' in item
    assert isinstance(item['image'], torch.Tensor)
    assert 'image_path' in item
    assert isinstance(item['image_path'], str)

def test_predict_dataset_collate_fn(cfg):
    dataset = PredictDataset(cfg=cfg)
    batch = [dataset[0], dataset[1]]
    collated_batch = dataset.collate_fn(batch)
    assert 'images' in collated_batch
    assert isinstance(collated_batch['images'], torch.Tensor)
    assert 'paths' in collated_batch
    assert isinstance(collated_batch['paths'], list)
