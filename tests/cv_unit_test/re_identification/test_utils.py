# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import pytest
import torch
from itertools import groupby
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from nvidia_tao_pytorch.core.connectors.checkpoint_connector import encrypt_checkpoint
from nvidia_tao_pytorch.core.utilities import patch_decrypt_checkpoint
from nvidia_tao_pytorch.config.re_identification.default_config import ReIDModelConfig, ReIDDatasetConfig, ReIDTrainExpConfig
from nvidia_tao_pytorch.cv.re_identification.dataloader.build_data_loader import train_collate_fn
from nvidia_tao_pytorch.cv.re_identification.dataloader.datasets.bases import ImageDataset
from nvidia_tao_pytorch.cv.re_identification.dataloader.datasets.market1501 import Market1501
from nvidia_tao_pytorch.cv.re_identification.dataloader.transforms import build_transforms
from nvidia_tao_pytorch.cv.re_identification.dataloader.sampler import RandomIdentitySampler
from nvidia_tao_pytorch.cv.re_identification.model.losses.triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from nvidia_tao_pytorch.cv.re_identification.model.losses.center_loss import CenterLoss
from nvidia_tao_pytorch.cv.re_identification.model.losses.metric_learning import ContrastiveLoss, CircleLoss, Arcface, Cosface, AMSoftmax
from nvidia_tao_pytorch.cv.re_identification.utils.reid_metric import euclidean_distance, cosine_similarity


@pytest.mark.cv_unit
def test_patch_decrypt_ckpt():
    fake_ckpt = {"state_dict": {"a": 1, "b": 2, "c": 3}}

    encrypted_ckpt = encrypt_checkpoint(fake_ckpt, "tao")

    assert encrypted_ckpt["state_dict_encrypted"] is not False, f"encrypted_ckpt[\"state_dict_encrypted\"] "\
        f"should be True. Got False."

    decrypted_ckpt = patch_decrypt_checkpoint(encrypted_ckpt, "tao")

    assert decrypted_ckpt["state_dict_encrypted"] is False, f"encrypted_ckpt[\"state_dict_encrypted\"] "\
        f"should be False. Got True."

@pytest.fixture
def _test_dir():
    os.system('tar -xvf /home/scratch.metropolis2/tao_ci/tao_pytorch/data/re_identification/test_data.tar.xz --directory tests/cv_unit_test/re_identification/')
    tmp_top_dir = "tests/cv_unit_test/re_identification/test_data"
    yield tmp_top_dir
    shutil.rmtree(tmp_top_dir)

@pytest.fixture
def _test_experiment_spec():
    model_config = OmegaConf.structured(ReIDModelConfig())
    dataset_config = OmegaConf.structured(ReIDDatasetConfig())
    train_config = OmegaConf.structured(ReIDTrainExpConfig())
    experiment_config = {"model": model_config,
                         "dataset": dataset_config,
                         "train": train_config}
    yield experiment_config

@pytest.mark.cv_unit
def test_build_transforms(_test_dir, _test_experiment_spec):
    _test_experiment_spec["model"].input_width = 50
    _test_experiment_spec["model"].input_height = 50
    _test_experiment_spec["dataset"].train_dataset_dir = os.path.join(_test_dir, "bounding_box_train")
    _test_experiment_spec["dataset"].test_dataset_dir = os.path.join(_test_dir, "bounding_box_test")
    _test_experiment_spec["dataset"].query_dataset_dir = os.path.join(_test_dir, "query")
    transforms = build_transforms(_test_experiment_spec)
    dataset = Market1501(_test_experiment_spec, True)

    data_set = ImageDataset(dataset.train, transforms)
    assert data_set[0][0].shape == (3, _test_experiment_spec["model"].input_height, _test_experiment_spec["model"].input_width),\
        "Incorrect transform image dimensions."

@pytest.mark.cv_unit
def test_random_identity_sampler(_test_dir, _test_experiment_spec):
    _test_experiment_spec["model"].input_width = 50
    _test_experiment_spec["model"].input_height = 50
    _test_experiment_spec["dataset"].train_dataset_dir = os.path.join(_test_dir, "bounding_box_train")
    _test_experiment_spec["dataset"].test_dataset_dir = os.path.join(_test_dir, "bounding_box_test")
    _test_experiment_spec["dataset"].query_dataset_dir = os.path.join(_test_dir, "query")

    dataset = Market1501(_test_experiment_spec, prepare_for_training=True)
    transforms = build_transforms(_test_experiment_spec, True)
    data_set = ImageDataset(dataset.train, transforms)
    sampler = RandomIdentitySampler(dataset.train, 16, 4)
    dataloader = DataLoader(dataset=data_set,
                            batch_size=16,
                            num_workers=8,
                            sampler = sampler,
                            collate_fn=train_collate_fn, pin_memory = True)

    for _, data in list(enumerate(dataloader)):
        ids = data[1]
        for index in range(0, len(ids), 4):
            g = groupby(ids[index: index + 4])
            assert next(g, True) and not next(g, False), "Incorrect output from identity sampler."

@pytest.mark.cv_unit
def test_cosine_similarity():
    a = torch.rand((5, 10))
    b = torch.rand((20, 10))
    actual = cosine_similarity(a, b)
    assert actual.shape == (5, 20), "Incorrect output matrix dimensions."

@pytest.mark.cv_unit
def test_eucldidean_similarity():
    a = torch.rand((5, 10))
    b = torch.rand((20, 10))
    actual = euclidean_distance(a, b)
    assert actual.shape == (5, 20), "Incorrect output matrix dimensions."

@pytest.mark.cv_unit
def test_triplet_loss(_test_experiment_spec):
    triplet = TripletLoss(_test_experiment_spec['train']["optim"]['triplet_loss_margin'])  # triplet loss
    feature = torch.rand((16,2048))
    label = [10,10,10,10,20,20,20,20,30,30,30,30,40,40,40,40]
    label = torch.tensor(label)
    assert triplet(feature, label)[0].shape == (), "Incorect loss dimensions."
    assert triplet(feature, label)[1].shape == (16,), "Incorect loss dimensions for a batch."
    assert triplet(feature, label)[2].shape == (16,), "Incorect loss dimensions for a batch."

@pytest.mark.cv_unit
def test_center_loss():
    center_loss = CenterLoss(751, feat_dim=2048, use_gpu=False)
    feature = torch.rand((16,2048))
    label = [10,10,10,10,20,20,20,20,30,30,30,30,40,40,40,40]
    label = torch.tensor(label)
    assert center_loss(feature, label).shape == (), "Incorect loss dimensions."

@pytest.mark.cv_unit
def test_cross_entropy_loss():
    xent = CrossEntropyLabelSmooth(num_classes=751, use_gpu=False)
    feature = torch.rand((16,2048))
    label = [10,10,10,10,20,20,20,20,30,30,30,30,40,40,40,40]
    label = torch.tensor(label)
    assert xent(feature, label).shape == (), "Incorect loss dimensions."

@pytest.mark.cv_unit
def test_constrastive_loss():
    constrastive_loss = ContrastiveLoss()
    feature = torch.rand((16,2048))
    label = [10,10,10,10,20,20,20,20,30,30,30,30,40,40,40,40]
    label = torch.tensor(label)
    loss_fn = ContrastiveLoss(margin=0.5)
    inputs = torch.randn(32, 10)
    targets = torch.randint(0, 5, (32,))
    loss = loss_fn(inputs, targets)
    assert loss_fn.margin == 0.5, "Parameters are initialized incorrectly."
    assert isinstance(loss, torch.Tensor) == True, "Incorrect output type."
    assert constrastive_loss(feature, label).shape == (), "Incorect loss dimensions."

@pytest.mark.cv_unit
def test_circle_loss():
    circle_loss = CircleLoss(in_features=10, num_classes=5)
    input_tensor = torch.randn(32, 10)
    targets = torch.randint(0, 5, (32,))
    logits = circle_loss(input_tensor, targets)
    assert circle_loss.weight.shape == (5, 10), "Parameters are initialized incorrectly."
    assert isinstance(logits, torch.Tensor) == True, "Incorrect output type."
    assert logits.shape == (32, 5), "Incorrect loss dimensions."

@pytest.mark.cv_unit
def test_arcface_loss():
    arcface = Arcface(in_features=10, out_features=5)
    assert isinstance(arcface.weight, torch.nn.Parameter) == True, "Incorrect output type."
    assert arcface.weight.shape == (5, 10), "Incorrect loss dimensions."

@pytest.mark.cv_unit
def test_cosface_loss():
    cosface = Cosface(in_features=10, out_features=5)
    assert isinstance(cosface.weight, torch.nn.Parameter) == True, "Incorrect output type."
    assert cosface.weight.shape == (5, 10), "Incorrect loss dimensions."
    cosface.weight.cuda()
    input_tensor = torch.randn(32, 10).cuda()
    labels = torch.randint(0, 5, (32,)).cuda()
    output = cosface(input_tensor, labels)
    assert output.shape == (32, 5), "Incorrect output dimensions."

@pytest.mark.cv_unit
def test_amsoftmax_loss():
    amsoftmax = AMSoftmax(in_features=10, out_features=5)
    assert isinstance(amsoftmax.W, torch.nn.Parameter) == True, "Incorrect object type."
    assert amsoftmax.W.shape == (10, 5), "Incorrect weight shape."
    input_tensor = torch.randn(32, 10)
    labels = torch.randint(0, 5, (32,))
    print(input_tensor.is_cuda)
    output = amsoftmax(input_tensor, labels)
    assert output.shape == (32, 5), "Incorrect output dimensions."