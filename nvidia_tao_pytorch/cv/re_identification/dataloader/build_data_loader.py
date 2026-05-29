# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build torch data loader."""
import os
import torch
from torch.utils.data import DataLoader
from nvidia_tao_pytorch.cv.re_identification.dataloader.datasets.market1501 import Market1501
from nvidia_tao_pytorch.cv.re_identification.dataloader.sampler import RandomIdentitySampler, RandomIdentitySamplerDDP
from nvidia_tao_pytorch.cv.re_identification.dataloader.datasets.bases import ImageDataset
from nvidia_tao_pytorch.cv.re_identification.dataloader.transforms import build_transforms


def list_dataset(top_dir):
    """
    Returns a dictionary of image paths.

    This function iterates over the given directory, considering every file as an image.
    It then stores the image paths in a dictionary with a value of 1, indicating that
    the image exists.

    Args:
        top_dir (str): Path to the top-level directory containing images.

    Returns:
        dict: A dictionary where keys are image file names and values are all 1s,
        indicating the existence of the corresponding images.
    """
    sample_dict = {}
    for img in os.listdir(top_dir):
        sample_dict[img] = 1
    return sample_dict


def train_collate_fn(batch):
    """
    Returns a processed batch of images for training.

    This function takes a batch of image data, unpacks the images and person IDs,
    stacks the images into a tensor, and returns both the image tensor and the person IDs.

    Args:
        batch (Tensor): A batch of image data.

    Returns:
        tuple: A tuple containing the tensor of stacked images and the tensor of person IDs.
    """
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    """
    Returns a processed batch of images for validation & testing.

    This function takes a batch of image data, unpacks the images, person IDs, camera IDs,
    and image paths, stacks the images into a tensor, and returns the image tensor, person IDs,
    camera IDs, and image paths.

    Args:
        batch (Tensor): A batch of image data.

    Returns:
        tuple: A tuple containing the tensor of stacked images, person IDs, camera IDs, and image paths.
    """
    imgs, pids, camids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, img_paths


def build_dataloader(cfg, is_train):
    """
    Builds a PyTorch DataLoader object for training or validation/testing.

    The DataLoader is created based on whether the process is for training or validation.
    For the training process, a RandomIdentitySampler is used to order the data and the
    function 'train_collate_fn' is used to process the data. For the validation process,
    the function 'val_collate_fn' is used to process the data.

    Args:
        cfg (DictConfig): Configuration file specifying the parameters for the DataLoader.
        is_train (bool): If True, the DataLoader is for training; otherwise, it's for validation/testing.

    Returns:
        DataLoader: The DataLoader object for training if 'is_train' is True.
        DataLoader: The DataLoader object for validation/testing if 'is_train' is False.
        int: The number of query samples.
        int: The number of classes in the dataset.
    """
    val_transforms = build_transforms(cfg, is_train=False)
    num_gpus = len(cfg["train"]["gpu_ids"])
    num_workers = cfg["dataset"]["num_workers"] * num_gpus
    dataset = Market1501(cfg, is_train)
    train_loader, val_loader = None, None
    if is_train:
        train_transforms = build_transforms(cfg, is_train=True)
        num_classes = dataset.num_train_pids
        train_dataset = ImageDataset(dataset.train, train_transforms)

        if num_gpus > 1:
            mini_batch_size = cfg["dataset"]["batch_size"] // num_gpus
            data_sampler = RandomIdentitySamplerDDP(dataset.train, cfg["dataset"]["batch_size"], cfg["dataset"]["num_instances"], num_gpus)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=cfg["dataset"]["batch_size"] * num_gpus,
                sampler=RandomIdentitySampler(dataset.train, cfg["dataset"]["batch_size"] * num_gpus,
                                              cfg["dataset"]["num_instances"] * num_gpus),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
        enumerate(train_loader)
    else:
        num_classes = dataset.num_gallery_pids
    val_dataset = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["dataset"]["val_batch_size"] * num_gpus, shuffle=False,
        num_workers=num_workers, collate_fn=val_collate_fn
    )

    enumerate(val_loader)
    return train_loader, val_loader, len(dataset.query), num_classes
