# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data source config class for DriveNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.utils.data as data
import numpy as np

from nvidia_tao_pytorch.cv.deformable_detr.utils.data_source_config import build_data_source_lists_per_gpu, build_data_source_lists
from nvidia_tao_pytorch.cv.deformable_detr.dataloader.od_dataset import ODDataset, ConcateODDataset


class UniformSampler(object):
    """Uniform Sampler Class from multi-source data."""

    def __init__(self,
                 data_sources,
                 transforms=None,
                 contiguous_labels=False):
        """Initialize Uniform Sampler Class.

        Only used in distributed training and sharded data, unifrom distribution sampling.

        Args:
            data_sources (dict): augmentation configuration.
            transforms (dict): transforms.
            contiguous_labels (bool): if True, remap category IDs to contiguous 0-based indices.
        """
        self.data_sources = data_sources
        self.transforms = transforms
        self.contiguous_labels = contiguous_labels

    def build_data_source(self, global_rank, num_gpus):
        """ Build the data source list from multi-source data.

        Args:
            global_rank (int): gpu global rank to load the subset of the data.
            num_gpus (int): total number of gpus to be used.

        Returns:
            train_dataset (Dataset): training datsaet.
            dataset_length (int): length of each dataset (to be used in uniform sampling).
            total_images_per_gpu (int): total number of images per gpus (to be used in uniform sampling).

        """
        # distribute json files to each GPU
        data_source_list = build_data_source_lists_per_gpu(self.data_sources, global_rank, num_gpus)

        # concate the json files per gpu, load only sepecific jsons to each gpu
        dataset_per_gpu = []
        dataset_length = []
        total_images_per_gpu = 0
        for data_source in data_source_list:
            image_dir = data_source.image_dir
            for _json_file in data_source.dataset_files:
                ds = ODDataset(dataset_dir=image_dir, json_file=_json_file, transforms=self.transforms, contiguous_labels=self.contiguous_labels)
                dataset_per_gpu.append(ds)
                dataset_length.append(len(ds))
                total_images_per_gpu = total_images_per_gpu + len(ds)

        if len(dataset_per_gpu) > 1:
            train_dataset = ConcateODDataset(dataset_per_gpu)
        else:
            train_dataset = dataset_per_gpu[0]

        return train_dataset, dataset_length, total_images_per_gpu

    def get_sampler(self, global_rank, num_gpus):
        """ Get uniform sampler from the data source list.

        Args:
            global_rank (int): gpu global rank to load the subset of the data.
            num_gpus (int): total number of gpus to be used.

        Returns:
            train_dataset (Dataset): training dataset.
            train_sampler (Sampler): training sampler.

        """
        train_dataset, dataset_length, total_images_per_gpu = self.build_data_source(global_rank, num_gpus)
        weights = np.concatenate([[(len(train_dataset) - d_len) / len(train_dataset)] * d_len for d_len in dataset_length])
        num_samples = int(total_images_per_gpu)

        train_sampler = data.WeightedRandomSampler(weights, num_samples, replacement=True)
        return train_dataset, train_sampler


class NonUniformSampler(object):
    """Non-Uniform Sampler Class from multi-source data."""

    def __init__(self,
                 data_sources,
                 transforms=None,
                 contiguous_labels=False):
        """Initialize NonUniform Sampler Class.

        Only used in distributed training and sharded data, and does not apply unifrom distribution sampling

        Args:
            data_sources (dict): augmentation configuration
            transforms (dict): transforms
            contiguous_labels (bool): if True, remap category IDs to contiguous 0-based indices.
        """
        self.data_sources = data_sources
        self.transforms = transforms
        self.contiguous_labels = contiguous_labels

    def build_data_source(self, global_rank, num_gpus):
        """ Build the data source list from multi-source data.

        Args:
            global_rank (int): gpu global rank to load the subset of the data.
            num_gpus (int): total number of gpus to be used.

        Returns:
            train_dataset (Dataset): training datsaet.
            dataset_length (int): length of each dataset (to be used in uniform sampling).
            total_images_per_gpu (int): total number of images per gpus (to be used in uniform sampling).

        """
        # distribute json files to each GPU
        data_source_list = build_data_source_lists_per_gpu(self.data_sources, global_rank, num_gpus)

        # concate the json files per gpu, load only sepecific jsons to each gpu
        dataset_per_gpu = []
        dataset_length = []
        total_images_per_gpu = 0
        for data_source in data_source_list:
            image_dir = data_source.image_dir
            for _json_file in data_source.dataset_files:
                ds = ODDataset(dataset_dir=image_dir, json_file=_json_file, transforms=self.transforms, contiguous_labels=self.contiguous_labels)
                dataset_per_gpu.append(ds)
                dataset_length.append(len(ds))
                total_images_per_gpu = total_images_per_gpu + len(ds)

        if len(dataset_per_gpu) > 1:
            train_dataset = ConcateODDataset(dataset_per_gpu)
        else:
            train_dataset = dataset_per_gpu[0]

        return train_dataset, dataset_length, total_images_per_gpu

    def get_sampler(self, global_rank, num_gpus):
        """ Get Default sampler from the data source list.

        Args:
            global_rank (int): gpu global rank to load the subset of the data.
            num_gpus (int): total number of gpus to be used.

        Returns:
            train_dataset (Dataset): training dataset.
            train_sampler (Sampler): training sampler.
        """
        train_dataset, _, _ = self.build_data_source(global_rank, num_gpus)
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

        return train_dataset, train_sampler


class DefaultSampler(object):
    """Default Sampler Class from multi or single source data."""

    def __init__(self,
                 data_sources,
                 is_distributed=False,
                 transforms=None,
                 contiguous_labels=False):
        """Default Sampler Constructor.

        Args:
            data_sources (dict): augmentation configuration.
            transforms (dict): transforms.
            is_distributed(bool): flag indicting whether torch is using distributed learning or not.
            contiguous_labels (bool): if True, remap category IDs to contiguous 0-based indices.
        """
        self.data_sources = data_sources
        self.transforms = transforms
        self.is_distributed = is_distributed
        self.contiguous_labels = contiguous_labels

    def build_data_source(self):
        """Build the data source list from multi-source data.

        Returns:
            train_dataset: training dataset.
        """
        # grab all the json files and concate them into one single dataset
        data_source_list = build_data_source_lists(self.data_sources)
        dataset_list = []
        for data_source in data_source_list:
            image_dir = data_source.image_dir
            for _json_file in data_source.dataset_files:
                dataset_list.append(ODDataset(dataset_dir=image_dir, json_file=_json_file, transforms=self.transforms, contiguous_labels=self.contiguous_labels))

        if len(dataset_list) > 1:
            train_dataset = ConcateODDataset(dataset_list)
        else:
            train_dataset = dataset_list[0]

        return train_dataset

    def get_sampler(self):
        """Get Default sampler from the data source list.

        Returns:
            train_dataset (Dataset): training dataset
            train_sampler (Sampler): training sampler
        """
        train_dataset = self.build_data_source()

        if self.is_distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        else:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)

        return train_dataset, train_sampler
