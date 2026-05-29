#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataloader Init."""
import copy

import PIL
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def get_dataset(data_path, module_name, transform, dataset_args):
    """Get dataset.

    Args:
        data_path: dataset file list.
        module_name: custom dataset name, Supports data_loaders.ImageDataset

    Returns:
        ConcatDataset object
    """
    from . import dataset
    s_dataset = getattr(dataset, module_name)(transform=transform, data_path=data_path,
                                              **dataset_args)
    return s_dataset


def get_transforms(transforms_config):
    """Get transforms."""
    tr_list = []
    for item in transforms_config:
        if 'args' not in item:
            args = {}
        else:
            args = item['args']
        cls = getattr(transforms, item['type'])(**args)
        tr_list.append(cls)
    tr_list = transforms.Compose(tr_list)
    return tr_list


class ICDARCollateFN:
    """ICDAR Collate."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        pass

    def __call__(self, batch):
        """Generate data dict."""
        data_dict = {}
        tensor_keys = []
        for sample in batch:
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, (np.ndarray, torch.Tensor, PIL.Image.Image)):
                    if k not in tensor_keys:
                        tensor_keys.append(k)
                data_dict[k].append(v)
        for k in tensor_keys:
            data_dict[k] = torch.stack(data_dict[k], 0)
        return data_dict


def get_dataloader(module_config, distributed=False):
    """Generate dataloader based on the config file."""
    if module_config is None:
        return None
    config = copy.deepcopy(module_config)
    dataset_args = config['args']
    if 'transforms' in dataset_args:
        img_transfroms = get_transforms(dataset_args.pop('transforms'))
    else:
        img_transfroms = None
    dataset_name = config['data_name']
    data_path = config['data_path']
    if data_path is None:
        return None

    data_path = [x for x in data_path if x is not None]
    if len(data_path) == 0:
        return None
    if 'collate_fn' not in config['loader'] or config['loader']['collate_fn'] is None or len(config['loader']['collate_fn']) == 0:
        config['loader']['collate_fn'] = None
    else:
        config['loader']['collate_fn'] = globals()[config['loader']['collate_fn']]()

    _dataset = get_dataset(data_path=data_path, module_name=dataset_name, transform=img_transfroms, dataset_args=dataset_args)
    sampler = None
    if distributed:
        config['loader']['shuffle'] = False
        config['loader']['pin_memory'] = True
    loader = DataLoader(dataset=_dataset, sampler=sampler, **config['loader'])

    return loader
