# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DepthNet Monocular datasets module."""

from torch.utils.data import ConcatDataset
from omegaconf import DictConfig

from .threedvlm import ThreeDVLM
from .fsd import FSD
from .nvclip import NvCLIP
from .issac_stereo import IssacStereo
from .crestereo import Crestereo
from .middlebury import Middlebury
from .nyudv2 import NYUDV2
from .nyudv2_relative import NYUDV2Relative
from .base_relative_mono import BaseRelativeMonoDataset
from .base_metric_mono import BaseMetricMonoDataset

DATASETS = {
    'threedvlm': ThreeDVLM,
    'fsd': FSD,
    'nvclip': NvCLIP,
    'issacstereo': IssacStereo,
    'crestereo': Crestereo,
    'middlebury': Middlebury,
    'nyudv2': NYUDV2,
    'nyudv2relative': NYUDV2Relative,
    'relativemonodataset': BaseRelativeMonoDataset,
    'metricmonodataset': BaseMetricMonoDataset,
}


def get_dataset_class(key):
    """Get Model class given the model key from spec file.

    Args:
        key (str): key of the dataset.

    Returns:
        model_cls (class): model class.
    """
    dataset_cls = DATASETS.get(key.lower(), None)
    if not dataset_cls:
        raise NotImplementedError(f"Dataset type {key} not supported. Supported list {list(DATASETS.keys())}")

    return dataset_cls


def build_mono_dataset(data_sources, transform, min_depth=None, max_depth=None, normalize_depth=False):
    """Load Monocular Relative Depth Dataset.

    Args:
        data_sources (List[DictConfig]): list of different data sources.
        transform (callable): augmentations to apply.
        min_depth (float): minimum depth value.
        max_depth (float): maximum depth value.
        normalize_depth (bool): whether to normalize the depth.
    """
    if isinstance(data_sources, DictConfig):
        data_sources = [data_sources]

    dataset_list = []
    for data_source in data_sources:
        data_file = data_source.data_file
        dataset_name = data_source.dataset_name
        model_cls = get_dataset_class(dataset_name)
        if min_depth is not None and max_depth is not None:
            dataset_list.append(model_cls(data_file, transform=transform, min_depth=min_depth,
                                          max_depth=max_depth, normalize_depth=normalize_depth))
        else:
            dataset_list.append(model_cls(data_file, transform=transform,
                                          normalize_depth=normalize_depth))

    if len(dataset_list) > 1:
        train_dataset = ConcatDataset(dataset_list)
    elif len(dataset_list) == 1:
        train_dataset = dataset_list[0]
    else:
        raise ValueError("No dataset found")

    return train_dataset
