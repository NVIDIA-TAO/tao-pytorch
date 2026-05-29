# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DepthNet dataloader module."""

from nvidia_tao_pytorch.cv.depth_net.dataloader.pl_mono_data_module import MonoDepthNetDataModule
from nvidia_tao_pytorch.cv.depth_net.dataloader.pl_stereo_data_module import StereoDepthNetDataModule

_pl_data_modules = {'MonoDataset': MonoDepthNetDataModule,
                    'StereoDataset': StereoDepthNetDataModule}


def build_pl_data_module(dataset_config):
    """Build lightning data_module given the dataset_config from spec file.

    Args:
        dataset_config (dict): dataset configuration.

    Returns:
        pl_data_module (class): lightning data module.
    """
    return _pl_data_modules[dataset_config.dataset_name](dataset_config)
