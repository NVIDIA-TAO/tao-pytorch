#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modules for the dataloader."""
# flake8: noqa: F401, F403
from .iaa_augment import IaaAugment
from .augment import *
from .random_crop_data import EastRandomCropData
from .make_border_map import MakeBorderMap
from .make_shrink_map import MakeShrinkMap
