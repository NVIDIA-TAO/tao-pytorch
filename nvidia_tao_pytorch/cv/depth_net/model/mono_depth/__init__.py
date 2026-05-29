# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Monocular DepthNet model module.

This module provides factory functions and model mappings for monocular depth estimation
models. It contains mappings between model type keys and their corresponding model classes
and loss functions for both metric and relative depth estimation tasks.
"""

from .depth_anything_v2.dpt import MetricDepthAnythingV2, RelativeDepthAnythingV2
from .loss import SiLogLoss, ScaleAndShiftInvariantLoss

_models_loss = {'MetricDepthAnything': (MetricDepthAnythingV2, SiLogLoss),
                'RelativeDepthAnything': (RelativeDepthAnythingV2, ScaleAndShiftInvariantLoss)
                }


def get_model_loss_class(key):
    """
    This function serves as a factory method that returns the appropriate model class
    and loss function based on the specified model type. It maps model type strings
    to their corresponding (ModelClass, LossClass) tuples.

    Args:
        key (str): Model type identifier. Supported keys:
            - 'MetricDepthAnything': Returns MetricDepthAnythingV2 model with SiLogLoss
            - 'RelativeDepthAnything': Returns RelativeDepthAnythingV2 model with ScaleAndShiftInvariantLoss

    Returns:
        tuple: A tuple containing (ModelClass, LossClass) where:
            - ModelClass: The model class to instantiate
            - LossClass: The loss function class to use for training
    """
    return _models_loss[key]
