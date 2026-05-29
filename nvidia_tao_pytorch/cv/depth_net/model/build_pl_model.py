# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build PyTorch lightning model module."""

from nvidia_tao_pytorch.cv.depth_net.model.mono_depth.pl_mono_model import MonoDepthNetPlModel
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.pl_stereo_model import StereoDepthNetPlModel

_pl_model_modules = {
    'MetricDepthAnything': MonoDepthNetPlModel,
    'RelativeDepthAnything': MonoDepthNetPlModel,
    'FoundationStereo': StereoDepthNetPlModel,
    # Why: FFS shares the StereoDepthNet PL module (same training / eval / export
    # loop); only the inner model class is FastFoundationStereo. Mapped here
    # because get_pl_module() and build_pl_model() both index this dict.
    'FastFoundationStereo': StereoDepthNetPlModel,
}


def get_pl_module(experiment_config):
    """
    This function retrieves the appropriate PyTorch Lightning model class based on the
    model type specified in the experiment configuration. It serves as a factory function
    to map model type strings to their corresponding model classes.

    Args:
        experiment_config (object): Experiment configuration object containing model settings.
            Must have the following attribute:
            - model.model_type (str): Type of model to retrieve. Supported types:
                - "MetricDepthAnything": For metric depth estimation
                - "RelativeDepthAnything": For relative depth estimation

    Returns:
        class: PyTorch Lightning model class corresponding to the specified model type.
    """
    return _pl_model_modules[experiment_config.model.model_type]


def build_pl_model(experiment_config, export=False):
    """
    This function creates a fully configured PyTorch Lightning model instance based on
    the provided experiment configuration. It automatically selects the appropriate
    model class and initializes it with the configuration parameters.

    Args:
        experiment_config (object): Experiment configuration object containing all model
            and training parameters. Must have the following attributes:
            - model.model_type (str): Type of model to build. Supported types:
                - "MetricDepthAnything": For metric depth estimation
                - "RelativeDepthAnything": For relative depth estimation
            - Additional model-specific configuration parameters as required by the
              selected model class constructor
        export (bool, optional): Whether the model is being used for export.
            Defaults to False.

    Returns:
        MonoDepthNetPlModel: Instantiated PyTorch Lightning model ready for training.
    """
    return _pl_model_modules[experiment_config.model.model_type](experiment_config, export=export)
