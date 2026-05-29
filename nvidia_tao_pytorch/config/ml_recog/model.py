# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the model."""

from dataclasses import dataclass
from typing import Optional

from nvidia_tao_pytorch.config.utils.types import (
    INT_FIELD,
    STR_FIELD,
)


@dataclass
class MLModelConfig:
    """Metric Learning Recognition model configuration for training, testing & validation."""

    backbone: str = STR_FIELD(
        value="resnet_50",
        default_value="resnet_50",
        display_name="backbone",
        description="The backbone name of the model",
        valid_options=",".join(["resnet_50", "resnet_101", "fan_small", "fan_base", "fan_large",
                               "fan_tiny", "nvdinov2_vit_large_legacy"])
    )
    pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="pretrained model path",
        description="[Optional] Path to the pretrained model. The weights are only loaded to the whole model.",
    )
    pretrained_trunk_path: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="pretrained_trunk_path",
        description="[Optional] Path to the pretrained trunk. The weights are only loaded to the trunk part.",
    )
    pretrained_embedder_path: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="pretrained_embedder_path",
        description="[Optional] Path to the pretrained embedder. The weights are only loaded to the embedder part.",
    )
    input_width: int = INT_FIELD(
        value=224,
        default_value=224,
        description="The input width of the images.",
        display_name="input_width",
        parent_param="TRUE",
    )
    input_height: int = INT_FIELD(
        value=224,
        default_value=224,
        description="The input height of the images.",
        display_name="input_height",
        parent_param="TRUE",
    )
    input_channels: int = INT_FIELD(
        value=3,
        default_value=3,
        description="The number of input channels.",
        display_name="input_channels",
    )
    feat_dim: int = INT_FIELD(
        value=256,
        default_value=256,
        description="The output size of the feature embeddings.",
        display_name="feature dimension",
    )
