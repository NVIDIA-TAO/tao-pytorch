# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the dataset."""

from typing import Optional, List
from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    STR_FIELD,
    LIST_FIELD,
)


@dataclass
class DataConvertExpConfig:
    """Configuration parameters for Data Converter"""

    source: str = STR_FIELD(
        value="",
        default_value="",
        display_name="Sorce dataset",
        description="Sorce dataset which follows torchvision.datasets.ImageFolder format",
    )
    results_dir: str = STR_FIELD(
        value="",
        default_value="",
        display_name="Result directory",
        description="Result directory",
    )
    dest_file_name: str = STR_FIELD(
        value="torchvision_datasets_ImageFolder.zip",
        default_value="torchvision_datasets_ImageFolder.zip",
        display_name="Destination zipped file name",
        description="Destination zipped file name generated from source dataset",
    )
    resolution: List[int] = LIST_FIELD(
        arrList=[128, 128],
        display_name="Resized resolution",
        description="The moving average parameter for adaptive learning rate."
    )
    transform: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="Transformation before resizing",
        description="Transformation such as 'center-crop' before resizing can avoid distortion",
        valid_options="center-crop"
    )
