# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the dataset."""

from typing import Optional, List
from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    INT_FIELD,
    STR_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD
)
from nvidia_tao_pytorch.config.common.quantization import QuantCalibrationDataset


@dataclass
class Dataset:
    """Dataset config."""

    type: str = STR_FIELD(
        value="ade",
        default_value="ade",
        display_name="",
        description="Dataset type",
        valid_options=",".join(["coco", "ade", "coco_panoptic"])
    )
    name: str = STR_FIELD(
        value="",
        default_value="",
        display_name="Dataset name",
        description="Dataset name",
    )
    panoptic_json: str = STR_FIELD(
        value="",
        default_value="",
        display_name="COCO Panoptic JSON",
        description="JSON file in COCO panoptic format",
    )
    instance_json: str = STR_FIELD(
        value="",
        default_value="",
        display_name="COCO Instance JSON",
        description="JSON file in COCO format",
    )
    img_dir: str = STR_FIELD(
        value="",
        default_value="",
        display_name="Raw image directory",
        description="Image directory (can be relative path to root_dir)",
    )
    panoptic_dir: str = STR_FIELD(
        value="",
        default_value="",
        display_name="Panoptic image directory",
        description="Directory of panoptic segmentation annotation images",
    )
    root_dir: str = STR_FIELD(
        value="",
        default_value="",
        display_name="Root image directory",
        description="Root image directory",
    )
    annot_file: str = STR_FIELD(
        value="",
        default_value="",
        display_name="Annotatioin file for semantic data",
        description="JSON file in JSONL format for image/mask pair",
    )
    batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        description="Batch size",
        math_cond=">0",
        valid_min=1,
        valid_max="inf",
        display_name="batch size"
    )
    num_workers: int = INT_FIELD(
        value=1,
        default_value=1,
        description="Number of workers",
        valid_min=0,
        valid_max="inf",
        display_name="Number of workers"
    )
    target_size: Optional[List[int]] = LIST_FIELD(
        arrList=[],
        default_value=[],
        description="""Target size for resizing.""",
        display_name="Target size",
    )


@dataclass
class AugmentationConfig:
    """Augmentation config."""

    train_min_size: List[int] = LIST_FIELD(
        arrList=[640],
        description="A list of sizes to perform random resize.",
        display_name="Train min size"
    )
    train_max_size: int = INT_FIELD(
        value=2560,
        valid_min=32,
        valid_max="inf",
        description="The maximum random crop size for training data",
        automl_enabled="TRUE",
        display_name="Train max size"
    )
    train_crop_size: List[int] = LIST_FIELD(
        arrList=[640, 640],
        description="The random crop size for training data in [H, W]",
        display_name="Train crop size"
    )
    test_min_size: int = INT_FIELD(
        value=640,
        valid_min=32,
        valid_max="inf",
        description="The minimum resize size for test data",
        automl_enabled="TRUE",
        display_name="Test min size"
    )
    test_max_size: int = INT_FIELD(
        value=640,
        valid_min=32,
        valid_max="inf",
        description="The maximum resize size for test",
        automl_enabled="TRUE",
        display_name="Test max size"
    )


@dataclass
class Mask2FormerDatasetConfig:
    """Data config."""

    train: Dataset = DATACLASS_FIELD(
        Dataset(),
        description="Configurable parameters to construct the train dataset.",
    )
    val: Dataset = DATACLASS_FIELD(
        Dataset(),
        description="Configurable parameters to construct the validation dataset.",
    )
    test: Dataset = DATACLASS_FIELD(
        Dataset(),
        description="Configurable parameters to construct the test dataset.",
    )
    pin_memory: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="pin_memory",
        description="""Flag to enable the dataloader to allocate pagelocked memory for faster
                    of data between the CPU and GPU."""
    )
    pixel_mean: List[float] = LIST_FIELD(
        arrList=[0.485, 0.456, 0.406],
        description="The input mean for RGB frames",
        display_name="input mean per pixel"
    )
    pixel_std: List[float] = LIST_FIELD(
        arrList=[0.229, 0.224, 0.225],
        description="The input standard deviation per pixel for RGB frames",
        display_name="input std per pixel"
    )
    augmentation: AugmentationConfig = DATACLASS_FIELD(
        AugmentationConfig(),
        description="Configuration parameters for data augmentation",
    )
    contiguous_id: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="contiguous id",
        description="""Flag to enable contiguous ids for labels."""
    )
    label_map: str = STR_FIELD(
        value="",
        default_value="",
        display_name="label map",
        description="A path to label map file"
    )
    quant_calibration_dataset: QuantCalibrationDataset = DATACLASS_FIELD(
        QuantCalibrationDataset(),
        description="Configurable parameters for the quantization calibration dataset.",
    )
