# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the dataset."""

from typing import Any, List
from dataclasses import dataclass, field

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    INT_FIELD,
    STR_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
    DICT_FIELD,
    FLOAT_FIELD
)
from nvidia_tao_pytorch.config.common.quantization import QuantCalibrationDataset


@dataclass
class Dataset:
    """Dataset config."""

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
    images: Any = field(default=None, metadata={
        "display_name": "image root",
        "value_type": "any",
        "description": "Path(s) to image root. String or list of strings. "
                       "Optional when the annotation JSON contains img_path.",
        "default_value": None,
    })
    annotations: Any = field(default=None, metadata={
        "display_name": "annotation root",
        "value_type": "any",
        "description": "Path(s) to annotation JSON. String or list of strings.",
        "default_value": None,
    })
    panoptic: Any = field(default=None, metadata={
        "display_name": "panoptic root",
        "value_type": "any",
        "description": "Path(s) to panoptic mask root. String or list of strings.",
        "default_value": None,
    })
    names: Any = field(default=None, metadata={
        "display_name": "dataset names",
        "value_type": "any",
        "description": "Human-readable name(s) for each dataset. String or list of strings "
                       "in the same order as annotations. Used as metric suffixes when "
                       "evaluating multiple datasets separately.",
        "default_value": None,
    })


@dataclass
class AugmentationConfig:
    """Augmentation config."""

    train_min_size: List[int] = LIST_FIELD(
        arrList=[800],
        description="A list of sizes to perform random resize.",
        display_name="Train min size"
    )
    train_max_size: int = INT_FIELD(
        value=1333,
        valid_min=32,
        valid_max="inf",
        description="The maximum random crop size for training data",
        automl_enabled="TRUE",
        display_name="Train max size"
    )
    train_crop_size: List[int] = LIST_FIELD(
        arrList=[1024, 1024],
        description="The random crop size for training data in [H, W]",
        display_name="Train crop size"
    )
    test_min_size: int = INT_FIELD(
        value=800,
        valid_min=32,
        valid_max="inf",
        description="The minimum resize size for test data",
        automl_enabled="TRUE",
        display_name="Test min size"
    )
    test_max_size: int = INT_FIELD(
        value=1333,
        valid_min=32,
        valid_max="inf",
        description="The maximum resize size for test",
        automl_enabled="TRUE",
        display_name="Test max size"
    )


@dataclass
class OneFormerDatasetConfig:
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
    workers: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="The number of parallel workers processing data",
        display_name="workers"
    )
    pin_memory: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="pin_memory",
        description="Flag to enable the dataloader to allocate pagelocked memory"
    )
    pixel_mean: List[float] = LIST_FIELD(
        arrList=[123.675, 116.28, 103.53],
        description="The input mean for RGB frames",
        display_name="input mean per pixel"
    )
    pixel_std: List[float] = LIST_FIELD(
        arrList=[58.395, 57.12, 57.375],
        description="The input standard deviation per pixel for RGB frames",
        display_name="input std per pixel"
    )
    augmentation: AugmentationConfig = DATACLASS_FIELD(
        AugmentationConfig(),
        description="Configuration parameters for data augmentation",
    )
    contiguous_id: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="contiguous id",
        description="Flag to enable contiguous ids for labels."
    )
    label_map: str = STR_FIELD(
        value="",
        display_name="label map",
        description="A path to label map file"
    )
    task_prob_train: dict = DICT_FIELD(
        hashMap={
            "semantic": 0.33,
            "instance": 0.66,
            "panoptic": 0.01
        },
        description="Task probabilities",
        display_name="task probabilities"
    )
    task_prob_val: dict = DICT_FIELD(
        hashMap={
            "semantic": 0.33,
            "instance": 0.66,
            "panoptic": 0.01
        },
        description="Task probabilities",
        display_name="task probabilities"
    )
    task_seq_len: int = INT_FIELD(
        value=77,
        description="Task sequence length",
        display_name="task sequence length"
    )
    max_seq_len: int = INT_FIELD(
        value=77,
        description="Maximum sequence length",
        display_name="maximum sequence length"
    )
    image_size: Any = field(default=1024, metadata={
        "display_name": "image size",
        "value_type": "any",
        "description": "Image size. Either a single integer (square) or a list of two integers [W, H].",
        "default_value": 1024,
    })
    min_scale: float = FLOAT_FIELD(
        value=0.1,
        description="Minimum scale",
        display_name="minimum scale"
    )
    max_scale: float = FLOAT_FIELD(
        value=2.0,
        description="Maximum scale",
        display_name="maximum scale"
    )
    cutmix_prob: float = FLOAT_FIELD(
        value=0.0,
        description="Cutmix probability",
        display_name="cutmix probability"
    )
    quant_calibration_dataset: QuantCalibrationDataset = DATACLASS_FIELD(
        QuantCalibrationDataset(),
        description="Configurable parameters for the quantization calibration dataset.",
    )
