# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the dataset."""

from typing import List
from dataclasses import dataclass
from omegaconf import MISSING

from nvidia_tao_pytorch.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    BOOL_FIELD,
    FLOAT_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD
)


@dataclass
class QuantCalibrationDataset:
    """Quantization calibration dataset config."""

    images_dir: str = STR_FIELD(
        value="",
        default_value="",
        description="Path to the directory containing calibration images.",
        display_name="calibration images directory"
    )


@dataclass
class Sparse4DTrainDatasetConfig:
    """Training dataset configuration for Sparse4D."""

    ann_file: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="Path to annotation file",
        display_name="Path to annotation file"
    )
    test_mode: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Test mode",
        display_name="Test mode"
    )
    use_valid_flag: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Use valid flag",
        display_name="Use valid flag"
    )
    with_seq_flag: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="With sequence flag",
        display_name="With sequence flag"
    )
    sequences_split_num: int = INT_FIELD(
        value=100,
        default_value=100,
        valid_min=1,
        valid_max="inf",
        description="Number of sequences",
        display_name="Number of sequences"
    )
    keep_consistent_seq_aug: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Keep consistent sequence augmentation",
        display_name="Keep consistent sequence augmentation"
    )
    same_scene_in_batch: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Same scene in batch",
        display_name="Same scene in batch"
    )


@dataclass
class Sparse4DValDatasetConfig:
    """Validation dataset configuration for Sparse4D."""

    ann_file: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="Path to annotation file",
        display_name="Path to annotation file"
    )
    test_mode: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Test mode",
        display_name="Test mode"
    )
    use_valid_flag: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Use valid flag",
        display_name="Use valid flag"
    )
    tracking: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Tracking",
        display_name="Tracking"
    )
    tracking_threshold: float = FLOAT_FIELD(
        value=0.2,
        default_value=0.2,
        valid_min=0,
        valid_max=1,
        description="Tracking threshold",
        display_name="Tracking threshold"
    )
    same_scene_in_batch: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Same scene in batch",
        display_name="Same scene in batch"
    )


@dataclass
class Sparse4DTestDatasetConfig:
    """Test dataset configuration for Sparse4D."""

    ann_file: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="Path to annotation file",
        display_name="Path to annotation file"
    )
    test_mode: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Test mode",
        display_name="Test mode"
    )
    use_valid_flag: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Use valid flag",
        display_name="Use valid flag"
    )
    tracking: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Tracking",
        display_name="Tracking"
    )
    tracking_threshold: float = FLOAT_FIELD(
        value=0.2,
        default_value=0.2,
        valid_min=0,
        valid_max=1,
        description="Tracking threshold",
        display_name="Tracking threshold"
    )
    same_scene_in_batch: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Same scene in batch",
        display_name="Same scene in batch"
    )


@dataclass
class Sparse4DAugmentationConfig:
    """Augmentation configuration for Sparse4D."""

    resize_lim: List[float] = LIST_FIELD(
        arrList=[0.7, 0.77],
        default_value=[0.7, 0.77],
        description="Resize limits",
        display_name="Resize limits"
    )
    final_dim: List[int] = LIST_FIELD(
        arrList=[512, 1408],
        default_value=[512, 1408],
        description="Final dimensions",
        display_name="Final dimensions"
    )
    bot_pct_lim: List[float] = LIST_FIELD(
        arrList=[0.0, 0.0],
        default_value=[0.0, 0.0],
        description="Bottom percentage limits",
        display_name="Bottom percentage limits"
    )
    rot_lim: List[float] = LIST_FIELD(
        arrList=[-5.4, 5.4],
        default_value=[-5.4, 5.4],
        description="Rotation limits in degrees",
        display_name="Rotation limits in degrees"
    )
    image_size: List[int] = LIST_FIELD(
        arrList=[1080, 1920],
        default_value=[1080, 1920],
        description="Original image size",
        display_name="Original image size"
    )
    rand_flip: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Random flip",
        display_name="Random flip"
    )
    rot3d_range: List[float] = LIST_FIELD(
        arrList=[-0.3925, 0.3925],
        default_value=[-0.3925, 0.3925],
        description="3D rotation range in radians",
        display_name="3D rotation range in radians"
    )


@dataclass
class Sparse4DNormalizeConfig:
    """Normalization configuration for Sparse4D."""

    mean: List[float] = LIST_FIELD(
        arrList=[123.675, 116.28, 103.53],
        default_value=[123.675, 116.28, 103.53],
        description="Mean values for normalization",
        display_name="Mean values for normalization"
    )
    std: List[float] = LIST_FIELD(
        arrList=[58.395, 57.12, 57.375],
        default_value=[58.395, 57.12, 57.375],
        description="Standard deviation values for normalization",
        display_name="Standard deviation values for normalization"
    )
    to_rgb: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Convert to RGB",
        display_name="Convert to RGB"
    )


@dataclass
class Sparse4DSequencesConfig:
    """Sequences configuration for Sparse4D."""

    split_num: int = INT_FIELD(
        value=100,
        default_value=100,
        valid_min=1,
        valid_max="inf",
        description="Number of sequence splits",
        display_name="Number of sequence splits"
    )
    keep_consistent_aug: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Keep consistent augmentation",
        display_name="Keep consistent augmentation"
    )
    same_scene_in_batch: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Keep same scene in batch",
        display_name="Keep same scene in batch"
    )


@dataclass
class Sparse4DTrackingConfig:
    """Tracking configuration for Sparse4D."""

    enabled: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Enable tracking",
        display_name="Enable tracking"
    )
    threshold: float = FLOAT_FIELD(
        value=0.2,
        default_value=0.2,
        valid_min=0,
        valid_max=1,
        description="Tracking threshold",
        display_name="Tracking threshold"
    )


@dataclass
class Omniverse3DDetTrackDatasetConfig:
    """Dataset configuration for Sparse4D."""

    type: str = STR_FIELD(
        value="omniverse_3d_det_track",
        default_value="omniverse_3d_det_track",
        description="Dataset type",
        display_name="Dataset type"
    )
    batch_size: int = INT_FIELD(
        value=2,
        default_value=2,
        valid_min=1,
        valid_max="inf",
        description="Batch size",
        display_name="Batch size"
    )
    use_h5_file_for_rgb: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Use H5 file",
        display_name="Use H5 file"
    )
    use_h5_file_for_depth: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Use H5 file",
        display_name="Use H5 file"
    )
    num_frames: int = INT_FIELD(
        value=200,
        default_value=200,
        valid_min=1,
        valid_max="inf",
        description="Number of frames",
        display_name="Number of frames"
    )
    num_bev_groups: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        description="Number of BEV groups",
        display_name="Number of BEV groups"
    )
    data_root: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="Path to data root",
        display_name="Path to data root"
    )
    classes: List[str] = LIST_FIELD(
        arrList=[
            "person", "gr1_t2", "agility_digit", "nova_carter",
        ],
        default_value=[
            "person", "gr1_t2", "agility_digit", "nova_carter",
        ],
        description="Classes to detect",
        display_name="Classes to detect"
    )
    num_workers: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=0,
        valid_max="inf",
        description="Number of workers",
        display_name="Number of workers"
    )
    num_ids: int = INT_FIELD(
        value=70,
        default_value=70,
        valid_min=1,
        valid_max="inf",
        description="Number of IDs",
        display_name="Number of IDs"
    )
    augmentation: Sparse4DAugmentationConfig = DATACLASS_FIELD(
        Sparse4DAugmentationConfig(),
        description="Augmentation config",
        display_name="Augmentation config"
    )
    normalize: Sparse4DNormalizeConfig = DATACLASS_FIELD(
        Sparse4DNormalizeConfig(),
        description="Normalize config",
        display_name="Normalize config"
    )
    sequences: Sparse4DSequencesConfig = DATACLASS_FIELD(
        Sparse4DSequencesConfig(),
        description="Sequences config",
        display_name="Sequences config"
    )
    train_dataset: Sparse4DTrainDatasetConfig = DATACLASS_FIELD(
        Sparse4DTrainDatasetConfig(),
        description="Train dataset config",
        display_name="Train dataset config"
    )
    val_dataset: Sparse4DValDatasetConfig = DATACLASS_FIELD(
        Sparse4DValDatasetConfig(),
        description="Val dataset config",
        display_name="Val dataset config"
    )
    test_dataset: Sparse4DTestDatasetConfig = DATACLASS_FIELD(
        Sparse4DTestDatasetConfig(),
        description="Test dataset config",
        display_name="Test dataset config"
    )
    quant_calibration_dataset: QuantCalibrationDataset = DATACLASS_FIELD(
        QuantCalibrationDataset(),
        description="Configurable parameters for quantization calibration dataset.",
    )
