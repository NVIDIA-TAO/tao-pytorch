# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the dataset."""

from dataclasses import dataclass
from typing import Optional, List, Dict
from omegaconf import MISSING

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
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
class DNDatasetConvertConfig:
    """Dataset Convert config."""

    data_root: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Path to the directory where the datasets are present.",
        display_name="dataset root"
    )
    results_dir: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="Path to where the converted dataset is serialized.",
        display_name="results directory"
    )
    image_dir_pattern: list = LIST_FIELD(
        arrList=[],
        description="""List of patterns for any path that should be included"
                    in the image path list, relative to the dataset root""",
        display_name="Image Path Pattern List"
    )
    right_dir_pattern: list = LIST_FIELD(
        arrList=[],
        description="""List of patterns for any path that should be included"
                    in the right image path list, relative to the dataset root""",
        display_name="Right Image Path Pattern List"
    )
    depth_dir_pattern: list = LIST_FIELD(
        arrList=[],
        description="""List of patterns for any path that should be included"
                    in the depth path list, relative to the dataset root""",
        display_name="Depth Map Path Pattern List"
    )
    nocc_dir_pattern: list = LIST_FIELD(
        arrList=[],
        description="""List of patterns for any path that should be included"
                    in the non-occluded map path list, relative to the dataset root""",
        display_name="Non-Occluded Path Pattern List"
    )
    split_ratio: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        description="The ratio for validation split.",
        display_name="validation split ratio",
        valid_min=0.0
    )
    depth_extension: str = STR_FIELD(
        value="png",
        default_value="png",
        description="The file extension of the depth images in the directory.",
        display_name="depth image extension"
    )
    image_extension: str = STR_FIELD(
        value="jpg",
        default_value="jpg",
        description="The file extension of the images in the directory.",
        display_name="image extension"
    )
    nocc_mask_dir_name: str = STR_FIELD(
        value="",
        default_value="",
        description="The relative directory path to find "
                    "non-occluded masks from the root directory",
        display_name="Non occluded mask"
    )
    nocc_extension: str = STR_FIELD(
        value="png",
        default_value="png",
        description="The file extension of the non-occluded mask. "
                    "Non-occluded masks are generally used to exclude "
                    "image regions that are hidded from a camera",
        display_name="Nocc image extension"
    )


@dataclass
class DepthNetAugmentationConfig:
    """Augmentation config."""

    input_mean: List[float] = LIST_FIELD(
        arrList=[0.485, 0.456, 0.406],
        description="The input mean for RGB frames",
        display_name="input mean per pixel"
    )
    input_std: List[float] = LIST_FIELD(
        arrList=[0.229, 0.224, 0.225],
        description="The input standard deviation per pixel for RGB frames",
        display_name="input std per pixel"
    )
    crop_size: List[int] = LIST_FIELD(
        arrList=[518, 518],
        description="The crop size for input RGB images [height, width]",
        display_name="augmentation crop size"
    )
    min_scale: float = FLOAT_FIELD(
        value=-0.2,
        valid_min=0.2,
        valid_max=1,
        description="The minimum scale in data augmentation",
        display_name="min scale"
    )
    max_scale: float = FLOAT_FIELD(
        value=0.4,
        valid_min=-0.2,
        valid_max=1,
        description="The maximum scale in data augmentation",
        display_name="max scale"
    )
    do_flip: bool = BOOL_FIELD(
        value="False",
        default_value="False",
        description="""A flag specifying whether to perform flip in data augmentation""",
        display_name="do flip"
    )
    yjitter_prob: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        description="The probability for y jitter",
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="TRUE",
        display_name="The probability for y jitter"
    )
    gamma: List[int] = LIST_FIELD(
        arrList=[1, 1, 1, 1],
        description="Gamma range in data augmentation",
        display_name="gamma range"
    )
    color_aug_prob: float = FLOAT_FIELD(
        value=0.2,
        default_value=0.2,
        description="The probability for asymmetric color augmentation",
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="TRUE",
        display_name="The probability for asymmetric color augmentation"
    )
    color_aug_brightness: float = FLOAT_FIELD(
        value=0.4,
        default_value=0.4,
        description="The color jitter brightness",
        valid_min=0.0,
        valid_max=1.0,
        display_name="The color jitter brightness"
    )
    color_aug_contrast: float = FLOAT_FIELD(
        value=0.4,
        default_value=0.4,
        description="The color jitter contrast",
        valid_min=0.0,
        valid_max=1.0,
        display_name="The color jitter contrast"
    )
    color_aug_saturation: List[float] = LIST_FIELD(
        arrList=[0.0, 1.4],
        description="The color jitter saturation",
        display_name="The color jitter saturation"
    )
    color_aug_hue_range: List[float] = LIST_FIELD(
        arrList=[-5 / 180.0, 5 / 180.0],
        description="The hue range in data augmentation",
        display_name="hue range augmentaiton"
    )
    eraser_aug_prob: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="The probability for eraser augmentation",
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="TRUE",
        display_name="The probability for eraser augmentation"
    )
    spatial_aug_prob: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        description="The probability for spatial augmentation",
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="TRUE",
        display_name="The probability for spatial augmentation"
    )
    stretch_prob: float = FLOAT_FIELD(
        value=0.8,
        default_value=0.8,
        description="The probability for stretch augmentation",
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="TRUE",
        display_name="The probability for stretch augmentation"
    )
    max_stretch: float = FLOAT_FIELD(
        value=0.2,
        default_value=0.2,
        description="The maximum stretch augmentation",
        valid_min=0.0,
        valid_max=1.0,
        display_name="The maximum stretch augmentation"
    )
    h_flip_prob: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="The probability for horizontal flip augmentation",
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="TRUE",
        display_name="The probability for horizontal flip augmentation"
    )
    v_flip_prob: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="The probability for vertical flip augmentation",
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="TRUE",
        display_name="The probability for vertical flip augmentation"
    )
    hshift_prob: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="The probability for horizontal shift augmentation",
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="TRUE",
        display_name="The probability for horizontal flip augmentation"
    )
    crop_min_valid_disp_ratio: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        description="The probability for minimum crop valid disparity ratio",
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="TRUE",
        display_name="The probability for min crop valid disparity ratio"
    )


@dataclass
class BaseDepthNetDatasetConfig:
    """BaseDataset config."""

    data_sources: Optional[List[Dict[str, str]]] = LIST_FIELD(
        arrList=None,
        default_value=[{"dataset_name": "", "data_file": ""}],
        description="""The list of data sources for training:
                    * dataset_name : The type of the dataset
                    * data_file : The path of the data file""",
        display_name="train data sources",
    )
    batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        description="The batch size for training and validation",
        display_name="batch size"
    )
    workers: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="The number of parallel workers processing data",
        display_name="batch size"
    )
    pin_memory: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="pin_memory",
        description="""Flag to enable the dataloader to allocated pagelocked memory for faster
                    of data between the CPU and GPU."""
    )
    augmentation: DepthNetAugmentationConfig = DATACLASS_FIELD(
        DepthNetAugmentationConfig(),
        description="Configuration parameters for data augmentation",
        display_name="augmentation",
    )


@dataclass
class DepthNetDatasetConfig:
    """DepthNet Dataset config."""

    dataset_name: str = STR_FIELD(
        value="StereoDataset",
        default_value="StereoDataset",
        description="Dataset Name",
        display_name="dataset mame",
        valid_options=",".join([
            "MonoDataset", "StereoDataset"
        ])
    )
    normalize_depth: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Normalize depth",
        display_name="normalize depth"
    )
    max_depth: Optional[float] = FLOAT_FIELD(
        value=None,
        valid_min=1.0,
        valid_max="inf",
        description="The maximum depth in meters in MetricDepthAnythingV2",
        display_name="max depth in meters"
    )
    min_depth: Optional[float] = FLOAT_FIELD(
        value=None,
        valid_min=0.0,
        valid_max="inf",
        description="The minimum depth in meters in MetricDepthAnythingV2",
        display_name="min depth in meters"
    )
    max_disparity: int = INT_FIELD(
        value=416,
        valid_min=1,
        valid_max=416,
        description="The maximum allowed disparity for which we compute losses during training",
        display_name="maximum dispairty"
    )
    baseline: float = FLOAT_FIELD(
        value=193.001 / 1e3,
        default_value=193.001 / 1e3,
        description="The baseline for stereo datasets",
        valid_min=0.0,
        valid_max="inf",
        display_name="Stereo baseline"
    )
    focal_x: float = FLOAT_FIELD(
        value=1998.842,
        default_value=1998.842,
        description="The focal length along x-axis",
        valid_min=0.0,
        valid_max="inf",
        display_name="The focal length along x-axis"
    )

    train_dataset: BaseDepthNetDatasetConfig = DATACLASS_FIELD(
        BaseDepthNetDatasetConfig(),
        description="Configurable parameters to construct the train dataset for a DepthNet experiment.",
    )
    val_dataset: BaseDepthNetDatasetConfig = DATACLASS_FIELD(
        BaseDepthNetDatasetConfig(),
        description="Configurable parameters to construct the val dataset for a DepthNet experiment.",
    )
    test_dataset: BaseDepthNetDatasetConfig = DATACLASS_FIELD(
        BaseDepthNetDatasetConfig(),
        description="Configurable parameters to construct the test dataset for a DepthNet experiment.",
    )
    infer_dataset: BaseDepthNetDatasetConfig = DATACLASS_FIELD(
        BaseDepthNetDatasetConfig(),
        description="Configurable parameters to construct the infer dataset for a DepthNet experiment.",
    )
    quant_calibration_dataset: QuantCalibrationDataset = DATACLASS_FIELD(
        QuantCalibrationDataset(),
        description="Configurable parameters for quantization calibration dataset.",
    )
