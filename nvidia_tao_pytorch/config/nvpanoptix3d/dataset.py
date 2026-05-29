# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the dataset."""

from typing import List
from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    STR_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD
)


@dataclass
class Dataset:
    """Dataset config."""

    base_dir: str = STR_FIELD(
        value="",
        default_value="",
        display_name="dataset root",
        description="Root directory of the dataset",
    )
    json_path: str = STR_FIELD(
        value="",
        default_value="",
        display_name="annotation file path",
        description="JSON file in JSON format for image/mask pair.",
    )
    batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        description="Batch size",
        math_cond=">0",
        valid_min=1,
        display_name="batch size"
    )
    num_workers: int = INT_FIELD(
        value=1,
        default_value=1,
        description="Number of workers in the dataloader.",
        valid_min=0,
        display_name="num workers"
    )


@dataclass
class AugmentationConfig:
    """Augmentation config."""

    train_min_size: List[int] = LIST_FIELD(
        arrList=[448],
        default_value=[448],
        description="A list of sizes to perform random resize.",
        display_name="train min size"
    )
    train_max_size: int = INT_FIELD(
        value=768,
        default_value=768,
        valid_min=32,
        valid_max=960,
        description="The maximum random crop size for training data.",
        display_name="train max size"
    )
    train_crop_size: List[int] = LIST_FIELD(
        arrList=[240, 240],
        default_value=[240, 240],
        description="The random crop size for training data in [H, W].",
        display_name="train crop size"
    )
    test_min_size: int = INT_FIELD(
        value=240,
        default_value=240,
        valid_min=32,
        valid_max=960,
        description="The minimum resize size for test data.",
        display_name="test min size"
    )
    test_max_size: int = INT_FIELD(
        value=960,
        default_value=960,
        valid_min=32,
        valid_max=960,
        description="The maximum resize size for test.",
        display_name="test max size"
    )
    color_aug_ssd: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Color augmentation.",
        display_name="color augmentation"
    )
    enable_crop: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable cropping for input image.",
        display_name="enable cropping"
    )
    crop_size: List[int] = LIST_FIELD(
        arrList=[240, 240],
        description="Size to crop input image.",
        display_name="input image size crop",
    )
    single_category_max_area: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        valid_min=0.0,
        valid_max=1.0,
        description="Maximum ratio of crop area that can be occupied by a single semantic category.",
        display_name="maximum ratio of crop area"
    )
    random_flip: str = STR_FIELD(
        value="",
        default_value="",
        description="Flip horizontal/vertical.",
        display_name="flip horizontal/vertical",
    )
    random_flip_prob: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.0,
        valid_max=1.0,
        description="Flip probability.",
        display_name="flip probability"
    )
    size_divisibility: float = FLOAT_FIELD(
        value=-1,
        default_value=-1,
        description="Size divisibility to pad.",
        display_name="size divisibility to pad"
    )
    gen_aug_weight: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max=1.0,
        description="Weight for generated augmentation, 0.0 will disable generated augmentation.",
        display_name="weight for generated augmentation"
    )


@dataclass
class NVPanoptix3DDatasetConfig:
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
        description="The number of parallel workers processing data",
        display_name="num workers"
    )
    pin_memory: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="pin memory",
        description="Flag to allocate pagelocked memory for faster of data between the CPU and GPU."
    )
    augmentation: AugmentationConfig = DATACLASS_FIELD(
        AugmentationConfig(),
        description="Configuration parameters for data augmentation",
    )
    contiguous_id: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="contiguous id",
        description="Flag to enable contiguous ids for labels."
    )
    label_map: str = STR_FIELD(
        value="",
        default_value="",
        display_name="label map path",
        description="A path to label map file."
    )
    name: str = STR_FIELD(
        value="front3d",
        default_value="front3d",
        display_name="dataset name",
        description="Dataset name.",
        valid_options=",".join(["front3d", "matterport", "synthetic_hospital", "synthetic_warehouse"])
    )
    downsample_factor: int = INT_FIELD(
        value=1,
        default_value=1,
        display_name="downsample factor",
        description="Downsample factor(1: Synthetic & Front3D, 2: Matterport3D).",
    )
    iso_value: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        description="ISO value to reconstruct mesh from TUDF volume.",
        display_name="ISO value"
    )
    ignore_label: int = INT_FIELD(
        value=255,
        default_value=255,
        description="Ignore label value.",
        display_name="ignore label value"
    )
    min_instance_pixels: int = INT_FIELD(
        value=200,
        default_value=200,
        description="Minimum number of pixels required for an instance to be considered valid.",
        display_name="minimum number of pixels"
    )
    img_format: str = STR_FIELD(
        value="RGB",
        default_value="RGB",
        description="Image format.",
        display_name="image format"
    )
    target_size: List[int] = LIST_FIELD(
        arrList=[320, 240],
        default_value=[320, 240],
        description="Input image size to resize.",
        display_name="input image size to resize",
    )
    reduced_target_size: List[int] = LIST_FIELD(
        arrList=[160, 120],
        default_value=[160, 120],
        description="Image size to process at 3D stage.",
        display_name="image size to process at 3D stage",
    )
    depth_size: List[int] = LIST_FIELD(
        arrList=[120, 160],
        default_value=[120, 160],
        description="Input depth size to resize.",
        display_name="input depth size to resize",
    )
    depth_bound: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable depth truncation in bounds.",
        display_name="enable depth truncation"
    )
    depth_min: float = FLOAT_FIELD(
        value=0.4,
        default_value=0.4,
        description="Min depth value.",
        display_name="min depth value"
    )
    depth_max: float = FLOAT_FIELD(
        value=6.0,
        default_value=6.0,
        description="Max depth value.",
        display_name="max depth value"
    )
    frustum_mask_path: str = STR_FIELD(
        value="meta/frustum_mask.npz",
        default_value="",
        display_name="relative frustum mask path",
        description="Relative frustum mask path."
    )
    occ_truncation_lvl: List[float] = LIST_FIELD(
        arrList=[8.0, 6.0],
        default_value=[8.0, 6.0],
        description="Value to create occuppancy volume from TUDF volume.",
        display_name="occ truncation level"
    )
    truncation_range: List[float] = LIST_FIELD(
        arrList=[0.0, 12.0],
        default_value=[0.0, 12.0],
        description="truncation range for TUDF volume.",
        display_name="TUDF truncation range"
    )
    enable_3d: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable 3d for training.",
        display_name="enable 3d"
    )
    enable_mp_occ: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Enable multi-plane occupancy.",
        display_name="enable multi-plane occupancy"
    )
    depth_scale: float = FLOAT_FIELD(
        value=25.0,
        default_value=25.0,
        description="Depth scale.",
        display_name="depth scale"
    )
    num_thing_classes: int = INT_FIELD(
        value=9,
        default_value=9,
        description="Number of thing classes.",
        display_name="number of thing classes"
    )
