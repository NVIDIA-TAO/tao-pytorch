# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the dataset."""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    BOOL_FIELD,
    LIST_FIELD,
    DICT_FIELD,
    DATACLASS_FIELD
)


@dataclass
class BEVFusionDataPreprocessorConfig:
    """Dataset PreProcessor config."""

    type: str = STR_FIELD(
        value="Det3DDataPreprocessor",
        default_value="Det3DDataPreprocessor",
        display_name="Data Pre-processor Type",
        description="Name of Data Pre-processor for 3D Fusion"
    )
    mean: List[float] = LIST_FIELD(
        arrList=[123.675, 116.28, 103.53],
        description="The input mean for RGB frames",
        display_name="input mean per pixel"
    )
    std: List[float] = LIST_FIELD(
        arrList=[58.395, 57.12, 57.375],
        description="The input standard deviation per pixel for RGB frames",
        display_name="input std per pixel"
    )
    bgr_to_rgb: bool = BOOL_FIELD(
        value=False,
        default_value=32,
        display_name="no convert bgr to rgb",
        description="whether to convert image from BGR to RGB."
    )
    pad_size_divisor: int = INT_FIELD(
        value=32,
        default_value=32,
        display_name="pad size divisor",
        description="The size of padded image should be divisible."
    )
    voxelize_cfg: Dict[Any, Any] = DICT_FIELD(
        hashMap={"max_num_points": 10,
                 "max_voxels": [120000, 160000],
                 "voxelize_reduce": True}
    )


@dataclass
class BEVFusionDatasetConfig:
    """Dataset config."""

    ann_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="annotation file",
        description="A path to the annotation pkl file"
    )
    data_prefix: Dict[str, str] = DICT_FIELD(
        hashMap={'pts': 'training/lidar_reduced',
                 'img': 'training/images/'},
        display_name="data prefix for points and images",
        description="Corresponding data prefix for points and images"
    )
    sampler: str = STR_FIELD(
        value="DefaultSampler",
        default_value="DefaultSampler",
        display_name="default data sampler",
        description="Name of data sampler."
    )
    batch_size: int = INT_FIELD(
        value=4,
        valid_min=1,
        valid_max="inf",
        default_value=4,
        display_name="batch size",
        description="The batch size for training and validation",
        automl_enabled="TRUE"
    )
    num_workers: int = INT_FIELD(
        value=8,
        valid_max="inf",
        valid_min=1,
        default_value=8,
        display_name="num workers",
        description="The number of parallel workers processing data",
        automl_enabled="TRUE"
    )
    pin_memory: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="pin_memory",
        description="""Flag to enable the dataloader to allocate pagelocked memory for faster
                       of data between the CPU and GPU."""
    )
    repeat_time: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        display_name="dataset repeat number",
        description="The number of repetition of the dataset when training."
    )


@dataclass
class BEVFusionDatasetExpConfig:
    """Dataset Experiment config."""

    type: str = STR_FIELD(
        value="KittiPersonDataset",
        default_value="KittiPersonDataset",
        display_name="dataset type",
        description="Dataset types for 3D Fusion",
        valid_options=",".join(["TAO3DSyntheticDataset", "TAO3DDataset", "KittiPersonDataset"])
    )
    root_dir: str = STR_FIELD(
        value="",
        default_value="",
        display_name="root directory of the dataset",
        description="A path to the root directory of the given dataset"
    )
    classes: List[str] = LIST_FIELD(
        arrList=['person'],
        default_value=['person'],
        display_name="list of classes",
        description="A List of the classes to be trained."
    )
    box_type_3d: str = STR_FIELD(
        value="lidar",
        default_value="lidar",
        display_name="3d bbox type in training",
        description="3D bounding boxes type to be used when training.",
        valid_options=",".join(["lidar", "camera"])
    )
    gt_box_type: str = STR_FIELD(
        value="camera",
        default_value="camera",
        display_name="3d bbox type in ground truth",
        description="3D bounding boxes type in ground truth.",
        valid_options=",".join(["lidar", "camera"])
    )
    origin: List[float] = LIST_FIELD(
        arrList=[0.5, 1.0, 0.5],
        display_name="bbox center origin",
        description="The origin of the given center point in ground truth 3D bounding boxes.",
    )
    default_cam_key: str = STR_FIELD(
        value="CAM2",
        default_value="CAM2",
        display_name="default camera name",
        description="Default camera name in dataset"
    )
    per_sequence: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="is per sequence",
        description="Whether to save results in per sequence format."
    )
    num_views: int = INT_FIELD(
        value=1,
        default_value=1,
        display_name="number of camera view",
        description="Number of camera view in dataset."
    )
    point_cloud_dim: int = INT_FIELD(
        value=4,
        default_value=4,
        display_name="point cloud data dimension",
        description="Input lidar point cloud data dimension"
    )
    train_dataset: BEVFusionDatasetConfig = DATACLASS_FIELD(
        BEVFusionDatasetConfig(),
        description="Configurable parameters to construct the train dataset."
    )
    val_dataset: BEVFusionDatasetConfig = DATACLASS_FIELD(
        BEVFusionDatasetConfig(),
        description="Configurable parameters to construct the validation dataset."
    )
    test_dataset: BEVFusionDatasetConfig = DATACLASS_FIELD(
        BEVFusionDatasetConfig(),
        description="Configurable parameters to construct the test dataset."
    )
    img_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="infer image file",
        description="Image file for single file inference"
    )
    pc_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="infer point cloud file",
        description="Point cloud file for single file inference"
    )
    cam2img: Optional[List[float]] = LIST_FIELD(
        arrList=None,
        default_value=None,
        display_name="camera instrinsics",
        description="Camera instrinsic matrix for single file inference"
    )
    lidar2cam: Optional[List[float]] = LIST_FIELD(
        arrList=None,
        default_value=None,
        display_name="lidar to camera extrinsic",
        description="Lidar to camera extrinsic matrix for single file inference"
    )
