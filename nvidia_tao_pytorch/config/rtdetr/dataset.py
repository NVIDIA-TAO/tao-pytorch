# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema for the dataset."""

from dataclasses import dataclass
from typing import Optional, List, Dict

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    DICT_FIELD,
    STR_FIELD,
)


@dataclass
class RTAugmentationConfig:
    """Augmentation config."""

    multi_scales: List[int] = LIST_FIELD(
        arrList=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
        description="A list of sizes to perform random resize.",
        display_name="multi-scales"
    )
    train_spatial_size: List[int] = LIST_FIELD(
        arrList=[640, 640],
        description="Input resolution to run evaluation during training. This is in the [h, w] order.",
        display_name="train spatial size"
    )
    eval_spatial_size: List[int] = LIST_FIELD(
        arrList=[640, 640],
        description="Input resolution to run evaluation during validation and testing. This is in the [h, w] order.",
        display_name="evaluation spatial size"
    )
    distortion_prob: float = FLOAT_FIELD(
        value=0.8,
        default_value=0.8,
        description="The probability for RandomPhotometricDistort",
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="TRUE",
        display_name="distortion probability"
    )
    iou_crop_prob: float = FLOAT_FIELD(
        value=0.8,
        default_value=0.8,
        description="The probability for RandomIoUCrop",
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="TRUE",
        display_name="iou crop probability"
    )
    preserve_aspect_ratio: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="preserve aspect ratio",
        description="""Flag to enable resize with preserving the aspect ratio."""
    )


@dataclass
class RTDatasetConfig:
    """Dataset config."""

    train_data_sources: Optional[List[Dict[str, str]]] = LIST_FIELD(
        arrList=None,
        default_value=[{"image_dir": "", "json_file": ""}],
        description="""The list of data sources for training:
                    * image_dir : The directory that contains the training images
                    * json_file : The path of the JSON file, which uses training-annotation COCO format""",
        display_name="train data sources",
    )
    val_data_sources: Optional[Dict[str, str]] = DICT_FIELD(
        hashMap=None,
        arrList=None,
        default_value={"image_dir": "", "json_file": ""},
        description="""The list of data sources for validation:
                    * image_dir : The directory that contains the validation images
                    * json_file : The path of the JSON file, which uses validation-annotation COCO format""",
        display_name="validation data sources",
    )
    test_data_sources: Optional[Dict[str, str]] = DICT_FIELD(
        hashMap=None,
        default_value={"image_dir": "", "json_file": ""},
        description="""The data source for testing:
                    * image_dir : The directory that contains the test images
                    * json_file : The path of the JSON file, which uses test-annotation COCO format""",
        display_name="test data sources",
    )
    infer_data_sources: Optional[Dict[str, str]] = DICT_FIELD(
        hashMap=None,
        default_value={"image_dir": [""], "classmap": ""},
        description="""The data source for inference:
                    * image_dir : The list of directories that contains the inference images
                    * classmap : The path of the .txt file that contains class names""",
        display_name="infer data sources",
    )
    quant_calibration_data_sources: Optional[Dict[str, str]] = DICT_FIELD(
        hashMap=None,
        default_value={"image_dir": "", "json_file": ""},
        description="""The data source for quantization calibration:
                    * image_dir : The directory that contains the quantization calibration images
                    * json_file(optional) : The path of the JSON file, which uses quantization calibration-\
                        annotation COCO format""",
        display_name="quantization calibration data sources",
    )
    batch_size: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=1,
        valid_max="inf",
        description="The batch size for training and validation",
        automl_enabled="TRUE",
        display_name="batch size"
    )
    workers: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="The number of parallel workers processing data",
        automl_enabled="TRUE",
        display_name="batch size"
    )
    remap_mscoco_category: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="remap mscoco category",
        description="""Flag to enable mapping of MSCOCO 91 classes to 80. Only required if we're directly
                    training using the original COCO annotation files.
                    For custom dataset, this value needs to be set False"""
    )
    pin_memory: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="pin_memory",
        description="""Flag to enable the dataloader to allocated pagelocked memory for faster
                    of data between the CPU and GPU."""
    )
    dataset_type: str = STR_FIELD(
        value="serialized",
        default_value="serialized",
        display_name="dataset type",
        description="""If set to default, we follow the standard CocoDetection` dataset structure
                    from the torchvision which loads COCO annotation in every subprocess. This leads to redudant
                    copy of data and can cause RAM to explod if workers` is high. If set to serialized,
                    the data is serialized through pickle and torch.Tensor` that allows the data to be shared
                    across subprocess. As a result, RAM usage can be greatly improved.""",
        valid_options=",".join(["serialized", "default"])
    )
    num_classes: int = INT_FIELD(
        value=80,
        default_value=80,
        description="The number of classes in the training data",
        math_cond=">0",
        valid_min=1,
        valid_max="inf",
        display_name="num classes"
    )
    eval_class_ids: Optional[List[int]] = LIST_FIELD(
        arrList=None,
        default_value=[1],
        description="""IDs of the classes for evaluation.""",
        display_name="eval class ids",
    )
    augmentation: RTAugmentationConfig = DATACLASS_FIELD(
        RTAugmentationConfig(),
        description="Configuration parameters for data augmentation",
        display_name="augmentation",
    )
