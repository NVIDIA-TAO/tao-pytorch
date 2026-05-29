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
    DICT_FIELD,
    STR_FIELD,
)


@dataclass
class DDDatasetConvertConfig:
    """Dataset Convert config."""

    input_source: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Path to the .txt files listing the data sources.",
        display_name="input source"
    )
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
    image_dir_name: str = STR_FIELD(
        value=None,
        default_value="",
        description="Name of the image directory relative to the dataset root.",
        display_name="image directory name"
    )
    label_dir_name: str = STR_FIELD(
        value=None,
        default_value="",
        description="Name of the directory containing the annotations, relative to the dataset root.",
        display_name="label directory name"
    )
    val_split: int = INT_FIELD(
        value=0,
        default_value=0,
        description="The id of the validation split if the partitions are > 1.",
        display_name="validation split",
        valid_min=0
    )
    num_shards: int = INT_FIELD(
        value=20,
        default_value=10,
        description="The number of shards per partition of the dataset.",
        display_name="number of shards",
        valid_min=0
    )
    num_partitions: int = INT_FIELD(
        value=1,
        default_value=1,
        description="The number of partitions the dataset.",
        display_name="number of partitions",
        valid_min=1
    )
    partition_mode: Optional[str] = STR_FIELD(
        value=None,
        default_value="random",
        description="""The method employed when partitioning the data to multiple folds. Two methods are supported:

        Random partitioning: The data is divided in to 2 folds,
            train and val. This mode requires that the val_split parameter be set.
        Sequence-wise partitioning: The data is divided into n partitions
            (defined by the num_partitions parameter) based on the number of sequences available.
        """,
        display_name="partition mode"
    )
    image_extension: str = STR_FIELD(
        value=".jpg",
        default_value=".jpg",
        description="The extension of the images in the directory.",
        display_name="image extension"
    )
    mapping_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Path to a JSON file containing the class mapping.",
        display_name="mapping path"
    )


@dataclass
class DDAugmentationConfig:
    """Augmentation config."""

    scales: List[int] = LIST_FIELD(
        arrList=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
        description="A list of sizes to perform random resize.",
        display_name="scales",
        value_type="list_2",
    )
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
    train_random_resize: List[int] = LIST_FIELD(
        arrList=[400, 500, 600],
        description="A list of sizes to perform random resize for training data",
        display_name="train random resize dimensions"
    )
    horizontal_flip_prob: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="The probability for horizonal flip during training",
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="TRUE",
        display_name="horizontal flip probability"
    )
    train_random_crop_min: int = INT_FIELD(
        value=384,
        valid_min=32,
        valid_max="inf",
        description="The minimum random crop size for training data",
        automl_enabled="TRUE",
        display_name="minumum random crop size",
        parent_param="TRUE"
    )
    train_random_crop_max: int = INT_FIELD(
        value=600,
        valid_min=32,
        valid_max=1333,
        description="The maximum random crop size for training data. Must be > train_random_crop_min",
        automl_enabled="TRUE",
        math_cond="> depends_on",
        display_name="Maximum random crop size",
        depends_on="dataset.augmentation.train_random_crop_min"
    )
    random_resize_max_size: int = INT_FIELD(
        value=1333,
        valid_min=1,
        valid_max="inf",
        description="The maximum random resize size for training data",
        display_name="random resize max size"
    )
    test_random_resize: int = INT_FIELD(
        value=800,
        valid_min=32,
        valid_max="inf",
        description="The random resize size for test data",
        automl_enabled="FALSE",
        display_name="random resize max size"
    )
    fixed_padding: bool = BOOL_FIELD(
        value="TRUE",
        default_value="TRUE",
        description=(
            "A flag specifying whether to resize the image (with no padding) to "
            "(sorted(scales[-1]), random_resize_max_size) to prevent a CPU memory leak."
        ),
        display_name="fixed padding"
    )
    fixed_random_crop: Optional[int] = INT_FIELD(
        value=None,
        default_value=1024,
        description=(
            "A flag to enable Large Scale Jittering, which is used for ViT backbones. "
            "The resulting image resolution is fixed to fixed_random_crop."
        ),
        display_name="fixed random crop",
        valid_min=1,
        valid_max="inf"
    )


@dataclass
class DDDatasetConfig:
    """Dataset config."""

    train_sampler: str = STR_FIELD(
        value="default_sampler",
        default_value="default_sampler",
        description=(
            "The minibatch sampling method. Non-default sampling methods can be enabled for multi-node jobs. "
            "The config doesn't have any effect if the :code:`dataset_type` isn't set to `default`."
        ),
        valid_options=",".join(["default_sampler", "non_uniform_sampler", "uniform_sampler"]),
        display_name="train sampler"
    )

    train_data_sources: Optional[List[Dict[str, str]]] = LIST_FIELD(
        arrList=None,
        default_value=[{"image_dir": "", "json_file": ""}],
        description="""The list of data sources for training:
                    * image_dir : The directory that contains the training images
                    * json_file : The path of the JSON file, which uses training-annotation COCO format""",
        display_name="train data sources",
    )
    val_data_sources: Optional[List[Dict[str, str]]] = LIST_FIELD(
        arrList=None,
        default_value=[{"image_dir": "", "json_file": ""}],
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
    pin_memory: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="pin_memory",
        description=(
            "Flag to enable the dataloader to allocated pagelocked memory for faster "
            "of data between the CPU and GPU."
        )
    )
    dataset_type: str = STR_FIELD(
        value="serialized",
        default_value="serialized",
        display_name="dataset type",
        description=(
            "If set to default, we follow the standard CocoDetection` dataset structure "
            "from the torchvision which loads COCO annotation in every subprocess. This leads to redudant "
            "copy of data and can cause RAM to explod if workers` is high. If set to serialized, "
            "the data is serialized through pickle and torch.Tensor` that allows the data to be shared "
            "across subprocess. As a result, RAM usage can be greatly improved."
        ),
        valid_options=",".join(["serialized", "default"])
    )
    num_classes: int = INT_FIELD(
        value=91,
        default_value=91,
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
    augmentation: DDAugmentationConfig = DATACLASS_FIELD(
        DDAugmentationConfig(),
        description="Configuration parameters for data augmentation",
        display_name="augmentation",
    )
