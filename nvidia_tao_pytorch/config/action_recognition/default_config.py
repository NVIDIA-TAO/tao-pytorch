# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file"""

from typing import Optional, List, Dict
from dataclasses import dataclass
from omegaconf import MISSING

from nvidia_tao_pytorch.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    BOOL_FIELD,
    LIST_FIELD,
    DICT_FIELD,
    DATACLASS_FIELD,
)
from nvidia_tao_pytorch.config.common.common_config import (
    CommonExperimentConfig,
    TrainConfig,
    EvaluateConfig,
    InferenceConfig
)


@dataclass
class ARModelConfig:
    """Action recognition model config."""

    model_type: str = STR_FIELD(
        value="rgb",
        value_type="ordered",
        default_value="rgb",
        valid_options="rgb,of,joint",
        description="The type of model architecture: [rgb, of, joint].",
        display="model type"
    )
    backbone: str = STR_FIELD(
        value="resnet_18",
        value_type="ordered",
        default_value="resnet_18",
        valid_options="resnet_18,resnet_34,resnet_50,resnet_101,resnet_152,i3d",
        description="The backbone of model architecture.",
        display="backbone"
    )
    input_type: str = STR_FIELD(
        value="2d",
        value_type="ordered",
        default_value="2d",
        valid_options="2d,3d",
        description="The type of model input: [2d, 3d].",
        display="input type"
    )
    of_seq_length: int = INT_FIELD(
        value=10,
        default_value=10,
        valid_min=1,
        valid_max="inf",
        description="The optical flow sequence length.",
        display="of seq length"
    )
    of_pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The pretrained weights for optical flow model.",
        display="of pretrained model path"
    )
    of_pretrained_num_classes: int = INT_FIELD(
        value=5,
        default_value=5,
        valid_min=0,
        valid_max="inf",
        description="The classes number of the pretrained weights for optical flow model.",
        display="of pretrained num classes"
    )  # 0 means the pretrained model has the same classes number
    rgb_seq_length: int = INT_FIELD(
        value=3,
        default_value=3,
        valid_min=1,
        valid_max="inf",
        description="The RGB sequence length.",
        display="rgb seq length"
    )
    rgb_pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The pretrained weights for RGB model.",
        display="rgb pretrained model path"
    )
    rgb_pretrained_num_classes: int = INT_FIELD(
        value=5,
        default_value=5,
        valid_min=0,
        valid_max="inf",
        description="The classes number of the pretrained weights for RGB model.",
        display="rgb pretrained num classes"
    )  # 0 means the pretrained model has the same classes number
    num_fc: int = INT_FIELD(
        value=64,
        default_value=64,
        valid_min=1,
        valid_max="inf",
        description="The number of hidden units in fully-connected layer.",
        display="num fc"
    )
    joint_pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The pretrained weights for joint pretrained model.",
        display="joint pretrained model path"
    )
    sample_strategy: str = STR_FIELD(
        value="random_interval",
        value_type="ordered",
        default_value="random_interval",
        valid_options="random_interval,consecutive",
        description="The sample strategy to sample frames from videos.",
        display="sample strategy"
    )
    sample_rate: int = INT_FIELD(
        value=1,
        default_value=1,
        description="The sample rate to sample frames from videos.",
        display="sample rate"
    )
    imagenet_pretrained: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="The bool flag to load imagenet pretrained weights.",
        display="imagenet pretrained"
    )  # Only for internal use. Will change to False when release
    # 0.0 for resnet18 2D on SHAD, 0.5 for I3D on HMDB51, 0.8 for ResNet3D on HMDB51
    dropout_ratio: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="TRUE",
        description="The dropout ratio for the model.",
        display="dropout ratio"
    )
    input_width: int = INT_FIELD(
        value=224,
        default_value=224,
        valid_min=1,
        valid_max="inf",
        description="The input width of the model.",
        display="input width"
    )
    input_height: int = INT_FIELD(
        value=224,
        default_value=224,
        valid_min=1,
        valid_max="inf",
        description="The input height of the model.",
        display="input height"
    )


@dataclass
class OptimConfig:
    """Optimizer config."""

    lr: float = FLOAT_FIELD(
        value=5e-4,
        default_value=5e-4,
        valid_min=0.0,
        valid_max="inf",
        automl_enabled="TRUE",
        description="Learning rate for training.",
        display="lr"
    )
    momentum: float = FLOAT_FIELD(
        value=0.9,
        default_value=0.9,
        valid_min=0.0,
        valid_max=1.0,
        description="Momentum coefficient for SGD.",
        display="momentum"
    )
    weight_decay: float = FLOAT_FIELD(
        value=5e-4,
        default_value=5e-4,
        valid_min=0,
        valid_max="inf",
        description="Weight decay coefficient for trainng.",
        display="weight decay"
    )
    lr_scheduler: str = STR_FIELD(
        value="MultiStep",
        value_type="ordered",
        default_value="MultiStep",
        valid_options="MultiStep,AutoReduce",
        description="Learning rate scheduler.",
        display="lr scheduler"
    )
    lr_monitor: str = STR_FIELD(
        value="val_loss",
        value_type="ordered",
        default_value="val_loss",
        valid_options="val_loss,train_loss",
        description="Learning rate monitor for AutoReduce learning rate scheduler.",
        display="lr monitor"
    )
    patience: int = INT_FIELD(
        value=1,
        default_value=1,
        description="Number of epochs for AutoReduce learning rate scheduler tolerance.",
        display="patience"
    )
    min_lr: float = FLOAT_FIELD(
        value=1e-4,
        default_value=1e-4,
        valid_min=0,
        valid_max="inf",
        description="Minimum learning rate.",
        display="min lr"
    )
    lr_steps: List[int] = LIST_FIELD(
        arrList=[15, 25],
        description="Steps to change learning rate in MultiStep scheduler.",
        display="lr steps",
        value_type="list_2"
    )
    lr_decay: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        valid_min=0.0,
        valid_max=1.0,
        description="Learning rate decay factor in learning rate scheduler.",
        display="lr decay"
    )


@dataclass
class ARAugmentationConfig:
    """Augmentation config."""

    train_crop_type: str = STR_FIELD(
        value="random_crop",
        value_type="ordered",
        default_value="random_crop",
        valid_options="random_crop,multi_scale_crop,no_crop",
        description="Crop type to crop image patches from the original input image.",
        display="train crop type"
    )
    scales: List[float] = LIST_FIELD(
        arrList=[1],
        description="Scales list for multi_scale_crop.",
        display="scales"
    )
    horizontal_flip_prob: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.0,
        valid_max=1.0,
        description="Probability to apply horizontal flip to images.",
        display="horizontal flip prob"
    )
    rgb_input_mean: List[float] = LIST_FIELD(
        arrList=[0.485, 0.456, 0.406],
        description="Mean value per channel to be substructed for RGB input.",
        display="rgb input mean"
    )
    rgb_input_std: List[float] = LIST_FIELD(
        arrList=[0.229, 0.224, 0.225],
        description="Std value to be divided for RGB input.",
        display="rgb input std"
    )
    of_input_mean: List[float] = LIST_FIELD(
        arrList=[0.5],
        description="Mean value per channel to be substructed for optical flow input.",
        display="of input mean"
    )
    of_input_std: List[float] = LIST_FIELD(
        arrList=[0.5],
        description="Std value to be divided for optical flow input.",
        display="of input std"
    )
    val_center_crop: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Bool flag to apply center crop in validation.",
        display="val cetner crop"
    )
    crop_smaller_edge: int = INT_FIELD(
        value=256,
        default_value=256,
        valid_min=1,
        description="Smaller edge length of the center crop in validation.",
        display="crop smaller edge"
    )


@dataclass
class ARDatasetConfig:
    """Dataset config."""

    train_dataset_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Absolute path to train dataset.",
        display="train dataset dir"
    )
    val_dataset_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Absolute path to validation dataset.",
        display="val dataset dir"
    )
    label_map: Optional[Dict[str, int]] = DICT_FIELD(
        hashMap=None,
        description="Dict mapping the class to class index",
        display="label map"
    )
    batch_size: int = INT_FIELD(
        value=2,
        default_value=2,
        valid_min=1,
        description="Batch size of model input.",
        display="batch size"
    )
    workers: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=0,
        description="Number of workers to process data.",
        display="workers"
    )
    clips_per_video: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        description="Number of clips sampled from single video.",
        display="clips per video"
    )
    augmentation_config: ARAugmentationConfig = DATACLASS_FIELD(
        ARAugmentationConfig(),
        description="Configurable parameters for dataset augmentation."
    )


@dataclass
class ARTrainExpConfig(TrainConfig):
    """Train experiment config."""

    optim: OptimConfig = DATACLASS_FIELD(
        OptimConfig(),
        description="Configurable parameters for optimizer."
    )
    clip_grad_norm: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        description="The L2 magnitude of graident to be clipped in the training.",
        display="clip grad norm"
    )


@dataclass
class ARInferenceExpConfig(InferenceConfig):
    """Inference experiment config."""

    inference_dataset_dir: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="The absolute path to inference dataset.",
        display="inference dataset dir"
    )
    batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        description="Batch size for inference.",
        display="batch size"
    )
    video_inf_mode: str = STR_FIELD(
        value="center",
        value_type="ordered",
        default_value="center",
        valid_options="center,conv,all",
        description="The video sampling mode for inference.",
        display="video inf mode"
    )
    video_num_segments: int = INT_FIELD(
        value=1,
        default_value=1,
        description="The number of clips to do inference for single video.",
        display="video num segments"
    )


@dataclass
class AREvalExpConfig(EvaluateConfig):
    """Evaluation experiment config."""

    test_dataset_dir: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="The number of clips to do inference for single video.",
        display="video num segments"
    )
    batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        description="Batch size of data for evaluation.",
        display="batch size"
    )
    video_eval_mode: str = STR_FIELD(
        value="center",
        value_type="ordered",
        default_value="center",
        valid_options="center,conv,all",
        description="The video sampling mode for evaluation.",
        display="video eval mode"
    )
    video_num_segments: int = INT_FIELD(
        value=10,
        default_value=10,
        description="The number of clips to do inference for single video.",
        display="video num segments"
    )


@dataclass
class ARExportExpConfig:
    """Export experiment config."""

    checkpoint: str = STR_FIELD(
        value=MISSING,
        default_value=MISSING,
        description="The absolute path to checkpoint.",
        display="checkpoint"
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="The absolute path to results directory.",
        display="results dir"
    )
    onnx_file: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        description="The absolute path to exported onnx file.",
        display="onnx file"
    )
    gpu_id: int = INT_FIELD(
        value=0,
        default_value=0,
        description="GPU ID",
        display="gpu id"
    )
    batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        description="Dummy batch size for export.",
        display="batch size"
    )


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: ARModelConfig = DATACLASS_FIELD(
        ARModelConfig(),
        description="Configurable parameters for the model."
    )
    dataset: ARDatasetConfig = DATACLASS_FIELD(
        ARDatasetConfig(),
        description="Configurable parameters for the dataset."
    )
    train: ARTrainExpConfig = DATACLASS_FIELD(
        ARTrainExpConfig(),
        description="Configurable parameters for a train experiment."
    )
    evaluate: AREvalExpConfig = DATACLASS_FIELD(
        AREvalExpConfig(),
        description="Configurable parameters for an evaluation experiment."
    )
    export: ARExportExpConfig = DATACLASS_FIELD(
        ARExportExpConfig(),
        description="Configurable parameters for an export experiment."
    )
    inference: ARInferenceExpConfig = DATACLASS_FIELD(
        ARInferenceExpConfig(),
        description="Configurable parameters for an inference experiment."
    )

    def __post_init__(self):
        """Set default model name for Action Recognition."""
        if self.model_name is None:
            self.model_name = "action_recognition"
