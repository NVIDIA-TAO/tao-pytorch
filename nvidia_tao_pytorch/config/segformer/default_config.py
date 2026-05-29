# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from omegaconf import MISSING

from nvidia_tao_pytorch.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    BOOL_FIELD,
    FLOAT_FIELD,
    LIST_FIELD,
    DICT_FIELD,
    DATACLASS_FIELD
)
from nvidia_tao_pytorch.config.common.common_config import (
    CommonExperimentConfig,
    ExportConfig,
    TrainConfig,
    EvaluateConfig,
    InferenceConfig,
    GenTrtEngineConfig,
    TrtConfig,
    CalibrationConfig
)
from nvidia_tao_pytorch.config.common.quantization import ModelQuantizationConfig, QuantCalibrationDataset


@dataclass
class SFOptimConfig:
    """Optimizer config."""

    monitor_name: str = STR_FIELD(
        value="val_loss",
        default_value="val_loss",
        description="Monitor Name"
    )
    optim: str = STR_FIELD(
        value="adamw",
        default_value="adamw",
        description="Optimizer",
        valid_options="adamw,adam,sgd"
    )
    lr: float = FLOAT_FIELD(
        value=0.00006,
        default_value=0.00006,
        valid_min=0,
        valid_max="inf",
        automl_enabled="TRUE",
        description="Optimizer learning rate"
    )
    policy: str = STR_FIELD(
        value="linear",
        default_value="linear",
        valid_options="linear,step",
        description="Optimizer policy"
    )
    momentum: float = FLOAT_FIELD(
        value=0.9,
        default_value=0.9,
        valid_min=0.0,
        valid_max=1.0,
        display_name="momentum - AdamW (beta1)",
        description="The momentum (beta1) for the AdamW optimizer.",
        automl_enabled="TRUE"
    )
    weight_decay: float = FLOAT_FIELD(
        value=0.01,
        default_value=0.01,
        valid_min=0.0,
        valid_max=1.0,
        display_name="weight decay",
        description="The weight decay coefficient.",
        automl_enabled="TRUE"
    )


@dataclass
class SegFormerHeadConfig:
    """Configuration parameters for SegFormer Head."""

    in_channels: List[int] = LIST_FIELD(
        arrList=[64, 128, 320, 512],
        description="number of input channels to decoder"
    )  # FANHybrid-S
    in_index: List[int] = LIST_FIELD(
        arrList=[0, 1, 2, 3],
        description="Input index for the head",
        display_name="Input Index",
        default_value=[0, 1, 2, 3]
    )  # No change
    feature_strides: List[int] = LIST_FIELD(
        arrList=[4, 8, 16, 32],
        description="Feature strides for the head",
        display_name="Feature Strides",
        default_value=[4, 8, 16, 32]
    )  # No change
    align_corners: bool = BOOL_FIELD(
        value=False,
        description="Align corners for the head",
        display_name="Align Corners",
        default_value=False
    )
    decoder_params: Dict[str, int] = DICT_FIELD(
        hashMap={"embed_dim": 768},
        description="Decoder parameters for the head",
        display_name="Decoder Parameters",
        default_value={"embed_dim": 768}
    )  # 256, 512, 768 -> Configurable


@dataclass
class BackboneConfig:
    """Configuration parameters for Backbone."""

    type: str = STR_FIELD(
        value="fan_small_12_p4_hybrid",
        default_value="fan_small_12_p4_hybrid",
        description="Backbone architure",
        display_name="Backbone architectures",
        valid_options=",".join([
            "fan_tiny_8_p4_hybrid",
            "fan_large_16_p4_hybrid",
            "fan_small_12_p4_hybrid",
            "fan_base_16_p4_hybrid",
            "vit_large_nvdinov2",
            "vit_giant_nvdinov2",
            "vit_base_nvclip_16_siglip",
            "vit_huge_nvclip_14_siglip"
        ]),
    )
    feat_downsample: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Feature downsample",
        description="Feature downsample for fan base backbone"
    )
    pretrained_backbone_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Path to the pretrained model"
    )
    freeze_backbone: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to freeze backbone",
        automl_enabled="TRUE"
    )


@dataclass
class SFModelConfig:
    """SF Model config."""

    backbone: BackboneConfig = DATACLASS_FIELD(BackboneConfig())
    decode_head: SegFormerHeadConfig = DATACLASS_FIELD(SegFormerHeadConfig())


@dataclass
class RandomFlip:
    """RandomFlip augmentation config."""

    vflip_probability: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0,
        valid_max=1,
        description="Vertical Flip probability",
        automl_enabled="TRUE"
    )
    hflip_probability: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0,
        valid_max=1,
        description="Horizontal Flip probability",
        automl_enabled="TRUE"
    )
    enable: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to enable augmentation",
        automl_enabled="TRUE"
    )


@dataclass
class RandomRotation:
    """RandomRotation augmentation config."""

    rotate_probability: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0,
        valid_max=1,
        description="Random Rotate probability",
        automl_enabled="TRUE"
    )
    angle_list: List[float] = LIST_FIELD(
        arrList=[90, 180, 270],
        default_value=[90, 180, 270],
        description="Random rotate angle probability"
    )
    enable: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to enable augmentation",
        automl_enabled="TRUE"
    )


@dataclass
class RandomColor:
    """RandomColor augmentation config."""

    brightness: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        valid_min=0.0,
        valid_max=2.0,
        description="Random Color Brightness (torchvision ColorJitter range)",
        automl_enabled="TRUE"
    )
    contrast: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        valid_min=0.0,
        valid_max=2.0,
        description="Random Color Contrast (torchvision ColorJitter range)",
        automl_enabled="TRUE"
    )
    saturation: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        valid_min=0.0,
        valid_max=2.0,
        description="Random Color Saturation (torchvision ColorJitter range)",
        automl_enabled="TRUE"
    )
    hue: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        valid_min=0.0,
        valid_max=0.5,
        description="Random Color Hue (torchvision ColorJitter requires 0 <= hue <= 0.5)",
        automl_enabled="TRUE"
    )
    enable: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to enable Random Color",
        automl_enabled="TRUE"
    )
    color_probability: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0,
        valid_max=1,
        description="Random Color Probability",
        automl_enabled="TRUE"
    )


@dataclass
class RandomCropWithScale:
    """RandomCropWithScale augmentation config."""

    scale_range: List[float] = LIST_FIELD(
        arrList=[1, 1.2],
        default_value=[1, 1.2],
        description="Random Scale range"
    )  # non configurable here
    enable: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to enable Random Crop with Scale",
        automl_enabled="TRUE"
    )


@dataclass
class SFAugmentationSegmentConfig:
    """Augmentation config for segmentation."""

    random_flip: RandomFlip = DATACLASS_FIELD(RandomFlip())
    random_rotate: RandomRotation = DATACLASS_FIELD(RandomRotation())
    random_color: RandomColor = DATACLASS_FIELD(RandomColor())
    with_scale_random_crop: RandomCropWithScale = DATACLASS_FIELD(RandomCropWithScale())
    with_random_blur: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to enable with_random_blur"
    )
    with_random_crop: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to enable with_random_crop"
    )
    mean: List[float] = LIST_FIELD(
        arrList=[0.5, 0.5, 0.5],
        default_value=[0.5, 0.5, 0.5],
        description="Mean for the augmentation",
        display_name="Mean"
    )  # non configurable here
    std: List[float] = LIST_FIELD(
        arrList=[0.5, 0.5, 0.5],
        default_value=[0.5, 0.5, 0.5],
        description="Standard deviation for the augmentation",
        display_name="Standard Deviation"
    )  # non configurable here


@dataclass
class DataPathFormat:
    """Dataset Path experiment config."""

    csv_path: str = STR_FIELD(value=MISSING, default_value="", description="Path to csv file for dataset")
    images_dir: str = STR_FIELD(value=MISSING, default_value="", description="Path to images directory for dataset")


@dataclass
class SFDatasetSegmentConfig:
    """Segmentation Dataset Config."""

    root_dir: str = STR_FIELD(value=MISSING, default_value="", description="Path to root directory for dataset")
    dataset: str = STR_FIELD(
        value="SFDataset",
        default_value="SFDataset",
        valid_options="SFDataset",
        description="dataset class"
    )
    num_classes: int = INT_FIELD(
        value=2,
        default_value=2,
        description="The number of classes in the training data",
        math_cond=">0",
        valid_min=2,
        valid_max="inf"
    )
    img_size: int = INT_FIELD(
        value=256,
        default_value=256,
        description="The input image size"
    )
    batch_size: int = INT_FIELD(
        value=-1,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="Batch size",
        display_name="Batch Size"
    )
    workers: int = INT_FIELD(
        value=8,
        default_value=1,
        valid_min=0,
        valid_max="inf",
        description="Workers",
        display_name="Workers",
    )
    shuffle: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Shuffle dataloader"
    )
    train_split: str = STR_FIELD(
        value="train",
        default_value="train",
        description="Train split folder name"
    )
    validation_split: str = STR_FIELD(
        value="val",
        default_value="val",
        description="Validation split folder name"
    )
    test_split: str = STR_FIELD(
        value="val",
        default_value="val",
        description="Test split folder name"
    )
    predict_split: str = STR_FIELD(
        value="test",
        default_value="test",
        description="Predict split folder name"
    )
    augmentation: SFAugmentationSegmentConfig = DATACLASS_FIELD(SFAugmentationSegmentConfig())
    label_transform: str = STR_FIELD(
        value="norm",
        default_value="norm",
        valid_options="norm,None",
        description="label transform"
    )
    palette: Optional[List[Dict[Any, Any]]] = LIST_FIELD(
        arrList=[
            {"label_id": 0, "mapping_class": "foreground", "rgb": [0, 0, 0], "seg_class": "foreground"},
            {"label_id": 1, "mapping_class": "background", "rgb": [1, 1, 1], "seg_class": "background"}
        ],
        description="Palette, be careful of label_transform, if norm then RGB value from 0~1, else 0~255",
        display_name="Palette"
    )
    quant_calibration_dataset: QuantCalibrationDataset = DATACLASS_FIELD(
        QuantCalibrationDataset(),
        description="Configurable parameters for the quantization calibration dataset.",
    )


@dataclass
class SFDatasetConfig:
    """Dataset config."""

    segment: SFDatasetSegmentConfig = DATACLASS_FIELD(SFDatasetSegmentConfig())


@dataclass
class TensorBoardLogger:
    """Configuration for the tensorboard logger."""

    enabled: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to enable tensorboard"
    )
    infrequent_logging_frequency: int = INT_FIELD(
        value=2,
        default_value=2,
        valid_min=0,
        valid_max="inf",
        description="infrequent_logging_frequency"
    )  # Defined per epoch


@dataclass
class SFTrainSegmentConfig:
    """Segmentation loss Config."""

    loss: str = STR_FIELD(
        value="ce",
        default_value="ce",
        valid_options="ce",
        description="ChangeNet Segment loss"
    )
    weights: List[float] = LIST_FIELD(
        arrList=[0.5, 0.5, 0.5, 0.8, 1.0],
        default_value=[0.5, 0.5, 0.5, 0.8, 1.0],
        description="Multi-scale Segment loss weight"
    )


@dataclass
class SFTrainExpConfig(TrainConfig):
    """Train Config."""

    optim: SFOptimConfig = DATACLASS_FIELD(SFOptimConfig())
    pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Pretrained model path",
        display_name="pretrained model path"
    )
    segment: SFTrainSegmentConfig = DATACLASS_FIELD(SFTrainSegmentConfig())
    tensorboard: Optional[TensorBoardLogger] = DATACLASS_FIELD(TensorBoardLogger())
    use_distributed_sampler: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Use distributed sampler for multi-GPU training",
        display_name="use_distributed_sampler"
    )
    sync_batchnorm: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable synchronized batch normalization for multi-GPU training",
        display_name="sync_batchnorm"
    )

    checkpointer: Optional[Dict[str, Any]] = None
    enable_lr_monitor: Optional[bool] = False


@dataclass
class SFEvalExpConfig(EvaluateConfig):
    """Evaluation experiment config."""

    vis_after_n_batches: int = INT_FIELD(
        value=16,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        description="Visualize evaluation segmentation results after n batches"
    )
    batch_size: int = INT_FIELD(
        value=-1,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="Batch size",
        display_name="Batch Size"
    )
    checkpoint: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="Path to checkpoint file",
        display_name="Path to checkpoint file"
    )


@dataclass
class SFInferenceExpConfig(InferenceConfig):
    """Inference experiment config."""

    vis_after_n_batches: int = INT_FIELD(
        value=16,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        description="Visualize evaluation segmentation results after n batches"
    )
    batch_size: int = INT_FIELD(
        value=-1,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="Batch size",
        display_name="Batch Size"
    )
    checkpoint: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="Path to checkpoint file",
        display_name="Path to checkpoint file"
    )


@dataclass
class SFExportExpConfig(ExportConfig):
    """Export experiment config."""

    input_width: int = INT_FIELD(
        value=544,
        default_value=544,
        description="Width of the input image tensor.",
        display_name="input width",
        valid_min=32,
    )

    serialize_nvdsinfer: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Serialize DeepStream config.",
        description=(
            "Flag to enable serializing the required configs for integrating with DeepStream."
        )
    )


@dataclass
class SFTrtConfig(TrtConfig):
    """Trt config."""

    data_type: str = STR_FIELD(
        value="FP32",
        default_value="fp16",
        description="Data type",
        display_name="Data type"
    )
    calibration: CalibrationConfig = DATACLASS_FIELD(CalibrationConfig())


@dataclass
class SFGenTrtEngineExpConfig(GenTrtEngineConfig):
    """Gen TRT Engine experiment config."""

    tensorrt: SFTrtConfig = DATACLASS_FIELD(SFTrtConfig())


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: SFModelConfig = DATACLASS_FIELD(SFModelConfig())
    dataset: SFDatasetConfig = DATACLASS_FIELD(SFDatasetConfig())
    train: SFTrainExpConfig = DATACLASS_FIELD(SFTrainExpConfig())
    evaluate: SFEvalExpConfig = DATACLASS_FIELD(SFEvalExpConfig())
    inference: SFInferenceExpConfig = DATACLASS_FIELD(SFInferenceExpConfig())
    export: SFExportExpConfig = DATACLASS_FIELD(SFExportExpConfig())
    gen_trt_engine: SFGenTrtEngineExpConfig = DATACLASS_FIELD(SFGenTrtEngineExpConfig())
    quantize: ModelQuantizationConfig = DATACLASS_FIELD(
        ModelQuantizationConfig(),
        description="Configurable parameters to run model quantization for a SegFormer experiment.",
    )

    def __post_init__(self):
        """Set default model name for SegFormer."""
        if self.model_name is None:
            self.model_name = "segformer"
