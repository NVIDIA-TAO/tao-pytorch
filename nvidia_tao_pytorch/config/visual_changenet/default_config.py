# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file"""

from typing import Optional, List, Dict
from dataclasses import dataclass

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
    ExportConfig,
    TrainConfig,
    EvaluateConfig,
    InferenceConfig,
    GenTrtEngineConfig,
    TrtConfig,
    CalibrationConfig
)

from nvidia_tao_pytorch.config.common.quantization import (
    ModelQuantizationConfig,
    QuantCalibrationDataset,
)


@dataclass
class CNOptimConfig:
    """Optimizer config."""

    monitor_name: str = STR_FIELD(value="val_loss", default_value="val_loss", description="Monitor Name")
    optim: str = STR_FIELD(
        value="adamw",
        default_value="adamw",
        description="Optimizer",
        valid_options="adamw,adam,sgd"
    )
    lr: float = FLOAT_FIELD(
        value=0.0001,
        default_value=0.0001,
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
        math_cond="> 0.0",
        display_name="momentum - AdamW",
        description="The momentum for the AdamW optimizer.",
        automl_enabled="TRUE"
    )
    weight_decay: float = FLOAT_FIELD(
        value=0.01,
        default_value=0.01,
        valid_min=0.0,
        valid_max=1.0,
        math_cond="> 0.0",
        display_name="weight decay",
        description="The weight decay coefficient.",
        automl_enabled="TRUE"
    )


@dataclass
class ChangeNetHeadConfig:
    """Configuration parameters for Visual ChangeNet Head."""

    in_channels: List[int] = LIST_FIELD(
        arrList=[128, 256, 384, 384],
        description="number of input channels to decoder"
    )  # FANHybrid-S
    in_index: List[int] = LIST_FIELD(
        arrList=[0, 1, 2, 3],
        description="Input index for the head",
        display_name="Input Index",
        default_value=[0, 1, 2, 3]
    )  # No change
    feature_strides: List[int] = LIST_FIELD(
        arrList=[4, 8, 16, 16],
        description="Feature strides for the head",
        display_name="Feature Strides",
        default_value=[4, 8, 16, 16]
    )  # No change
    align_corners: bool = BOOL_FIELD(
        value=False,
        description="Align corners for the head",
        display_name="Align Corners",
        default_value=False
    )
    decoder_params: Dict[str, int] = DICT_FIELD(
        hashMap={"embed_dim": 256},
        description="Decoder parameters for the head",
        display_name="Decoder Parameters",
        default_value={"embed_dim": 256}
    )  # 256, 512, 768 -> Configurable
    use_summary_token: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Use summary token",
        description="Flag to use summary token"
    )


@dataclass
class BackboneConfig:
    """Configuration parameters for Backbone."""

    type: str = STR_FIELD(
        value="fan_small_12_p4_hybrid",
        default_value="fan_small_12_p4_hybrid",
        description="Backbone architure",
        display_name="Backbone architectures",
        valid_options=(
            "fan_tiny_8_p4_hybrid,fan_small_12_p4_hybrid,"
            "fan_base_16_p4_hybrid,fan_large_16_p4_hybrid,vit_large_nvdinov2"
        )
    )
    feat_downsample: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Feature downsample",
        description="Feature downsample"
    )
    pretrained_backbone_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Path to the pretrained model"
    )
    freeze_backbone: bool = BOOL_FIELD(value=False, default_value=False, description="Flag to freeze backbone")


@dataclass
class CNModelClassifyConfig:
    """CN Model Classification config."""

    train_margin_euclid: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        valid_min=1,
        valid_max="inf",
        description="Contrastive loss training margin",
        automl_enabled="TRUE"
    )
    eval_margin: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        valid_min=0,
        valid_max="inf",
        description="Evaluation threshold score for contrastive loss",
        automl_enabled="TRUE"
    )
    embedding_vectors: int = INT_FIELD(
        value=5,
        default_value=5,
        valid_min=1,
        valid_max="inf",
        description="Number of embedding vectors - architecture 1"
    )
    embed_dec: int = INT_FIELD(
        value=5,
        default_value=5,
        valid_min=1,
        valid_max="inf",
        description="Number of embedding vectors - architecture 2"
    )
    learnable_difference_modules: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=1,
        valid_max=4,
        description="Number of learnable difference modules",
        automl_enabled="TRUE"
    )
    difference_module: Optional[str] = STR_FIELD(
        "euclidean",
        default_value="euclidean",
        valid_options="learnable,euclidean",
        description="Type of difference module used - Choose architecture type"
    )


@dataclass
class CNModelConfig:
    """CN Model config."""

    backbone: BackboneConfig = DATACLASS_FIELD(BackboneConfig())
    decode_head: ChangeNetHeadConfig = DATACLASS_FIELD(ChangeNetHeadConfig())
    classify: CNModelClassifyConfig = DATACLASS_FIELD(CNModelClassifyConfig())


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
        math_cond="> 0.0",
        description="Random Color Brightness",
        automl_enabled="TRUE"
    )
    contrast: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        valid_min=0.0,
        valid_max=2.0,
        math_cond="> 0.0",
        description="Random Color Contrast",
        automl_enabled="TRUE"
    )
    saturation: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        valid_min=0.0,
        valid_max=2.0,
        math_cond="> 0.0",
        description="Random Color Saturation",
        automl_enabled="TRUE"
    )
    hue: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        valid_min=0.0,
        valid_max=0.5,
        math_cond="> 0.0",
        description="Random Color Hue",
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
class CNAugmentationSegmentConfig:
    """Augmentation config for segmentation."""

    random_flip: RandomFlip = DATACLASS_FIELD(RandomFlip())
    random_rotate: RandomRotation = DATACLASS_FIELD(RandomRotation())
    random_color: RandomColor = DATACLASS_FIELD(RandomColor())
    with_scale_random_crop: RandomCropWithScale = DATACLASS_FIELD(RandomCropWithScale())
    with_random_blur: bool = BOOL_FIELD(value=True, default_value=True, description="Flag to enable with_random_blur")
    with_random_crop: bool = BOOL_FIELD(value=True, default_value=True, description="Flag to enable with_random_crop")
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
class CNAugmentationClassifyConfig:
    """Augmentation config for classification."""

    rgb_input_mean: List[float] = LIST_FIELD(
        arrList=[0.485, 0.456, 0.406],
        default_value=[0.485, 0.456, 0.406],
        description="Mean for the augmentation",
        display_name="Mean"
    )
    rgb_input_std: List[float] = LIST_FIELD(
        arrList=[0.229, 0.224, 0.225],
        default_value=[0.226, 0.226, 0.226],
        description="Standard deviation for the augmentation",
        display_name="Standard Deviation"
    )
    random_flip: RandomFlip = DATACLASS_FIELD(RandomFlip())
    random_rotate: RandomRotation = DATACLASS_FIELD(RandomRotation())
    random_color: RandomColor = DATACLASS_FIELD(RandomColor())
    with_scale_random_crop: RandomCropWithScale = DATACLASS_FIELD(RandomCropWithScale())
    with_random_blur: bool = BOOL_FIELD(value=True, default_value=True, description="Flag to enable with_random_blur")
    with_random_crop: bool = BOOL_FIELD(value=True, default_value=True, description="Flag to enable with_random_crop")
    augment: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to enable augmentation",
        automl_enabled="TRUE"
    )


@dataclass
class DataPathFormat:
    """Dataset Path experiment config."""

    csv_path: str = STR_FIELD(value="", default_value="", description="Path to csv file for dataset")
    images_dir: str = STR_FIELD(value="", default_value="", description="Path to images directory for dataset")


@dataclass
class CNDatasetClassifyConfig:
    """Classification Dataset Config."""

    train_dataset: DataPathFormat = DATACLASS_FIELD(DataPathFormat())
    validation_dataset: DataPathFormat = DATACLASS_FIELD(DataPathFormat())
    test_dataset: DataPathFormat = DATACLASS_FIELD(DataPathFormat())
    infer_dataset: DataPathFormat = DATACLASS_FIELD(DataPathFormat())
    image_ext: Optional[str] = STR_FIELD(value=None, default_value=".jpg", description="Image extension")
    batch_size: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="Batch size",
        display_name="Batch Size",
    )
    workers: int = INT_FIELD(
        value=8,
        default_value=1,
        valid_min=0,
        valid_max="inf",
        description="Workers",
        display_name="Workers",
    )
    fpratio_sampling: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        valid_min=0.0,
        valid_max=1.0,
        description="Sampling ratio for minority class",
        automl_enabled="TRUE"
    )
    num_input: int = INT_FIELD(
        value=8,
        default_value=4,
        valid_min=1,
        valid_max="inf",
        description="Number of input lighting conditions"
    )
    input_map: Optional[Dict[str, int]] = DICT_FIELD(
        None,
        default_value={
            "LowAngleLight": 0,
            "SolderLight": 1,
            "UniformLight": 2,
            "WhiteLight": 3
        },
        description="input mapping"
    )
    grid_map: Optional[Dict[str, int]] = DICT_FIELD(
        None,
        default_value={"x": 2, "y": 2},
        description="grid map"
    )
    concat_type: Optional[str] = STR_FIELD(
        value=None,
        value_type="ordered",
        default_value="linear",
        valid_options="linear,grid",
        description="concat type"
    )
    image_width: int = INT_FIELD(
        value=224,
        default_value=224,
        description="Width of the input image tensor."
    )
    image_height: int = INT_FIELD(
        value=224,
        default_value=224,
        description="Height of the input image tensor."
    )
    augmentation_config: CNAugmentationClassifyConfig = DATACLASS_FIELD(CNAugmentationClassifyConfig())
    num_classes: int = INT_FIELD(
        value=2,
        default_value=2,
        description="The number of classes in the training data",
        math_cond=">0",
        valid_min=2,
        valid_max="inf"
    )
    num_golden: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        description="Number of golden samples for each input"
    )
    quant_calibration_dataset: QuantCalibrationDataset = DATACLASS_FIELD(
        QuantCalibrationDataset(),
        description="Configurable parameters for the quantization calibration dataset.",
    )


@dataclass
class CNDatasetSegmentConfig:
    """Segmentation Dataset Config."""

    root_dir: str = STR_FIELD(value="", default_value="", description="Path to root directory for dataset")
    label_transform: str = STR_FIELD(
        value="norm",
        default_value="norm",
        valid_options="norm,None",
        description="label transform"
    )
    data_name: str = STR_FIELD(
        value="LEVIR",
        default_value="LEVIR",
        valid_options="LEVIR,LandSCD,custom",
        description="dataset name"
    )
    dataset: str = STR_FIELD(
        value="CNDataset",
        default_value="CNDataset",
        valid_options="CNDataset",
        description="dataset class"
    )
    multi_scale_train: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Multi scale training",
        automl_enabled="TRUE"
    )
    multi_scale_infer: bool = BOOL_FIELD(value=False, default_value=False, description="Multi scale inference")
    num_classes: int = INT_FIELD(
        value=2,
        default_value=2,
        description="The number of classes in the training data",
        math_cond=">0",
        valid_min=2,
        valid_max="inf"
    )
    img_size: int = INT_FIELD(
        value=224,
        default_value=224,
        description="The input image size"
    )
    batch_size: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="Batch size",
        display_name="Batch Size",
    )
    workers: int = INT_FIELD(
        value=8,
        default_value=1,
        valid_min=0,
        valid_max="inf",
        description="Workers",
        display_name="Workers",
    )
    shuffle: bool = BOOL_FIELD(value=True, default_value=True, description="Shuffle dataloader")
    image_folder_name: str = STR_FIELD(value="A", default_value="A", description="image_folder_name")
    change_image_folder_name: str = STR_FIELD(value="B", default_value="B", description="change_image_folder_name")
    list_folder_name: str = STR_FIELD(value="list", default_value="list", description="list folder name")
    annotation_folder_name: str = STR_FIELD(value="label", default_value="label", description="label folder name")
    augmentation: CNAugmentationSegmentConfig = DATACLASS_FIELD(CNAugmentationSegmentConfig())
    train_split: str = STR_FIELD(value="train", default_value="train", description="Train split folder name")
    validation_split: str = STR_FIELD(value="val", default_value="val", description="Validation split folder name")
    test_split: str = STR_FIELD(value="test", default_value="test", description="Test split folder name")
    predict_split: str = STR_FIELD(value="test", default_value="test", description="Predict split folder name")
    label_suffix: str = STR_FIELD(value=".png", default_value=".png", description="Suffix of images")
    color_map: Optional[Dict[str, List[int]]] = DICT_FIELD(None, description="Class label index to RGB color mapping")
    quant_calibration_dataset: QuantCalibrationDataset = DATACLASS_FIELD(
        QuantCalibrationDataset(),
        description="Configurable parameters for the quantization calibration dataset.",
    )


@dataclass
class CNDatasetConfig:
    """Dataset config."""

    segment: CNDatasetSegmentConfig = DATACLASS_FIELD(CNDatasetSegmentConfig())
    classify: CNDatasetClassifyConfig = DATACLASS_FIELD(CNDatasetClassifyConfig())


@dataclass
class TensorBoardLogger:
    """Configuration for the tensorboard logger."""

    enabled: bool = BOOL_FIELD(value=False, default_value=False, description="Flag to enable tensorboard")
    infrequent_logging_frequency: int = INT_FIELD(
        value=2,
        default_value=2,
        valid_min=0,
        valid_max="inf",
        description="infrequent_logging_frequency"
    )  # Defined per epoch


@dataclass
class CNTrainClassifyConfig:
    """Classifier loss config."""

    loss: str = STR_FIELD(
        value="contrastive",
        default_value="contrastive",
        valid_options="ce,contrastive",
        description="ChangeNet Classify loss"
    )
    cls_weight: List[float] = LIST_FIELD(
        arrList=[1.0, 10.0],
        default_value=[1.0, 10.0],
        description="ChangeNet Classify ce loss class weight"
    )


@dataclass
class CNTrainSegmentConfig:
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
        description="ChangeNet Segment loss weight"
    )


@dataclass
class CNTrainExpConfig(TrainConfig):
    """Train Config."""

    optim: CNOptimConfig = DATACLASS_FIELD(CNOptimConfig())
    pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Pretrained model path",
        display_name="pretrained model path"
    )
    classify: CNTrainClassifyConfig = DATACLASS_FIELD(CNTrainClassifyConfig())
    segment: CNTrainSegmentConfig = DATACLASS_FIELD(CNTrainSegmentConfig())
    tensorboard: Optional[TensorBoardLogger] = DATACLASS_FIELD(TensorBoardLogger())
    precision: str = STR_FIELD(
        value='32-true',
        default_value='32-true',
        description="Precision",
        display_name="precision"
    )
    sync_batchnorm: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Synchronize batch normalization across devices"
    )
    use_distributed_sampler: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Use distributed sampler for multi-GPU training"
    )


@dataclass
class CNEvalExpConfig(EvaluateConfig):
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


@dataclass
class CNInferenceExpConfig(InferenceConfig):
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


@dataclass
class ExportExpConfig(ExportConfig):
    """Export experiment config."""

    serialize_nvdsinfer: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Serialize DeepStream config.",
        description=(
            "Flag to enable serializing the required configs for integrating with DeepStream."
        )
    )
    input_channel: int = INT_FIELD(
        value=3,
        default_value=3,
        description="Input channel",
        display_name="Input channel"
    )
    input_width: int = INT_FIELD(
        value=224,
        default_value=224,
        description="Input width",
        display_name="Input width",
    )
    input_height: int = INT_FIELD(
        value=224,
        default_value=224,
        description="Input height",
        display_name="Input height",
    )


@dataclass
class CNTrtConfig(TrtConfig):
    """Trt config."""

    max_batch_size: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        description=(
            "The maximum batch size in the optimization profile for "
            "the input tensor of the TensorRT engine."
        ),
        display_name="Maximum batch size"
    )
    data_type: str = STR_FIELD(
        value="fp32",
        default_value="fp32,fp16",
        description="Data type",
        display_name="Data type"
    )
    calibration: CalibrationConfig = DATACLASS_FIELD(CalibrationConfig())


@dataclass
class CNGenTrtEngineExpConfig(GenTrtEngineConfig):
    """Gen TRT Engine experiment config."""

    tensorrt: CNTrtConfig = DATACLASS_FIELD(CNTrtConfig())


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: CNModelConfig = DATACLASS_FIELD(CNModelConfig())
    dataset: CNDatasetConfig = DATACLASS_FIELD(CNDatasetConfig())
    train: CNTrainExpConfig = DATACLASS_FIELD(CNTrainExpConfig())
    evaluate: CNEvalExpConfig = DATACLASS_FIELD(CNEvalExpConfig())
    inference: CNInferenceExpConfig = DATACLASS_FIELD(CNInferenceExpConfig())
    export: ExportConfig = DATACLASS_FIELD(ExportExpConfig())
    gen_trt_engine: CNGenTrtEngineExpConfig = DATACLASS_FIELD(CNGenTrtEngineExpConfig())
    quantize: ModelQuantizationConfig = DATACLASS_FIELD(
        ModelQuantizationConfig(),
        description="Configurable parameters to run model quantization for a Visual ChangeNet experiment.",
    )
    task: Optional[str] = STR_FIELD(
        value="segment",
        default_value="segment",
        valid_options="segment,classify"
    )

    def __post_init__(self):
        """Set default model name for Visual ChangeNet."""
        if self.model_name is None:
            self.model_name = "visual_changenet"
