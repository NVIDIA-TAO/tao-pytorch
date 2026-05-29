# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RADIO Default config file"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from omegaconf import MISSING

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    DICT_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
)
from nvidia_tao_pytorch.config.common.common_config import (
    CommonExperimentConfig,
    ExportConfig,
    TrainConfig,
    EvaluateConfig,
    GenTrtEngineConfig,
    InferenceConfig,
    TrtConfig,
    CalibrationConfig,
)

from nvidia_tao_pytorch.config.common.distillation_config import DistillationConfig
from nvidia_tao_pytorch.config.common.quantization import ModelQuantizationConfig


@dataclass
class OptimConfig:
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
        valid_options="linear,step,cosine,multistep",
        description="Optimizer policy"
    )
    policy_params: Dict[str, Any] = DICT_FIELD(
        {"step_size": 30, "gamma": 0.1, "milestones": [10, 20]},
        default_value={"step_size": 30, "gamma": 0.1},
        description="Optimizer policy parameters"
    )
    momentum: float = FLOAT_FIELD(
        value=0.9,
        default_value=0.9,
        math_cond="> 0.0",
        display_name="momentum - AdamW",
        description="The momentum for the AdamW optimizer.",
        automl_enabled="TRUE"
    )
    weight_decay: float = FLOAT_FIELD(
        value=0.01,
        default_value=0.01,
        math_cond="> 0.0",
        display_name="weight decay",
        description="The weight decay coefficient.",
        automl_enabled="TRUE"
    )
    betas: Optional[List[float]] = LIST_FIELD(
        [0.9, 0.999],
        automl_enabled="TRUE",
        description="coefficients used for computing running averages on adamw"
    )
    skip_names: Optional[List[str]] = LIST_FIELD(
        [],
        description="layers names which do not need weight decay"
    )
    warmup_epochs: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max="inf",
        description="Warmup epochs."
    )


@dataclass
class BackboneConfig:
    """Configuration parameters for Backbone."""

    type: str = STR_FIELD(
        value="fan_small_12_p4_hybrid",
        default_value="fan_small_12_p4_hybrid",
        description="Backbone architure",
        display_name="Backbone architectures",
        valid_options=",".join([
            "faster_vit_0_224",
            "faster_vit_1_224",
            "faster_vit_2_224",
            "faster_vit_3_224",
            "faster_vit_4_224",
            "faster_vit_5_224",
            "faster_vit_6_224",
            "faster_vit_4_21k_224",
            "faster_vit_4_21k_384",
            "faster_vit_4_21k_512",
            "faster_vit_4_21k_768",
            "fan_tiny_12_p16_224",
            "fan_small_12_p16_224_se_attn",
            "fan_small_12_p16_224",
            "fan_base_18_p16_224",
            "fan_large_24_p16_224",
            "fan_tiny_8_p4_hybrid",
            "fan_small_12_p4_hybrid",
            "fan_base_16_p4_hybrid",
            "fan_large_16_p4_hybrid",
            "fan_xlarge_16_p4_hybrid",
            "fan_swin_tiny_patch4_window7_224",
            "fan_swin_small_patch4_window7_224",
            "fan_swin_base_patch4_window7_224",
            "fan_swin_large_patch4_window7_224",
            "vit_large_patch14_dinov2_swiglu",
            "vit_large_patch14_dinov2_swiglu_legacy",
            "vit_giant_patch14_reg4_dinov2_swiglu",
            "efficientvit_b0",
            "efficientvit_b1",
            "efficientvit_b2",
            "efficientvit_b3",
            "efficientvit_l0",
            "efficientvit_l1",
            "efficientvit_l2",
            "efficientvit_l3",
            "vit_base_patch16",
            "vit_large_patch16",
            "vit_huge_patch14",
            "convnext_tiny",
            "convnext_small",
            "convnext_base",
            "convnext_large",
            "convnext_xlarge",
            "convnextv2_atto",
            "convnextv2_femto",
            "convnextv2_pico",
            "convnextv2_nano",
            "convnextv2_tiny",
            "convnextv2_base",
            "convnextv2_large",
            "convnextv2_huge",
            "hiera_tiny_224",
            "hiera_small_224",
            "hiera_base_224",
            "hiera_base_plus_224",
            "hiera_large_224",
            "hiera_huge_224",
            "resnet_18",
            "resnet_34",
            "resnet_50",
            "resnet_101",
            "resnet_152",
            "resnet_18d",
            "resnet_34d",
            "resnet_50d",
            "resnet_101d",
            "resnet_152d",
            "swin_tiny_patch4_window7_224",
            "swin_small_patch4_window7_224",
            "swin_base_patch4_window7_224",
            "swin_large_patch4_window7_224",
            "swin_base_patch4_window12_384",
            "swin_large_patch4_window12_384",
            "gc_vit_xxtiny",
            "gc_vit_xtiny",
            "gc_vit_tiny",
            "gc_vit_small",
            "gc_vit_base",
            "gc_vit_large",
            "gc_vit_base_384",
            "gc_vit_large_384",
            "edgenext_xx_small",
            "edgenext_x_small",
            "edgenext_small",
            "edgenext_base",
            "edgenext_xx_small_bn_hs",
            "edgenext_x_small_bn_hs",
            "edgenext_small_bn_hs",
            "c_radio_p1_vit_huge_patch16_mlpnorm",
            "c_radio_p2_vit_huge_patch16_mlpnorm",
            "c_radio_p3_vit_huge_patch16_mlpnorm",
            "c_radio_v2_vit_base_patch16",
            "c_radio_v2_vit_large_patch16",
            "c_radio_v2_vit_huge_patch16",
            "c_radio_v3_vit_large_patch16_reg4_dinov2",
            "c_radio_v3_vit_base_patch16_reg4_dinov2",
            "c_radio_v3_vit_huge_patch16_reg4_dinov2",
            "c_radio_v4_vit_huge_patch16",
            "c_radio_v4_vit_so400m_patch16",
            "vit_l_14_siglip_clipa_224",
            "vit_l_14_siglip_clipa_336",
            "vit_h_14_siglip_clipa_224",
            "mit_b0",
            "mit_b1",
            "mit_b2",
            "mit_b3",
            "mit_b4",
            "mit_b5",
        ]),
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
    freeze_norm: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to freeze norm",
        automl_enabled="TRUE"
    )


@dataclass
class ModelConfig:
    """Model config."""

    backbone: BackboneConfig = DATACLASS_FIELD(BackboneConfig())


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
    angle_range: Optional[List[float]] = LIST_FIELD(
        arrList=[-15, 15],
        default_value=[-15, 15],
        description="Angle range (min, max) in degrees for continuous rotation"
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
        math_cond="> 0.0",
        description="Random Color Brightness",
        automl_enabled="TRUE"
    )
    contrast: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        math_cond="> 0.0",
        description="Random Color Contrast",
        automl_enabled="TRUE"
    )
    saturation: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        math_cond="> 0.0",
        description="Random Color Saturation",
        automl_enabled="TRUE"
    )
    hue: float = FLOAT_FIELD(
        value=0,
        default_value=0,
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
class RandomErase:
    """RandomErase augmentation config."""

    enable: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to enable Random Erase",
        automl_enabled="TRUE"
    )
    erase_probability: float = FLOAT_FIELD(
        value=0.2,
        default_value=0.2,
        valid_min=0,
        valid_max=1,
        description="Random Erase Probability",
        automl_enabled="TRUE"
    )


@dataclass
class RandomAug:
    """RandomAug augmentation config."""

    enable: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to enable Random Aug",
        automl_enabled="TRUE"
    )


@dataclass
class AugmentationConfig:
    """Augmentation config."""

    random_flip: RandomFlip = DATACLASS_FIELD(RandomFlip())
    random_rotate: RandomRotation = DATACLASS_FIELD(RandomRotation())
    random_color: RandomColor = DATACLASS_FIELD(RandomColor())
    random_erase: RandomErase = DATACLASS_FIELD(RandomErase())
    random_aug: RandomAug = DATACLASS_FIELD(RandomAug())
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
        arrList=[0.485, 0.456, 0.406],
        default_value=[0.485, 0.456, 0.406],
        description="Mean for the augmentation",
        display_name="Mean"
    )
    std: List[float] = LIST_FIELD(
        arrList=[0.229, 0.224, 0.225],
        default_value=[0.229, 0.224, 0.225],
        description="Std for the augmentation",
        display_name="Std"
    )
    multi_scales: List[Any] = LIST_FIELD(
        arrList=[{224: 0.1}, {256: 0.2}, {288: 0.3}, {320: 0.4}],
        default_value=[{224: 0.1}, {256: 0.2}, {288: 0.3}, {320: 0.4}],
        description="Multi scales for the augmentation",
        display_name="Multi scales"
    )
    patch_size: Optional[int] = INT_FIELD(
        value=0,
        default_value=0,
        description="ViT patch size for patch-aligned crops (0=disabled, e.g. 14, 16)"
    )
    use_continuous_rotation: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Use continuous angle rotation instead of discrete"
    )
    perspective_distortion: Optional[Dict[str, Any]] = DICT_FIELD(
        {},
        default_value={},
        description="Config for perspective transform: enable, scale, prob"
    )


@dataclass
class ImageTarRoot:
    """Image Tar Root config."""

    root_dir: str = STR_FIELD(
        value="",
        default_value="",
        description="Path to image tar root directory for dataset",
        display_name="image tar root directory"
    )
    samples_per_file: int = INT_FIELD(
        value=10000,
        default_value=10000,
        description="Number of samples per file",
        display_name="samples per file"
    )
    steps_per_epoch: int = INT_FIELD(
        value=10000,
        default_value=10000,
        description="Number of steps per epoch",
        display_name="steps per epoch"
    )
    scale_factor: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        description="Scale factor for the dataset",
        display_name="scale factor"
    )


@dataclass
class DataPathFormat:
    """Dataset Path experiment config."""

    images_dir: Optional[str] = STR_FIELD(
        value="",
        default_value="",
        description="Path to images directory for dataset",
        display_name="image directory"
    )
    tar_data_sources: List[ImageTarRoot] = LIST_FIELD(
        arrList=[],
        default_value=[],
        description="List of tar data sources",
        display_name="tar data sources"
    )
    seed: int = INT_FIELD(
        value=42,
        default_value=42,
        description="Random seed for data pipeline",
    )
    full_equivariance: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable full equivariance augmentation",
    )
    shift_equivariance: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable shift equivariance augmentation",
    )
    data_weight_mode: Optional[str] = STR_FIELD(
        value="inv_frequency",
        default_value="inv_frequency",
        description="Sample weighting mode (inv_frequency, None, etc.)",
    )
    prefetch: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Enable data prefetching in the pipeline",
    )
    native_resolution_filter: Optional[Dict[str, Any]] = DICT_FIELD(
        None,
        default_value=None,
        description="Optional native image resolution filter applied before resize/crop",
    )


@dataclass
class UnstructuredTrainData:
    """Train Data Dataclass"""

    folder_path: Optional[str] = STR_FIELD(
        value="", default_value="", description="Dataset directory path"
    )


@dataclass
class DatasetConfig:
    """Classification Dataset Config."""

    root_dir: str = STR_FIELD(
        value="",
        default_value="",
        description="Path to folder that contains classes.txt which indicate class name and train ID. \
        Can be optional then the mapping will be generated from pipeline."
    )
    num_classes: int = INT_FIELD(
        value=0,
        default_value=0,
        description="The number of classes in the training data",
        math_cond=">=0",
        valid_min=0,
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
        automl_enabled="TRUE"
    )
    workers: int = INT_FIELD(
        value=8,
        default_value=1,
        valid_min=0,
        valid_max="inf",
        description="Workers",
        display_name="Workers",
        automl_enabled="TRUE"
    )
    augmentation: AugmentationConfig = DATACLASS_FIELD(AugmentationConfig())
    train_dataset: DataPathFormat = DATACLASS_FIELD(
        DataPathFormat(),
        description="Configuration for the training dataset path",
        display_name="Training Dataset"
    )
    train_nolabel: UnstructuredTrainData = DATACLASS_FIELD(UnstructuredTrainData())
    val_dataset: DataPathFormat = DATACLASS_FIELD(
        DataPathFormat(),
        description="Configuration for the validation dataset path",
        display_name="Validation Dataset"
    )
    test_dataset: DataPathFormat = DATACLASS_FIELD(
        DataPathFormat(),
        description="Configuration for the testing dataset path",
        display_name="Testing Dataset"
    )
    quant_calibration_dataset: DataPathFormat = DATACLASS_FIELD(
        DataPathFormat(),
        description="Configuration for the quantization calibration dataset path",
        display_name="Quantization Calibration Dataset"
    )
    val_img_size: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        description="Image size for validation. Defaults to img_size when not set.",
        display_name="Validation Image Size"
    )
    val_batch_size: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        description="Batch size for validation. Defaults to batch_size when not set.",
        display_name="Validation Batch Size"
    )
    knn_validation: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Enable KNN Top-1 validation. Requires val_dataset to have a sibling 'train' split available.",
        display_name="KNN Validation",
    )
    knn_num_classes: int = INT_FIELD(
        value=1000,
        default_value=1000,
        description="Number of classes for KNN Top-1 evaluation. Defaults to 1000 (ImageNet).",
        display_name="KNN Number of Classes",
    )
    knn_max_train_batches: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        description=(
            "Cap on train-split batches when building the KNN index. None means use the full train split. "
            "Set to a small value (e.g. 50) for fast smoke testing."
        ),
        display_name="KNN Max Train Batches",
    )


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
class TrainExpConfig(TrainConfig):
    """Train Config."""

    optim: OptimConfig = DATACLASS_FIELD(OptimConfig())
    pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Pretrained model path",
        display_name="pretrained model path"
    )
    tensorboard: Optional[TensorBoardLogger] = DATACLASS_FIELD(TensorBoardLogger())
    enable_ema: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to enable EMA"
    )
    ema_decay: float = FLOAT_FIELD(
        value=0.998,
        default_value=0.998,
        display_name="EMA decay",
        description="EMA decay"
    )
    clip_grad_norm: float = FLOAT_FIELD(
        value=2.0,
        default_value=2.0,
        display_name="Grad norm",
        description="Gradient Norm"
    )
    precision: str = STR_FIELD(
        value="fp32",
        default_value="fp32",
        description="Precision",
        valid_options="fp16, bf16, fp32"
    )


@dataclass
class EvalExpConfig(EvaluateConfig):
    """Evaluation experiment config."""

    vis_after_n_batches: int = INT_FIELD(
        value=16,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        description="Visualize evaluation segmentation results after n batches"
    )
    checkpoint: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="Path to checkpoint file",
        display_name="Path to checkpoint file"
    )
    is_quantized: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to indicate if the model is quantized",
        display_name="Flag to indicate if the model is quantized"
    )


@dataclass
class VitDetConfig:
    """ViTDet windowed-attention augmentation config (applied to the student).

    During training, with probability ``prob``, a random window size from
    ``window_sizes`` is selected and self-attention in the student ViT
    alternates between local (windowed) and global layers. This acts as
    a regularizer and can reduce memory during training.
    """

    prob: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max=1.0,
        description="Probability of activating windowed attention per forward pass. 0 disables ViTDet."
    )
    window_sizes: List[int] = LIST_FIELD(
        arrList=[],
        default_value=[],
        description=(
            "Candidate window sizes in patches (e.g. [6, 7, 8, 9, 12, 16]). "
            "One is randomly chosen per forward pass."
        )
    )
    num_global: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        description="Number of global-attention layers. Defaults to 4 if neither num_global nor num_windowed is set."
    )
    num_windowed: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        description="Number of windowed layers between each global layer. Alternative to num_global."
    )


@dataclass
class TeacherConfig:
    """Configuration for a single teacher model with distillation parameters."""

    model: ModelConfig = DATACLASS_FIELD(
        ModelConfig(),
        description="Configuration hyper parameters for the teacher model.",
        display_name="model"
    )
    loss_type: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="Teacher-specific distillation loss type",
        valid_options="""
        KL (KL divergence),
        CE (cross entropy),
        L1 (L1 loss),
        L2 (L2 loss),
        FD (smooth L1),
        CS (cosine similarity),
        BALANCED (balanced feature loss),
        MSE (mean squared error)""",
        description="Loss function for this teacher's logits distillation. If None, uses global loss_type."
    )
    loss_lambda: Optional[float] = FLOAT_FIELD(
        value=None,
        default_value=None,
        math_cond="> 0.0 <= 1.0",
        display_name="Teacher-specific distillation weight",
        description="Weight for this teacher's distillation loss. If None, uses global loss_lambda.",
    )
    pretrained_teacher_model_path: Optional[str] = STR_FIELD(
        value="",
        display_name="Pretrained teacher model path",
        description="Path to the pre-trained teacher model."
    )
    mode: str = STR_FIELD(
        value="auto",
        default_value="auto",
        description="Distillation mode",
        valid_options="logits, summary, spatial, auto, combo"
    )
    stochastic_resolutions: Optional[Dict[int, float]] = DICT_FIELD(
        {},
        default_value={},
        description="Per-sample stochastic resolutions for input resizing. Keys=resolutions, values=probabilities."
    )
    input_size: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        description="Input size for the teacher model"
    )
    patch_size: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        description="Patch size for the teacher ViT model"
    )
    upsample_factor: int = INT_FIELD(
        value=1,
        default_value=1,
        description="Upsample factor for teacher spatial features"
    )
    match_student_resolution: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to match the student resolution"
    )
    norm_mean: Optional[List[float]] = LIST_FIELD(
        [],
        default_value=[],
        description=(
            "Per-teacher image normalization mean (3 values, e.g. [0.485, 0.456, 0.406] for "
            "ImageNet/DINOv3, [0.5, 0.5, 0.5] for SAM3/SigLIP2). "
            "If empty, dataset augmentation mean is used."
        )
    )
    norm_std: Optional[List[float]] = LIST_FIELD(
        [],
        default_value=[],
        description=(
            "Per-teacher image normalization std (3 values, e.g. [0.229, 0.224, 0.225] for "
            "ImageNet/DINOv3, [0.5, 0.5, 0.5] for SAM3/SigLIP2). "
            "If empty, dataset augmentation std is used."
        )
    )
    summary_loss_weight: Optional[float] = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        math_cond=">= 0.0",
        display_name="Summary (CLS) loss weight for combo mode",
        description=(
            "Weight for summary/CLS token loss when mode is combo. "
            "Applied as summary_loss_weight * loss_summary."
        )
    )
    summary_loss_type: Optional[str] = STR_FIELD(
        value="CE",
        default_value="CE",
        display_name="Summary (CLS) loss type for combo mode",
        valid_options="CE, angle, cosine, tangent_sphere"
    )
    summary_token_idx: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        display_name="RADIO summary token index",
        description=(
            "Optional per-teacher RADIO summary-token slot. If unset and the student "
            "checkpoint exposes upstream teacher token slots, the distiller infers it."
        )
    )
    fd_loss_weight: Optional[float] = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        math_cond=">= 0.0",
        display_name="Feature distillation (spatial) loss weight for combo mode",
        description=(
            "Weight for spatial/feature distillation loss when mode is combo. "
            "Applied as fd_loss_weight * loss_spatial."
        )
    )
    spatial_mlp_version: str = STR_FIELD(
        value="v2",
        default_value="v2",
        display_name="Spatial projection head type",
        valid_options="v2, attn",
        description=(
            "Projection head for spatial distillation. 'attn' matches the "
            "attention-based C-RADIO v4 feature-projection heads."
        )
    )
    spatial_num_inner: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        display_name="Spatial projection inner blocks",
        description=(
            "Optional override for the number of inner MLP blocks in the spatial "
            "projection head. If unset, the distiller chooses a version-specific default."
        )
    )
    adaptor: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="Teacher adaptor type",
        valid_options="featsharp",
        description="Adaptor to apply to teacher features. 'featsharp' wraps the teacher "
        "with a pre-trained FeatSharp upsampler to produce high-resolution spatial targets."
    )
    upsampler_checkpoint: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="FeatSharp checkpoint path",
        description="Path to a pre-trained FeatSharp checkpoint for this teacher. "
        "Required when adaptor='featsharp'."
    )
    do_upsample: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="Enable FeatSharp upsampling",
        description="If True, load the learned FeatSharp upsampler. If False, only apply "
        "the normalizer/bias from the checkpoint (identity upsampling)."
    )
    featsharp_lib_path: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="FeatSharp library path",
        description="Path to the directory containing the 'featsharp' package "
        "(e.g. '.../evfm/libs/FeatUp'). Only needed if featsharp is not installed as a package."
    )


@dataclass
class ClassDistillationConfig(DistillationConfig):
    """Distillation config for classifier."""

    teacher: List[TeacherConfig] = DATACLASS_FIELD(
        MISSING,
        description=(
            "Configuration hyper parameters for the teacher model(s). "
            "Can be a single ModelConfig/TeacherConfig or a list for multiple teachers."
        ),
        display_name="teacher"
    )
    vitdet: Optional[VitDetConfig] = DATACLASS_FIELD(
        None,
        description="ViTDet windowed-attention augmentation applied to the student during training. "
        "Set prob > 0 and provide window_sizes to enable. Takes precedence over per-teacher vitdet fields.",
        display_name="ViTDet Augmentation"
    )
    loss_type: str = STR_FIELD(
        value="KL",
        default_value="KL",
        display_name="Distillation loss type",
        valid_options="""
        KL (KL divergence),
        CE (cross entropy),
        L1 (L1 loss),
        L2 (L2 loss),
        FD (smooth L1),
        CS (cosine similarity),
        BALANCED (balanced feature loss),
        MSE (mean squared error)""",
        description="Loss function for logits distillation."
    )
    loss_lambda: Optional[float] = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        math_cond="> 0.0 <= 1.0",
        display_name="distill weight",
        description="The weight to be applied to the distillation loss as compared to task loss",
    )
    pretrained_teacher_model_path: Optional[str] = STR_FIELD(
        value="",
        display_name="Pretrained teacher model path",
        description="Path to the pre-trained teacher model."
    )
    mode: str = STR_FIELD(
        value="auto",
        default_value="auto",
        description="Distillation mode",
        valid_options="logits, summary, spatial, auto, combo"
    )
    use_mlp: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to use MLP for projection"
    )
    mlp_hidden_size: int = INT_FIELD(
        value=1024,
        default_value=1024,
        valid_min=0,
        valid_max="inf",
        description="MLP hidden size"
    )
    mlp_num_inner: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max=10,
        description="MLP number of inner layers"
    )


@dataclass
class InferenceExpConfig(InferenceConfig):
    """Inference experiment config."""

    vis_after_n_batches: int = INT_FIELD(
        value=16,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        description="Visualize evaluation segmentation results after n batches"
    )
    checkpoint: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="Path to checkpoint file",
        display_name="Path to checkpoint file"
    )
    is_quantized: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to indicate if the model is quantized",
        display_name="Flag to indicate if the model is quantized"
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
    is_quantized: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to indicate if the model is quantized",
        display_name="Flag to indicate if the model is quantized"
    )


@dataclass
class TrtExpConfig(TrtConfig):
    """Trt config."""

    data_type: str = STR_FIELD(
        value="FP32",
        default_value="fp16",
        description="Data type",
        display_name="Data type"
    )
    calibration: CalibrationConfig = DATACLASS_FIELD(CalibrationConfig())


@dataclass
class GenTrtEngineExpConfig(GenTrtEngineConfig):
    """Gen TRT Engine experiment config."""

    tensorrt: TrtExpConfig = DATACLASS_FIELD(TrtExpConfig())


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: ModelConfig = DATACLASS_FIELD(ModelConfig())
    dataset: DatasetConfig = DATACLASS_FIELD(DatasetConfig())
    train: TrainExpConfig = DATACLASS_FIELD(TrainExpConfig())
    evaluate: EvalExpConfig = DATACLASS_FIELD(EvalExpConfig())
    inference: InferenceExpConfig = DATACLASS_FIELD(InferenceExpConfig())
    export: ExportExpConfig = DATACLASS_FIELD(ExportExpConfig())
    gen_trt_engine: GenTrtEngineExpConfig = DATACLASS_FIELD(GenTrtEngineExpConfig())
    distill: ClassDistillationConfig = DATACLASS_FIELD(ClassDistillationConfig())
    quantize: ModelQuantizationConfig = DATACLASS_FIELD(ModelQuantizationConfig())

    def __post_init__(self):
        """Set default model name for RADIO."""
        if self.model_name is None:
            self.model_name = "radio"
