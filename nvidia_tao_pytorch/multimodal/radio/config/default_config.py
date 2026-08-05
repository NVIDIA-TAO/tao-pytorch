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
        valid_options="adamw,adam,sgd,lamb,fusedlamb"
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
    sched_on_updates: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description=(
            "Step the LR schedule per optimizer update instead of per epoch. "
            "This makes schedule granularity independent of the selected policy."
        )
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
    input_dim: List[int] = LIST_FIELD(
        arrList=[3, 224, 224],
        default_value=[3, 224, 224],
        description="Input C,H,W for RADIO student backbones."
    )
    output_dim: Optional[List[int]] = LIST_FIELD(
        arrList=[],
        default_value=[],
        description="Optional output C,H,W for RADIO student backbones. Empty derives from the backbone stride/patch size."
    )
    feature_dim: int = INT_FIELD(
        value=256,
        default_value=256,
        description="RADIO feature dimension for student backbones."
    )
    student_norm_mean: List[float] = LIST_FIELD(
        arrList=[],
        default_value=[],
        description="Optional student input normalization mean for ViT RADIO student backbones. Empty uses the backbone default."
    )
    student_norm_std: List[float] = LIST_FIELD(
        arrList=[],
        default_value=[],
        description="Optional student input normalization std for ViT RADIO student backbones. Empty uses the backbone default."
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
        description="The input image size (square crops)."
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
        description="Image size for validation (square crops). Defaults to img_size when not set.",
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
    warmstart_training_checkpoint_path: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description=(
            "Load student, projection-head, and distillation-statistics weights from a "
            "training checkpoint without restoring optimizer, scheduler, epoch, or data state"
        ),
        display_name="weights-only training checkpoint"
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
    checkpoint_keep_last_n: int = INT_FIELD(
        value=-1,
        default_value=-1,
        valid_min=-1,
        valid_max="inf",
        display_name="Rolling checkpoint count",
        description=(
            "Max number of checkpoints to keep on disk (maps to ModelCheckpoint.save_top_k). "
            "-1 keeps every checkpoint (legacy behaviour); a positive N keeps only N. Which N "
            "are kept depends on checkpoint_monitor. The _latest symlink is always maintained "
            "for resume."
        )
    )
    checkpoint_keep_milestone_every: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max="inf",
        display_name="Milestone checkpoint interval",
        description=(
            "Independently of checkpoint_keep_last_n's rolling window, also keep a permanent "
            "checkpoint every N epochs (0 disables). These are saved under <results_dir>/milestones "
            "with save_top_k=-1 so they are never pruned -- useful for probing/eval at fixed "
            "epochs on a long run without racing the rolling-window deletion."
        )
    )
    checkpoint_monitor: str = STR_FIELD(
        value="",
        default_value="",
        display_name="Checkpoint selection metric",
        description=(
            "Metric used to rank checkpoints when checkpoint_keep_last_n > 0. Empty string keeps "
            "the N most recent checkpoints (ranks by epoch). Set to a logged metric to keep the "
            "best N instead, e.g. 'total_loss_epoch' / 'distillation_loss_epoch' (lowest training "
            "loss), or a validation metric such as 'val_distillation_loss' or 'val_spatial_cka'. "
            "When monitoring a validation metric, ensure validation runs at the checkpoint cadence."
        )
    )
    checkpoint_monitor_mode: str = STR_FIELD(
        value="min",
        default_value="min",
        display_name="Checkpoint selection mode",
        valid_options="min, max",
        description=(
            "Whether lower ('min') or higher ('max') checkpoint_monitor values are better. "
            "Use 'min' for losses and 'max' for metrics like val_spatial_cka."
        )
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
    input_size: Any = INT_FIELD(
        value=None,
        default_value=None,
        description="Input size for the teacher model. Use an int for square views or [height, width] for rectangular views."
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
    student_resolution: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        description=(
            "Square student resolution at which THIS teacher supervises the student "
            "and routes this teacher's loss to it. In partitioned training, each rank "
            "runs only its partition's resolution. If unset, the teacher uses "
            "dataset.img_size."
        )
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
    spatial_loss_type: Optional[str] = STR_FIELD(
        value="mse",
        default_value="mse",
        display_name="Spatial feature loss type for combo/spatial mode",
        valid_options="mse, dampened_mse, balanced, cosine, gram, channel_kl",
        description=(
            "Spatial feature loss for feature-map distillation. "
            "balanced uses mostly cosine alignment with a small SmoothL1 term."
        )
    )
    spatial_focal_weight: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max=1.0,
        display_name="Spatial focal weighting strength",
        description=(
            "Optional teacher-saliency weighting for spatial feature loss. "
            "0 disables focal weighting; 1 uses only normalized teacher-saliency weights."
        )
    )
    spatial_focal_gamma: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        valid_min=0.0,
        valid_max="inf",
        display_name="Spatial focal gamma",
        description="Exponent applied to normalized teacher spatial saliency before loss weighting."
    )
    spatial_focal_max_weight: float = FLOAT_FIELD(
        value=4.0,
        default_value=4.0,
        valid_min=0.0,
        valid_max="inf",
        display_name="Spatial focal max weight",
        description=(
            "Optional clamp for per-token spatial focal weights before re-normalization. "
            "Set 0 to disable clamping."
        )
    )
    intermediate_loss_weight: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max="inf",
        display_name="Intermediate spatial loss weight",
        description=(
            "Global weight for optional lower-resolution student feature-map supervision "
            "in combo distillation mode."
        )
    )
    intermediate_loss_weights: List[float] = LIST_FIELD(
        [],
        default_value=[],
        description=(
            "Relative per-level weights for intermediate spatial losses. "
            "If empty, levels are weighted uniformly and normalized to sum to 1."
        )
    )
    intermediate_feature_dims: List[int] = LIST_FIELD(
        [],
        default_value=[],
        description=(
            "Student channel dimensions for intermediate spatial projection heads, "
            "one per intermediate feature map (highest-resolution first)."
        )
    )
    intermediate_focal_weight: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max=1.0,
        display_name="Intermediate spatial focal weighting strength",
        description="Teacher-saliency focal weighting strength for intermediate spatial losses."
    )
    intermediate_mlp_version: str = STR_FIELD(
        value="residual",
        default_value="residual",
        display_name="Intermediate spatial projection head type",
        valid_options="v2, attn, residual",
        description="Projection head type for lower-resolution intermediate spatial maps."
    )
    intermediate_num_inner: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        display_name="Intermediate projection inner blocks",
        description="Optional override for the number of inner blocks in intermediate projectors."
    )
    spatial_norm_type: str = STR_FIELD(
        value="phi",
        default_value="phi",
        display_name="Spatial target normalization",
        valid_options="phi, zca, pca",
        description=(
            "Teacher feature normalization for spatial distillation. 'phi' preserves "
            "legacy PHI standardization; 'zca'/'pca' use true covariance whitening."
        )
    )
    spatial_whiten_update_period: int = INT_FIELD(
        value=100,
        default_value=100,
        valid_min=1,
        valid_max="inf",
        display_name="Spatial whitening update period",
        description="Number of forward passes between whitening projection updates."
    )
    spatial_whiten_freeze_after_steps: int = INT_FIELD(
        value=3000,
        default_value=3000,
        valid_min=0,
        valid_max="inf",
        display_name="Spatial whitening freeze step",
        description=(
            "Stop updating whitening statistics after this many teacher forwards. "
            "Set 0 to use the legacy update horizon."
        )
    )
    spatial_whiten_shrinkage: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max=1.0,
        display_name="Spatial whitening shrinkage",
        description="Shrink covariance toward isotropic covariance before whitening."
    )
    spatial_whiten_eigen_floor: float = FLOAT_FIELD(
        value=1.0e-6,
        default_value=1.0e-6,
        valid_min=0.0,
        valid_max="inf",
        display_name="Spatial whitening eigen floor",
        description="Relative eigenvalue floor used to limit whitening amplification."
    )
    spatial_whiten_max_gain: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max="inf",
        display_name="Spatial whitening max gain",
        description="Maximum whitening gain. Set 0 to disable explicit gain clipping."
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
        valid_options="v2, attn, residual",
        description=(
            "Projection head for spatial distillation. 'attn' matches the "
            "attention-based C-RADIO v4 feature-projection heads; 'residual' "
            "uses a constrained residual MLP on top of a linear lift."
        )
    )
    spatial_projector_residual_scale: float = FLOAT_FIELD(
        value=0.25,
        default_value=0.25,
        valid_min=0.0,
        valid_max="inf",
        display_name="Residual projector scale",
        description="Scale applied to the nonlinear residual branch of the residual spatial projector."
    )
    spatial_projector_output_norm: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Residual projector output norm",
        description="Apply LayerNorm to residual projector output before spatial normalization."
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
    summary_mlp_version: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="Summary projection head type",
        valid_options="v2, attn, residual",
        description=(
            "Projection head for the combo-mode summary projector. Defaults to "
            "spatial_mlp_version when unset."
        )
    )
    summary_num_inner: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        display_name="Summary projection inner blocks",
        description=(
            "Optional override for the number of inner MLP blocks in the summary "
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
        "(for example, a local FeatUp checkout). Only needed if featsharp is not installed."
    )
    shared_teacher_key: str = STR_FIELD(
        value="",
        default_value="",
        description=(
            "Stable teacher identity. Teacher arms with the same key share "
            "projection heads and normalization state."
        )
    )
    rank_partition: int = INT_FIELD(
        value=-1,
        default_value=-1,
        description="Contiguous rank-partition index assigned to this teacher arm."
    )
    local_batch_size: int = INT_FIELD(
        value=0,
        default_value=0,
        description="Per-rank batch size for this teacher arm's rank partition."
    )
    mosaic_inner_size: int = INT_FIELD(
        value=0,
        default_value=0,
        description=(
            "Inner teacher canvas size for mosaic batching. Zero disables mosaic batching."
        )
    )
    mosaic_outer_size: int = INT_FIELD(
        value=0,
        default_value=0,
        description="Outer per-example view size packed into the teacher mosaic."
    )
    mosaic_downsample: int = INT_FIELD(
        value=0,
        default_value=0,
        description="Teacher feature stride used to unpack mosaic targets."
    )


@dataclass
class ClassDistillationConfig(DistillationConfig):
    """Distillation config for classifier."""

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="Distillation results directory",
        description="Directory where distillation outputs are written."
    )
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
    partitioned_ranks: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description=(
            "Enable native partitioned distillation. Each contiguous rank partition "
            "uses its configured local batch, student resolution, and teacher views; "
            "teacher arms may share projection and normalization state."
        )
    )
    num_rank_partitions: int = INT_FIELD(
        value=4,
        default_value=4,
        math_cond=">= 2",
        description="Number of contiguous rank partitions."
    )
    rebalance_teacher_loss: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description=(
            "Rebalance each teacher loss by world_size / teacher_world_size so "
            "DDP averaging preserves each teacher's objective under partitioning."
        )
    )
    sync_bn_mode: str = STR_FIELD(
        value="global",
        default_value="global",
        valid_options="off, global",
        description=(
            "BatchNorm synchronization for non-partitioned training. Partitioned "
            "training always keeps running statistics rank-local until epoch reduction."
        )
    )
    broadcast_buffers: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description=(
            "DDP buffer broadcast for non-partitioned training. Partitioned training "
            "always disables rank-0 buffer broadcast."
        )
    )
    dist_bn_mode: str = STR_FIELD(
        value="off",
        default_value="off",
        valid_options="off, broadcast, reduce",
        description=(
            "Distribute student BatchNorm running_mean/running_var at each train epoch end. "
            "Partitioned training always uses reduction."
        )
    )
    teacher_bf16: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description=(
            "Cast frozen teachers to bfloat16 to halve resident weight memory. Needed to fit heavy "
            "high-res teacher sets (e.g. DINOv3-7B ~28GB fp32 -> ~14GB bf16) in 80GB under "
            "partitioned_ranks. Teachers are inference-only distillation targets, so bf16 is "
            "numerically safe (training precision is bf16 regardless)."
        )
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
    train_projection_heads: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description=(
            "Include the trainable distillation projection heads in the optimizer. "
            "Default true preserves the established TAO behavior where the projection "
            "heads (which map student features into each teacher's space) are optimized "
            "alongside the student. Set false to optimize only the student parameters."
        )
    )
    projector_lr: Optional[float] = FLOAT_FIELD(
        value=None,
        default_value=None,
        valid_min=0.0,
        valid_max="inf",
        description=(
            "Optional learning rate for distillation projection heads. If unset, "
            "projection heads use the main optimizer learning rate."
        )
    )
    spectral_projection_heads: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description=(
            "Apply spectral normalization to Linear layers inside distillation "
            "projection heads. This bounds projector gain while leaving the "
            "student backbone unchanged."
        )
    )
    spectral_reparam_backbone: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description=(
            "Apply learnable-scale spectral reparametrization to every "
            "eligible Linear in the STUDENT BACKBONE (stage-4 attention qkv/proj + "
            "MLP Linears). Uses the same power_iterations/eps/alpha as the "
            "projection-head spectral settings. "
            "Distinct from spectral_projection_heads, which only touches the heads."
        )
    )
    spectral_projection_heads_power_iterations: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=1,
        valid_max="inf",
        description="Number of power iterations for spectral-normalized projection-head Linear layers."
    )
    spectral_projection_heads_eps: float = FLOAT_FIELD(
        value=1.0e-12,
        default_value=1.0e-12,
        valid_min=0.0,
        valid_max="inf",
        description="Numerical epsilon for spectral-normalized projection-head Linear layers."
    )
    spectral_projection_heads_learnable_scale: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description=(
            "Use learnable-scale spectral reparametrization "
            "(weight * (softplus(scale) + alpha) / sigma, distillation/spectral_reparam.py) "
            "instead of stock torch.nn.utils.parametrizations.spectral_norm for "
            "projection-head Linear layers. Only takes effect when "
            "spectral_projection_heads is True."
        )
    )
    spectral_projection_heads_alpha: float = FLOAT_FIELD(
        value=0.05,
        default_value=0.05,
        valid_min=0.0,
        valid_max="inf",
        description=(
            "Softplus offset 'alpha' for the learnable-scale spectral reparam "
            "used when spectral_projection_heads_learnable_scale is True."
        )
    )
    freeze_projection_heads_after_warmup: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description=(
            "After head_warmup_epochs, freeze distillation projection heads and "
            "train only the student backbone."
        )
    )
    warmstart_projection_heads: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description=(
            "Warm-start distillation projection heads from compatible upstream "
            "checkpoint heads when available."
        )
    )
    head_warmup_epochs: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max="inf",
        description=(
            "Number of initial epochs to freeze the student backbone while "
            "training only the distillation projection heads."
        )
    )
    freeze_student_norms: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description=(
            "Freeze student normalization layers during distillation. This keeps "
            "BatchNorm/SyncBatchNorm/InstanceNorm modules in eval mode and freezes "
            "normalization affine parameters, including LayerNorm and GroupNorm."
        )
    )
    freeze_distillation_statistics: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description=(
            "Freeze PHI/whitening and summary-loss running statistics after checkpoint "
            "initialization. Use this for continuation runs with calibrated statistics."
        )
    )
    baseline_anchor_model_path: Optional[str] = STR_FIELD(
        value="",
        default_value="",
        description=(
            "Optional checkpoint path for the frozen student baseline anchor. If "
            "empty and any baseline anchor weight is nonzero, the checkpoint-loaded "
            "student at initialization is deep-copied and used as the anchor."
        )
    )
    baseline_anchor_loss_type: str = STR_FIELD(
        value="normalized_mse",
        default_value="normalized_mse",
        valid_options="normalized_mse, mse, cosine",
        description=(
            "Loss used for baseline anchor preservation. normalized_mse compares "
            "L2-normalized features and is scale-tolerant; mse compares raw values; "
            "cosine uses 1 - cosine similarity."
        )
    )
    baseline_anchor_spatial_loss_weight: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max="inf",
        description="Weight for preserving the frozen baseline student's final spatial feature map."
    )
    baseline_anchor_summary_loss_weight: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max="inf",
        description="Weight for preserving the frozen baseline student's summary feature."
    )
    baseline_anchor_intermediate_loss_weight: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max="inf",
        description=(
            "Total weight for preserving selected intermediate student features. "
            "The configured layer losses are averaged before this weight is applied."
        )
    )
    baseline_anchor_intermediate_layers: List[str] = LIST_FIELD(
        [],
        default_value=[],
        description=(
            "Student module names to anchor with forward hooks, e.g. "
            "['_model.levels.1', '_model.levels.2']."
        )
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
