# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

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
    EvaluateConfig,
    InferenceConfig,
    GenTrtEngineConfig,
    TrtConfig
)


@dataclass
class NormConfig:
    """Configuration parameters for Normalization Preprocessing."""

    type: str = STR_FIELD(value="SyncBN", description="Type of normalization to use", valid_options="BN,SyncBN")
    requires_grad: bool = BOOL_FIELD(value=True, description="Bool whether to train the gamma beta parameters of BN")


@dataclass
class TestModelConfig:
    """Configuration parameters for Inference."""

    mode: str = STR_FIELD(value="whole", description="Mode of Inference", valid_options="whole")
    # crop_size: Optional[List[int]] = None  # Configurable
    # stride: Optional[List[int]] = None  # Configurable


@dataclass
class LossDecodeConfig:
    """Configuration parameters for Loss."""

    type: str = STR_FIELD(value="CrossEntropyLoss", description="Type of loss to use")
    use_sigmoid: bool = BOOL_FIELD(value=False, description="Bool whether to use sigmoid")
    loss_weight: float = FLOAT_FIELD(
        value=1.0,
        description="Loss weight",
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="TRUE"
    )


@dataclass
class SegformerHeadConfig:
    """Configuration parameters for Segformer Head."""

    type: str = STR_FIELD(value="TAOSegFormerHead", default_value="TAOSegFormerHead", description="Type of head to use")
    in_channels: List[int] = LIST_FIELD(
        arrList=[64, 128, 320, 512],
        description="Input channels for the head",
        display_name="Input Channels",
        default_value=[64, 128, 320, 512]
    )
    in_index: List[int] = LIST_FIELD(
        arrList=[0, 1, 2, 3],
        description="Input index for the head",
        display_name="Input Index",
        default_value=[0, 1, 2, 3]
    )
    feature_strides: List[int] = LIST_FIELD(
        arrList=[4, 8, 16, 32],
        description="Feature strides for the head",
        display_name="Feature Strides",
        default_value=[4, 8, 16, 32]
    )
    channels: int = INT_FIELD(
        value=128,
        description="Channels for the head",
        display_name="Channels",
        default_value=128
    )
    dropout_ratio: float = FLOAT_FIELD(
        value=0.1,
        description="Dropout ratio for the head",
        display_name="Dropout Ratio",
        default_value=0.1,
        valid_min=0.0,
        valid_max=1.0,
        automl_enabled="TRUE"
    )
    num_classes: int = INT_FIELD(
        value=2,
        description="Number of classes for the head",
        display_name="Number of Classes",
        default_value=19
    )
    norm_cfg: NormConfig = DATACLASS_FIELD(NormConfig())
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
    )
    loss_decode: LossDecodeConfig = DATACLASS_FIELD(LossDecodeConfig())  # Non-configurable since there is only one loss


@dataclass
class BackboneConfig:
    """Configuration parameters for Backbone."""

    type: str = STR_FIELD(
        value="mit_b1",
        description="Type of backbone to use",
        valid_options=",".join([
            "mit_b0", "mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5",
            "fan_tiny_8_p4_hybrid", "fan_large_16_p4_hybrid",
            "fan_small_12_p4_hybrid", "fan_base_16_p4_hybrid",
            "vit_large_nvdinov2", "vit_giant_nvdinov2",
            "vit_base_nvclip_16_siglip", "vit_huge_nvclip_14_siglip"
        ])
    )
    init_cfg: Dict[str, Any] = DICT_FIELD(
        hashMap={"type": "Pretrained", "checkpoint": None},
        description="Model backbone initialisation config"
    )


@dataclass
class SFModelConfig:
    """SF model config."""

    type: str = STR_FIELD(value="EncoderDecoder", description="Type of model to use", valid_options="EncoderDecoder")
    pretrained_model_path: Optional[str] = STR_FIELD(
        value="",
        default_type="",
        description="Path to the pretrained model"
    )
    backbone: BackboneConfig = DATACLASS_FIELD(BackboneConfig())
    decode_head: SegformerHeadConfig = DATACLASS_FIELD(SegformerHeadConfig())
    # @seanf setting this triggers an error in mmseg, report this bug
    # Basically this one should be generalized https://github.com/open-mmlab/mmsegmentation/issues/3011
    test_cfg: TestModelConfig = DATACLASS_FIELD(TestModelConfig())
    input_width: int = INT_FIELD(value=512, description="Input width", valid_min=0)
    input_height: int = INT_FIELD(value=512, description="Input height", valid_min=0)


# Use the field parameter in order to define as dictionaries
@dataclass
class RandomCropCfg:
    """Configuration parameters for Random Crop Aug."""

    crop_size: List[int] = LIST_FIELD(
        arrList=[512, 512],
        description="Crop size for the augmentation",
        display_name="Crop Size",
        default_value=[512, 512]
    )
    cat_max_ratio: float = FLOAT_FIELD(
        value=0.75,
        description="Max ratio for the augmentation",
        display_name="Max Ratio",
        default_value=0.75,
        valid_min=0.0,
        valid_max=1.0
    )


@dataclass
class ResizeCfg:
    """Configuration parameters for Resize Preprocessing."""

    img_scale: Optional[List[int]] = LIST_FIELD(
        arrList=None,
        default_type=None,
        description="Image scale for the augmentation",
        display_name="Image Scale"
    )
    ratio_range: List[float] = LIST_FIELD(
        arrList=[0.5, 2.0],
        description="Ratio range for the augmentation",
        display_name="Ratio Range"
    )
    keep_ratio: bool = BOOL_FIELD(value=True, description="Bool whether to keep ratio", display_name="Keep Ratio")


@dataclass
class SFAugmentationConfig:
    """Augmentation config."""

    # @subha: TO Do: Add some more augmentation configurations which were not used in Segformer (later)
    random_crop: RandomCropCfg = DATACLASS_FIELD(RandomCropCfg())
    resize: ResizeCfg = DATACLASS_FIELD(ResizeCfg())
    random_flip: Dict[str, float] = DICT_FIELD(
        hashMap={"prob": 0.5},
        description="Random flip for the augmentation",
        display_name="Random Flip"
    )
    color_aug: Dict[str, str] = DICT_FIELD(
        hashMap={"type": "PhotoMetricDistortion"},
        description="Color Augmentation for the augmentation",
        display_name="Color Augmentation"
    )


@dataclass
class ImgNormConfig:
    """Configuration parameters for Img Normalization."""

    mean: List[float] = LIST_FIELD(
        arrList=[123.675, 116.28, 103.53],
        description="Mean for the augmentation",
        display_name="Mean"
    )
    std: List[float] = LIST_FIELD(
        arrList=[58.395, 57.12, 57.375],
        description="Standard deviation for the augmentation",
        display_name="Standard Deviation"
    )
    to_rgb: bool = BOOL_FIELD(value=True, description="Bool whether to convert to RGB", display_name="To RGB")
    pad_val: int = INT_FIELD(
        value=0,
        description="Pad value for the augmentation",
        display_name="Pad Value",
        default_value=0
    )
    seg_pad_val: int = INT_FIELD(
        value=255,
        description="Seg Pad value for the augmentation",
        display_name="Seg Pad Value",
        default_value=255
    )
    type: str = STR_FIELD(
        value="SegDataPreProcessor",
        description="Type of data preprocessor",
        valid_options="SegDataPreProcessor"
    )


@dataclass
class PipelineConfig:
    """Configuration parameters for Validation Pipe."""

    img_norm_cfg: ImgNormConfig = DATACLASS_FIELD(ImgNormConfig())
    multi_scale: Optional[List[int]] = LIST_FIELD(
        arrList=None,
        default_type=None,
        description="Multi scale for the augmentation",
        display_name="Multi Scale"
    )
    augmentation_config: SFAugmentationConfig = DATACLASS_FIELD(SFAugmentationConfig())
    Pad: Dict[str, int] = DICT_FIELD(
        hashMap={"size_ht": 1024, "size_wd": 1024, "pad_val": 0, "seg_pad_val": 255},
        description="Pad for the augmentation",
        display_name="Pad"
    )
    CollectKeys: List[str] = LIST_FIELD(
        arrList=["img", "gt_semantic_seg"],
        display_name="Collect Keys",
        description="Collect Keys"
    )


@dataclass
class seg_class:
    """Indiv color."""

    seg_class: str = STR_FIELD(value="background", description="Segmentation class", valid_options="background")
    mapping_class: str = STR_FIELD(value="background", description="Mapping class", valid_options="background")
    label_id: int = INT_FIELD(value=0, description="Label ID", valid_min=0)
    rgb: List[int] = LIST_FIELD(arrList=[255, 255, 255], description="RGB values", display_name="RGB Values")


@dataclass
class SFListDatasetConfig:
    """Dataset Config."""

    img_dir: List[str] = LIST_FIELD(
        arrList=["path"],
        display_name="Train Image directory list",
        description=(
            "List of strings, where each string is path to a segformer support dataset "
            "containing images"
        )
    )
    ann_dir: List[str] = LIST_FIELD(
        arrList=["path"],
        display_name="Train Annotation directory list",
        description=(
            "List of strings, where each string is path to a segformer support dataset "
            "containing annotations"
        )
    )
    pipeline: PipelineConfig = DATACLASS_FIELD(PipelineConfig())


@dataclass
class SFDatasetConfig:
    """Dataset Config."""

    img_dir: Optional[Any] = STR_FIELD(value="", description="Image directory", default_value="")
    ann_dir: Optional[Any] = STR_FIELD(value="", description="Annotation directory", default_value="")
    pipeline: PipelineConfig = DATACLASS_FIELD(PipelineConfig())


@dataclass
class SFDatasetExpConfig:
    """Dataset config."""

    data_root: Optional[str] = STR_FIELD(value=None, description="Data root", default_value="")
    img_norm_cfg: ImgNormConfig = DATACLASS_FIELD(ImgNormConfig())
    train_dataset: SFListDatasetConfig = DATACLASS_FIELD(SFListDatasetConfig())
    val_dataset: SFDatasetConfig = DATACLASS_FIELD(SFDatasetConfig())
    test_dataset: SFDatasetConfig = DATACLASS_FIELD(SFDatasetConfig())
    palette: Optional[List[Dict[Any, Any]]] = LIST_FIELD(
        arrList=[
            {
                "label_id": 0,
                "mapping_class": "foreground",
                "rgb": [0, 0, 0],
                "seg_class": "foreground"
            },
            {
                "label_id": 1,
                "mapping_class": "background",
                "rgb": [255, 255, 255],
                "seg_class": "background"
            }
        ],
        description="Palette",
        display_name="Palette"
    )
    seg_class_default: seg_class = DATACLASS_FIELD(seg_class())
    dataloader: str = STR_FIELD(value="DataLoader", description="Dataloader", valid_options="DataLoader")
    img_suffix: Optional[str] = STR_FIELD(value=".png", description="Image suffix", display_name="Image Suffix")
    seg_map_suffix: Optional[str] = STR_FIELD(
        value=".png",
        description="Segmentation map suffix",
        display_name="Segmentation Map Suffix"
    )
    repeat_data_times: int = INT_FIELD(value=2, description="Repeat data times", display_name="Repeat Data Times")
    batch_size: int = INT_FIELD(
        value=32,
        description="Batch size",
        display_name="Batch Size",
        valid_min=1,
        valid_max="inf"
    )
    workers_per_gpu: int = INT_FIELD(
        value=8,
        description="Workers per GPU",
        display_name="Workers per GPU",
        valid_min=1,
        valid_max="inf"
    )
    shuffle: bool = BOOL_FIELD(value=True, description="Bool whether to shuffle", display_name="Shuffle")
    input_type: str = STR_FIELD(value="grayscale", description="Input type", valid_options="rgb,grayscale")

    # This is set to false because we only have 2 classes in the example
    reduce_zero_label: bool = BOOL_FIELD(
        value=False,
        description="Bool whether to reduce zero label",
        display_name="Reduce Zero Label"
    )
    type: str = STR_FIELD(value="BaseSegDataset", description="Type of dataset", valid_options="BaseSegDataset")


@dataclass
class SFEnvConfig:
    """Env Config for Segformer."""

    cudnn_benchmark: bool = BOOL_FIELD(
        value=True,
        description="Bool whether to use cudnn benchmark",
        display_name="Cudnn Benchmark"
    )
    mp_cfg: Dict[Any, Any] = DICT_FIELD(
        hashMap={'mp_start_method': 'fork', 'opencv_num_threads': 0},
        description="mp config dictionary"
    )
    dist_cfg: Dict[str, str] = DICT_FIELD(
        hashMap={'backend': 'nccl'},
        display_name="Dist Config",
        description="Dist Config"
    )


@dataclass
class SFExpConfig:
    """Overall Exp Config for Segformer."""

    manual_seed: int = INT_FIELD(value=47, description="Manual seed", display_name="Manual Seed")
    distributed: bool = BOOL_FIELD(
        value=True,
        description="Bool whether to use distributed",
        display_name="Distributed"
    )
    deterministic: bool = BOOL_FIELD(
        value=False,
        description="Bool whether to use deterministic",
        display_name="Deterministic"
    )
    gpu_ids: List[int] = LIST_FIELD(arrList=[0], display_name="GPU IDs", description="GPU IDs")
    MASTER_ADDR: str = STR_FIELD(value="127.0.0.1", description="Master address", display_name="Master Address")
    MASTER_PORT: int = INT_FIELD(value=25678, description="Master port", display_name="Master Port")
    env_cfg: SFEnvConfig = DATACLASS_FIELD(SFEnvConfig())
    default_scope: str = STR_FIELD(value="mmseg", description="Default scope", display_name="Default Scope")
    log_level: str = STR_FIELD(value="INFO", description="Log level", display_name="Log Level", valid_options="INFO")


@dataclass
class MultiStepLRConfig:
    """Configuration parameters for Multi Step Optimizer."""

    lr_steps: List[int] = LIST_FIELD(
        arrList=[15, 25],
        description="Learning rate steps",
        display_name="Learning rate steps"
    )
    lr_decay: float = FLOAT_FIELD(
        value=0.1,
        default_value=0.1,
        description="Learning rate decay",
        display_name="Learning rate decay"
    )


# This abides by API spec but is translated to mmengine via LinearLR/PolyLRConfig
@dataclass
class LRConfig:
    """Configuration parameters for LR Scheduler."""

    policy: str = STR_FIELD(
        value="poly",
        description="Learning rate policy",
        display_name="Learning rate policy",
        valid_options="poly"
    )
    warmup: str = STR_FIELD(
        value="linear",
        description="Warmup policy",
        display_name="Warmup policy",
        valid_options="linear"
    )
    warmup_iters: int = INT_FIELD(
        value=1500,
        description="Warmup iterations",
        display_name="Warmup iterations",
        min_value=0,
        max_value=1000000
    )
    warmup_ratio: float = FLOAT_FIELD(
        value=1e-6,
        description="Warmup ratio",
        display_name="Warmup ratio",
        min_value=0.0,
        max_value=1.0
    )
    power: float = FLOAT_FIELD(value=1.0, description="Power", display_name="Power")
    min_lr: float = FLOAT_FIELD(value=0.0, description="Minimum learning rate", display_name="Minimum learning rate")
    by_epoch: bool = BOOL_FIELD(value=False, description="By epoch", display_name="By epoch")


@dataclass
class LinearLRConfig:
    """Configuration parameters for Linear LR."""

    type: str = STR_FIELD(value="LinearLR", description="Type of learning rate schedule", valid_options="LinearLR")
    start_factor: float = FLOAT_FIELD(value=1e-6, description="Start factor")
    by_epoch: bool = BOOL_FIELD(value=False, description="Whether the schedule is by epoch")
    begin: int = INT_FIELD(value=0, description="Begin step/epoch")
    end: int = INT_FIELD(value=1500, description="End step/epoch")


@dataclass
class PolyLRConfig:
    """Configuration parameters for Polynomial LR decay."""

    type: str = STR_FIELD(value="PolyLR", description="Type of learning rate schedule", valid_options="PolyLR")
    eta_min: float = FLOAT_FIELD(value=0.0, description="Minimum learning rate")
    power: float = FLOAT_FIELD(value=1.0, description="Power of the polynomial decay")
    begin: int = INT_FIELD(value=1500, description="Begin step/epoch")
    end: int = INT_FIELD(value=160000, description="End step/epoch")
    by_epoch: bool = BOOL_FIELD(value=False, description="Whether the schedule is by epoch")


@dataclass
class ParamwiseConfig:
    """Configuration parameters for Parameters."""

    pos_block: Dict[str, float] = DICT_FIELD({"decay_mult": 0.0}, description="Positional block parameters")
    norm: Dict[str, float] = DICT_FIELD({"decay_mult": 0.0}, description="Normalization parameters")
    head: Dict[str, float] = DICT_FIELD({"lr_mult": 10.0}, description="Head parameters")


@dataclass
class SFOptimConfig:
    """Optimizer config."""

    type: str = STR_FIELD(value="AdamW", description="Optimizer type", valid_options="AdamW")
    lr: float = FLOAT_FIELD(
        value=0.00006,
        description="Learning rate",
        default_value=0.00006,
        valid_min=0,
        valid_max="inf",
        automl_enabled="TRUE"
    )
    betas: List[float] = LIST_FIELD([0.0, 0.9], description="Beta coefficients", display_name="Betas")
    weight_decay: float = FLOAT_FIELD(
        value=5e-4,
        default_value=5e-4,
        math_cond="> 0.0",
        display_name="weight decay",
        description="The weight decay coefficient.",
        automl_enabled="TRUE"
    )
    paramwise_cfg: ParamwiseConfig = DATACLASS_FIELD(ParamwiseConfig())


@dataclass
class TrainerConfig:
    """Train Config."""

    sf_optim: Optional[SFOptimConfig] = DATACLASS_FIELD(SFOptimConfig())
    lr_config: LRConfig = DATACLASS_FIELD(LRConfig())
    grad_clip: float = FLOAT_FIELD(value=0.0, description="Gradient clipping", display_name="Gradient clipping")
    find_unused_parameters: bool = BOOL_FIELD(
        value=True,
        description="Find unused parameters",
        display_name="Find unused parameters"
    )


@dataclass
class SFTrainExpConfig:
    """Train experiment config."""

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_type="",
        description="Results directory",
        display_name="Results directory"
    )
    encryption_key: Optional[str] = STR_FIELD(
        value=None,
        default_type="",
        description="Encryption key",
        display_name="Encryption key"
    )
    exp_config: SFExpConfig = DATACLASS_FIELD(SFExpConfig())
    trainer: TrainerConfig = DATACLASS_FIELD(TrainerConfig())
    num_gpus: int = INT_FIELD(
        value=1,
        default_value=1,
        description="Number of GPUs",
        display_name="Number of GPUs",
        min_value=1,
        popular="yes"
    )
    gpu_ids: List[int] = LIST_FIELD(arrList=[0], description="GPU IDs", display_name="GPU IDs")
    max_iters: int = INT_FIELD(
        value=10,
        description="Maximum iterations",
        display_name="Maximum iterations",
        min_value=1,
        popular="yes"
    )
    logging_interval: int = INT_FIELD(value=1, description="Logging interval", display_name="Logging interval")
    checkpoint_interval: int = INT_FIELD(
        value=1,
        description="Checkpoint interval",
        display_name="Checkpoint interval",
        popular="yes"
    )
    resume_training_checkpoint_path: Optional[str] = STR_FIELD(
        value=None,
        default_type="",
        description="Resume training checkpoint path",
        display_name="Resume training checkpoint path"
    )
    validation_interval: Optional[int] = INT_FIELD(
        value=1,
        default_value=1,
        description="Validation interval",
        display_name="Validation interval",
        popular="yes"
    )
    validate: bool = BOOL_FIELD(value=True, description="Validate", display_name="Validate")
    param_scheduler: List[Any] = LIST_FIELD(
        arrList=[vars(LinearLRConfig()), vars(PolyLRConfig())],
        description="Parameter scheduler",
        display_name="Parameter scheduler"
    )
    resume: bool = BOOL_FIELD(value=False, default_value=False, description="Resume", display_name="Resume")
    default_hooks: Dict[Any, Any] = DICT_FIELD(
        hashMap={
            "timer": {"type": "IterTimerHook"},
            "logger": {
                "type": "TAOTextLoggerHook",
                "interval": 1,
                "log_metric_by_epoch": False
            },
            "param_scheduler": {"type": "ParamSchedulerHook"},
            "checkpoint": {
                "type": "CheckpointHook",
                "by_epoch": False,
                "interval": 1
            },
            "sampler_seed": {"type": "DistSamplerSeedHook"},
            "visualization": {"type": "SegVisualizationHook"}
        },
        description="Default hooks",
        display_name="Default hooks"
    )


@dataclass
class SFInferenceExpConfig(InferenceConfig):
    """Inference experiment config."""

    encryption_key: Optional[str] = STR_FIELD(
        value="",
        default_type="",
        description="Encryption key",
        display_name="Encryption key"
    )
    exp_config: SFExpConfig = DATACLASS_FIELD(SFExpConfig())


@dataclass
class SFEvalExpConfig(EvaluateConfig):
    """Inference experiment config."""

    encryption_key: Optional[str] = STR_FIELD(
        value="",
        default_type="",
        description="Encryption key",
        display_name="Encryption key"
    )
    exp_config: SFExpConfig = DATACLASS_FIELD(SFExpConfig())


@dataclass
class SFTrtConfig(TrtConfig):
    """Trt config."""

    data_type: str = STR_FIELD(
        value="fp32",
        default_value="fp32,fp16",
        description="Data type",
        display_name="Data type"
    )


@dataclass
class OnnxConfig:
    """Onnx config."""

    type: str = STR_FIELD(value="onnx", description="Type of Export", display_name="Type of Export")
    export_params: bool = BOOL_FIELD(value=True, description="Export parameters", display_name="Export parameters")
    keep_initializers_as_inputs: bool = BOOL_FIELD(
        value=True,
        description="Keep initializers as inputs",
        display_name="Keep initializers as inputs"
    )
    opset_version: int = INT_FIELD(
        value=13,
        description="Opset version",
        display_name="Opset version",
        valid_options="11,12,13"
    )
    save_file: Optional[str] = STR_FIELD(value=None, default_type="", description="Save file", display_name="Save file")
    input_names: Optional[Any] = STR_FIELD(
        value=None,
        default_type=None,
        description="Input names",
        display_name="Input names"
    )
    output_names: Optional[Any] = STR_FIELD(
        value=None,
        default_type=None,
        description="Output names",
        display_name="Output names"
    )
    input_shape: Optional[Any] = STR_FIELD(
        value=None,
        default_type=None,
        description="Input shape",
        display_name="Input shape"
    )
    optimize: bool = BOOL_FIELD(value=True, description="Optimize", display_name="Optimize")


@dataclass
class CodebaseConfig:
    """Codebase config for mmseg"""

    type: str = STR_FIELD(value="mmseg", description="Type", display_name="Type")
    task: str = STR_FIELD(value="Segmentation", description="Task", display_name="Task")
    with_argmax: bool = BOOL_FIELD(value=True, description="With argmax", display_name="With argmax")


@dataclass
class SFExportExpConfig:
    """Export experiment config."""

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_type="",
        description="Results directory",
        display_name="Results directory"
    )
    encryption_key: Optional[str] = STR_FIELD(
        value=None,
        default_type="",
        description="Encryption key",
        display_name="Encryption key"
    )
    verify: bool = BOOL_FIELD(value=True, description="Verify", display_name="Verify")
    simplify: bool = BOOL_FIELD(value=True, description="Simplify", display_name="Simplify")
    batch_size: int = INT_FIELD(value=1, description="Batch size", display_name="Batch size", min_value=1)
    opset_version: int = INT_FIELD(
        value=13,
        description="Opset version",
        display_name="Opset version",
        valid_options="11,12,13"
    )
    trt_engine: Optional[str] = STR_FIELD(
        value=None,
        default_type="",
        description="TRT engine",
        display_name="TRT engine"
    )
    checkpoint: Optional[str] = STR_FIELD(
        value=None,
        default_type="",
        description="Checkpoint",
        display_name="Checkpoint"
    )
    onnx_file: Optional[str] = STR_FIELD(value=None, default_type="", description="ONNX file", display_name="ONNX file")
    exp_config: SFExpConfig = DATACLASS_FIELD(SFExpConfig())
    trt_config: SFTrtConfig = DATACLASS_FIELD(SFTrtConfig())
    num_gpus: int = INT_FIELD(value=1, description="Number of GPUs", display_name="Number of GPUs", popular="yes")
    input_channel: int = INT_FIELD(
        value=3,
        description="Input channel",
        display_name="Input channel",
        valid_options="1,3",
        popular="yes"
    )
    input_width: int = INT_FIELD(value=1024, description="Input width", display_name="Input width", popular="yes")
    input_height: int = INT_FIELD(value=1024, description="Input height", display_name="Input height", popular="yes")

    onnx_config: OnnxConfig = DATACLASS_FIELD(OnnxConfig())
    codebase_config: CodebaseConfig = DATACLASS_FIELD(CodebaseConfig())


@dataclass
class GenTrtEngineExpConfig(GenTrtEngineConfig):
    """Gen TRT Engine experiment config."""

    tensorrt: SFTrtConfig = DATACLASS_FIELD(SFTrtConfig())


@dataclass
class ExperimentConfig:
    """Experiment config."""

    model_name: Optional[str] = STR_FIELD(value=None)
    model: SFModelConfig = DATACLASS_FIELD(SFModelConfig())
    dataset: SFDatasetExpConfig = DATACLASS_FIELD(SFDatasetExpConfig())
    train: SFTrainExpConfig = DATACLASS_FIELD(SFTrainExpConfig())
    evaluate: SFEvalExpConfig = DATACLASS_FIELD(SFEvalExpConfig())
    inference: SFInferenceExpConfig = DATACLASS_FIELD(SFInferenceExpConfig())
    gen_trt_engine: GenTrtEngineExpConfig = DATACLASS_FIELD(GenTrtEngineExpConfig())
    export: SFExportExpConfig = DATACLASS_FIELD(SFExportExpConfig())
    encryption_key: Optional[str] = STR_FIELD(
        value="",
        default_type="",
        description="Encryption key",
        display_name="Encryption key"
    )
    results_dir: str = STR_FIELD(
        value="",
        default_value="",
        description="Results directory",
        display_name="Results directory"
    )
