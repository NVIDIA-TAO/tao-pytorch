# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default Config"""

from typing import Optional, List
from dataclasses import dataclass
from omegaconf import MISSING

from nvidia_tao_pytorch.config.common.common_config import (
    CommonExperimentConfig,
    InferenceConfig,
    TrainConfig,
    GenTrtEngineConfig,
    TrtConfig
)
from nvidia_tao_pytorch.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    BOOL_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
)

SUPPORTED_BACKBONES = [
    *["vit_l"]
]

SUPPORTED_BACKBONES = [
    *["vit_l", "vit_b", "vit_s"]
]

map_params = {
    'embed_dim': {
        'vit_l': 1024,
        'vit_b': 768,
        'vit_s': 384
    },
    'depth': {
        'vit_l': 24,
        'vit_b': 12,
        'vit_s': 12
    },
    'num_heads': {
        'vit_l': 16,
        'vit_b': 12,
        'vit_s': 6
    },
    'init_values': {
        'vit_l': 1e-5,
        'vit_b': 1e-5,
        'vit_s': 1e-5
    },
    'drop_path_schedule': {
        'vit_l': 'linear',
        'vit_b': 'linear',
        'vit_s': 'linear'
    },
    'num_classes': {
        'vit_l': 0,
        'vit_b': 0,
        'vit_s': 0
    },
}


@dataclass
class DataPathFormat:
    """Dataset Path experiment config."""

    images_dir: str = STR_FIELD(
        value="",
        default_value="",
        description="Path to images directory for dataset",
        display_name="image directory"
    )


@dataclass
class NVDINOv2TransformConfig:
    """NVDINOv2 Data Transform Config."""

    n_global_crops: int = INT_FIELD(
        value=2,
        default_value=2,
        valid_min=1,
        valid_max="inf",
        description="Number of global crops to generate",
        display_name="Number of Global Crops",
        popular="yes"
    )
    global_crops_scale: List[float] = LIST_FIELD(
        arrList=[0.32, 1.0],
        default_value=[0.32, 1.0],
        description="Scale range for global crops",
        display_name="Global Crops Scale",
        popular="yes"
    )
    global_crops_size: int = INT_FIELD(
        value=224,
        default_value=224,
        valid_min=1,
        valid_max="inf",
        description="Size of global crops",
        display_name="Global Crops Size",
        popular="yes"
    )
    n_local_crops: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="Number of local crops to generate",
        display_name="Number of Local Crops",
        popular="yes"
    )
    local_crops_scale: List[float] = LIST_FIELD(
        arrList=[0.05, 0.32],
        default_value=[0.05, 0.32],
        description="Scale range for local crops",
        display_name="Local Crops Scale",
        popular="yes"
    )
    local_crops_size: int = INT_FIELD(
        value=98,
        default_value=98,
        valid_min=1,
        valid_max="inf",
        description="Size of local crops",
        display_name="Local Crops Size",
        popular="yes"
    )


@dataclass
class NVDINOv2DatasetConfig:
    """NVDINOv2 Dataset Config."""

    train_dataset: DataPathFormat = DATACLASS_FIELD(
        DataPathFormat(),
        description="Configuration for the training dataset path",
        display_name="Training Dataset"
    )
    test_dataset: DataPathFormat = DATACLASS_FIELD(
        DataPathFormat(),
        description="Configuration for the testing dataset path",
        display_name="Testing Dataset"
    )
    batch_size: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=1,
        valid_max="inf",
        description="The batch size for training",
        automl_enabled="TRUE",
        display_name="batch size",
        popular="yes"
    )
    pin_memory: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="pin_memory",
        description=(
            "Flag to enable the dataloader to allocated pagelocked memory for faster "
            "of data between the CPU and GPU."
        ),
        popular="yes"
    )
    workers: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="The number of parallel workers processing data",
        automl_enabled="TRUE",
        display_name="batch size",
        popular="yes"
    )
    transform: NVDINOv2TransformConfig = DATACLASS_FIELD(
        NVDINOv2TransformConfig(),
        description="Configuration parameters for data transformation",
        display_name="transform",
    )


@dataclass
class BackboneConfig:
    """Configuration parameters for Backbone."""

    teacher_type: str = STR_FIELD(
        value="vit_l",
        default_value="vit_l",
        display_name="backbone",
        description=(
            "The teacher backbone name of the model. "
            "TAO implementation of NVDINOv2 support vit_l and vit_s"
        ),
        valid_options=",".join(SUPPORTED_BACKBONES),
        popular="no"
    )
    student_type: str = STR_FIELD(
        value="vit_l",
        default_value="vit_l",
        display_name="backbone",
        description=(
            "The student backbone name of the model. "
            "TAO implementation of NVDINOv2 support vit_l and vit_s"
        ),
        valid_options=",".join(SUPPORTED_BACKBONES),
        popular="no"
    )
    num_register_tokens: int = INT_FIELD(
        value=0,
        default_value=0,
        valid_min=0,
        valid_max="inf",
        description="Number of register tokens",
        display_name="num register tokens",
        popular="yes"
    )
    drop_path_rate: float = FLOAT_FIELD(
        value=0.4,
        default_value=0.4,
        description="Drop path rate for stochastic depth regularization",
        display_name="drop path rate",
        popular="yes"
    )
    patch_size: int = INT_FIELD(
        value=14,
        default_value=14,
        description="Size of patches",
        display_name="patch size",
        valid_options="14,16",
        popular="yes"
    )
    img_size: int = INT_FIELD(
        value=518,
        default_value=518,
        description="Size of images for the backbone",
        display_name="image size",
        valid_options="224,518",
        popular="yes"
    )


@dataclass
class NVDINOv2HeadConfig:
    """Configuration parameters for NVDINOv2 head."""

    num_layers: int = INT_FIELD(
        value=3,
        default_value=3,
        valid_min=1,
        valid_max="inf",
        description="Number of layers in the NVDINOv2 head",
        display_name="number of Layers",
        popular="yes"
    )
    hidden_dim: int = INT_FIELD(
        value=2048,
        default_value=2048,
        valid_min=1,
        valid_max="inf",
        description="Dimension of the hidden layers in the NVDINOv2 head",
        display_name="hidden dimension",
        popular="yes"
    )
    bottleneck_dim: int = INT_FIELD(
        value=384,
        default_value=384,
        valid_min=1,
        valid_max="inf",
        description="Dimension of the bottleneck layer in the NVDINOv2 head",
        display_name="bottleneck dimension",
        popular="yes"
    )


@dataclass
class NVDINOv2ModelDistillConfig:
    """NVDINOv2 Model config."""

    enable: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Whether to run distillation",
        display_name="distillation",
        popular="yes"
    )
    disable_masking: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Whether to disable masking when distillation",
        display_name="disable_masking",
        popular="yes"
    )
    pretrained_non_distill_pl_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_type=None,
        description=(
            "Path to a pre-trained pl model from non-distillation DINOv2 SSL pipe "
            "for initializing teacher in distillation."
        )
    )


@dataclass
class NVDINOv2ModelConfig:
    """NVDINOv2 Model config."""

    distill: NVDINOv2ModelDistillConfig = DATACLASS_FIELD(
        NVDINOv2ModelDistillConfig(),
        description="Configuration for the NVDINOv2 distillation"
    )
    backbone: BackboneConfig = DATACLASS_FIELD(
        BackboneConfig(),
        description="Configuration for the NVDINOv2 backbone"
    )
    head: NVDINOv2HeadConfig = DATACLASS_FIELD(
        NVDINOv2HeadConfig(),
        description="Configuration for the NVDINOv2 head"
    )


@dataclass
class NVDINOv2BaseSchedulerConfig:
    """Scheduler Config."""

    val_base: float = FLOAT_FIELD(
        value=1e-8,
        default_value=1e-8,
        description="The value after warm-up for scheduler",
        display_name="base value",
        popular="yes"
    )
    val_final: float = FLOAT_FIELD(
        value=1e-6,
        default_value=1e-6,
        description="Final value for scheduler",
        display_name="final value",
        popular="yes"
    )
    val_start: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        description="Starting value for scheduler",
        display_name="starting value",
        popular="yes"
    )
    warm_up_steps: int = INT_FIELD(
        value=0,
        default_value=0,
        description="Number of warm-up steps",
        display_name="warm-up steps",
        popular="yes"
    )
    max_decay_steps: int = INT_FIELD(
        value=2500000,
        default_value=2500000,
        description="Maximum decay steps",
        display_name="max decay steps",
        popular="yes"
    )


@dataclass
class NVDINOv2LearningRateConfig(NVDINOv2BaseSchedulerConfig):
    """Learning Rate Scheduler Config."""

    val_base: float = FLOAT_FIELD(
        value=7.07e-6,
        default_value=7.07e-6,
        description="The value after warm-up for scheduler",
        display_name="base value",
        popular="yes"
    )
    warm_up_steps: int = INT_FIELD(
        value=100000,
        default_value=100000,
        description="Number of warm-up steps",
        display_name="warm-up steps",
        popular="yes"
    )


@dataclass
class NVDINOv2LastLayerLearningRateConfig(NVDINOv2BaseSchedulerConfig):
    """Last Layer Learning Rate Scheduler Config."""

    val_base: float = FLOAT_FIELD(
        value=7.07e-6,
        default_value=7.07e-6,
        description="The value after warm-up for scheduler.",
        display_name="base value",
        popular="yes"
    )
    warm_up_steps: int = INT_FIELD(
        value=100000,
        default_value=100000,
        description="Number of warm-up steps",
        display_name="warm-up steps",
        popular="yes"
    )
    freeze_steps: int = INT_FIELD(
        value=1250,
        default_value=1250,
        description="Number of freeze steps",
        display_name="freeze steps",
        popular="yes"
    )


@dataclass
class NVDINOv2WeightDecayConfig(NVDINOv2BaseSchedulerConfig):
    """Weight Decay Scheduler Config."""

    val_base: float = FLOAT_FIELD(
        value=0.04,
        default_value=0.04,
        description="The value after warm-up for scheduler",
        display_name="base value",
        popular="yes"
    )
    val_final: float = FLOAT_FIELD(
        value=0.2,
        default_value=0.2,
        description="Final value for scheduler",
        display_name="final value",
        popular="yes"
    )


@dataclass
class NVDINOv2MomentumConfig(NVDINOv2BaseSchedulerConfig):
    """Momentum Scheduler Config."""

    val_base: float = FLOAT_FIELD(
        value=0.994,
        default_value=0.994,
        description="The value after warm-up for scheduler",
        display_name="base value",
        popular="yes"
    )
    val_final: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        description="Final value for scheduler",
        display_name="final value",
        popular="yes"
    )


@dataclass
class NVDINOv2TeacherTemperatureConfig(NVDINOv2BaseSchedulerConfig):
    """Teacher Temperature Scheduler Config."""

    val_base: float = FLOAT_FIELD(
        value=0.07,
        default_value=0.07,
        description="The value after warm-up for scheduler",
        display_name="base value",
        popular="yes"
    )
    val_final: float = FLOAT_FIELD(
        value=0.07,
        default_value=0.07,
        description="Final value for scheduler",
        display_name="final value",
        popular="yes"
    )
    val_start: float = FLOAT_FIELD(
        value=0.04,
        default_value=0.04,
        description="Starting value for scheduler",
        display_name="starting value",
        popular="yes"
    )
    warm_up_steps: int = INT_FIELD(
        value=37500,
        default_value=37500,
        description="Number of warm-up steps",
        display_name="warm-up steps",
        popular="yes"
    )
    max_decay_steps: int = INT_FIELD(
        value=37500,
        default_value=37500,
        description="Maximum decay steps",
        display_name="max decay steps",
        popular="yes"
    )


@dataclass
class NVDINOv2SchedulerConfig:
    """Schedulers Config."""

    learning_rate: NVDINOv2LearningRateConfig = DATACLASS_FIELD(
        NVDINOv2LearningRateConfig(),
        description="Learning rate scheduler configuration"
    )
    last_layer_learning_rate: NVDINOv2LastLayerLearningRateConfig = DATACLASS_FIELD(
        NVDINOv2LastLayerLearningRateConfig(),
        description="Last layer learning rate scheduler configuration"
    )
    weight_decay: NVDINOv2WeightDecayConfig = DATACLASS_FIELD(
        NVDINOv2WeightDecayConfig(),
        description="Weight decay scheduler configuration"
    )
    momentum: NVDINOv2MomentumConfig = DATACLASS_FIELD(
        NVDINOv2MomentumConfig(),
        description="Momentum scheduler configuration"
    )
    teacher_temperature: NVDINOv2TeacherTemperatureConfig = DATACLASS_FIELD(
        NVDINOv2TeacherTemperatureConfig(),
        description="Teacher temperature scheduler configuration"
    )


@dataclass
class NVDINOv2OptimConfig:
    """Optimizer config."""

    optim: str = STR_FIELD(
        value="adamw",
        default_value="adamw",
        description="Optimizer type",
        display_name="optimizer",
        valid_options="adamw,",
        popular="yes"
    )


@dataclass
class NVDINOv2TrainExpConfig(TrainConfig):
    """Train Config."""

    pretrained_model_path: Optional[str] = STR_FIELD(
        value=None,
        default_type=None,
        description=(
            "Path to a pre-trained NVDINOv2 model to initialize the current training from."
        )
    )
    layerwise_decay: float = FLOAT_FIELD(
        value=1.0,
        default_value=1.0,
        description="Layerwise decay factor",
        display_name="layerwise decay factor",
        popular="yes"
    )
    clip_grad_norm: float = FLOAT_FIELD(
        value=3.0,
        default_value=3.0,
        description="Value to clip gradients norm",
        display_name="clip gradient norm",
        popular="yes"
    )
    num_prototypes: int = INT_FIELD(
        value=131072,
        default_value=131072,
        description="Number of prototypes",
        display_name="number of prototypes",
        popular="yes"
    )
    precision: str = STR_FIELD(
        value="16-mixed",
        default_value="16-mixed",
        description="Precision",
        display_name="precision",
        popular="yes"
    )
    use_custom_attention: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Whether to use memory_efficient_attention",
        display_name="custom_attention",
        popular="yes"
    )
    schedulers: NVDINOv2SchedulerConfig = DATACLASS_FIELD(
        NVDINOv2SchedulerConfig(),
        description="Schedulers configuration for NVDINOv2 training"
    )
    optim: NVDINOv2OptimConfig = DATACLASS_FIELD(
        NVDINOv2OptimConfig(),
        description="Optimizer configuration for NVDINOv2"
    )


@dataclass
class NVDINOv2TrtConfig(TrtConfig):
    """Trt config."""

    data_type: str = STR_FIELD(
        value="fp32",
        default_value="fp32,fp16",
        description="Data type",
        display_name="Data type"
    )


@dataclass
class NVDINOv2InferenceExpConfig(InferenceConfig):
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
class NVDINOv2ExportExpConfig:
    """Export experiment config."""

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Results directory",
        display_name="Results directory"
    )
    gpu_id: int = INT_FIELD(
        value=0,
        default_value=0,
        description="GPU ID",
        display_name="GPU ID",
        value_min=0
    )
    checkpoint: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="Path to checkpoint file",
        display_name="Path to checkpoint file"
    )
    onnx_file: Optional[str] = STR_FIELD(
        value=MISSING,
        default_value="",
        description="ONNX file",
        display_name="ONNX file"
    )
    on_cpu: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to export on cpu",
        display_name="On CPU"
    )
    input_channel: int = INT_FIELD(
        value=3,
        default_value=3,
        description="Input channel",
        display_name="Input channel"
    )
    input_width: int = INT_FIELD(
        value=518,
        default_value=518,
        description="Input width",
        display_name="Input width",
        valid_min=128
    )
    input_height: int = INT_FIELD(
        value=518,
        default_value=518,
        description="Input height",
        display_name="Input height",
        valid_min=128
    )
    opset_version: int = INT_FIELD(
        value=17,
        default_value=12,
        valid_min=1,
        display_name="opset version",
        description=(
            "Operator set version of the ONNX model used to generate the TensorRT engine."
        )
    )
    batch_size: int = INT_FIELD(
        value=-1,
        default_value=-1,
        description="Batch size",
        display_name="Batch size",
        valid_min=0
    )
    verbose: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Verbose",
        display_name="Verbose"
    )


@dataclass
class GenTrtEngineExpConfig(GenTrtEngineConfig):
    """Gen TRT Engine experiment config."""

    tensorrt: NVDINOv2TrtConfig = DATACLASS_FIELD(NVDINOv2TrtConfig())


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: NVDINOv2ModelConfig = DATACLASS_FIELD(
        NVDINOv2ModelConfig(),
        description="Configurable parameters to construct the model for a NVDINOv2 experiment.",
    )
    dataset: NVDINOv2DatasetConfig = DATACLASS_FIELD(
        NVDINOv2DatasetConfig(),
        description="Configurable parameters to construct the dataset for a NVDINOv2 experiment.",
    )
    train: NVDINOv2TrainExpConfig = DATACLASS_FIELD(
        NVDINOv2TrainExpConfig(),
        description="Configurable parameters to construct the trainer for a NVDINOv2 experiment.",
    )
    inference: NVDINOv2InferenceExpConfig = DATACLASS_FIELD(
        NVDINOv2InferenceExpConfig(),
        description=(
            "Configurable parameters to construct the inference trainer for a NVDINOv2 experiment."
        ),
    )
    export: NVDINOv2ExportExpConfig = DATACLASS_FIELD(
        NVDINOv2ExportExpConfig(),
        description="Configurable parameters to export for a NVDINOv2 experiment.",
    )
    gen_trt_engine: GenTrtEngineExpConfig = DATACLASS_FIELD(
        GenTrtEngineExpConfig(),
        description=(
            "Configurable parameters to generate TensorRT engine for a NVDINOv2 experiment."
        ),
    )

    def __post_init__(self):
        """Set default model name for NVDINOv2."""
        if self.model_name is None:
            self.model_name = "nvdinov2"
