# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file."""

from typing import List, Optional
from dataclasses import dataclass

from nvidia_tao_pytorch.config.common.common_config import (
    CommonExperimentConfig,
    EvaluateConfig,
    ExportConfig,
    TrtConfig,
    GenTrtEngineConfig,
    InferenceConfig,
    TrainConfig,
)
from nvidia_tao_pytorch.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    BOOL_FIELD,
    FLOAT_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class OptimConfig:
    """Optimizer config."""

    type: str = STR_FIELD(
        value="AdamW",
        default_value="AdamW",
        description="Type of optimizer used to train the network.",
        valid_options=",".join([
            "AdamW"
        ])
    )
    monitor_name: str = STR_FIELD(
        value="train_loss",
        description="The metric value to be monitored for the :code:`AutoReduce` Scheduler.",
        display_name="monitor_name",
        valid_options=",".join(
            ["val_loss", "train_loss"]
        )
    )
    lr: float = FLOAT_FIELD(
        value=2e-4,
        valid_min=0.0,
        valid_max=1.0,
        math_cond="> 0.0",
        display_name="learning rate",
        description="The initial learning rate for training the model.",
        automl_enabled="TRUE"
    )
    backbone_multiplier: float = FLOAT_FIELD(
        value=0.1,
        valid_min=0.0,
        valid_max=1.0,
        math_cond="> 0.0",
        display_name="backbone learning rate multiplier",
        description="A multiplier for backbone learning rate.",
        automl_enabled="TRUE",
        popular="yes",
    )
    momentum: float = FLOAT_FIELD(
        value=0.9,
        valid_min=0.0,
        valid_max=1.0,
        math_cond="> 0.0",
        display_name="momentum - AdamW",
        description="The momentum for the AdamW optimizer.",
        automl_enabled="TRUE",
        popular="yes",
    )
    weight_decay: float = FLOAT_FIELD(
        value=0.05,
        valid_min=0.0,
        valid_max=1.0,
        math_cond="> 0.0",
        display_name="weight decay",
        description="The weight decay coefficient.",
        automl_enabled="TRUE",
        popular="yes",
    )
    layer_decay: float = FLOAT_FIELD(
        value=0.75,
        math_cond="> 0.0",
        display_name="layer decay",
        description="The layer decay coefficient.",
        popular="yes",
    )
    lr_scheduler: str = STR_FIELD(
        value="MultiStep",  # {val_loss, train_loss}
        description="""The learning scheduler:
                    * MultiStep : Decrease the lr by lr_decay from lr_steps
                    * cosine : Poly learning rate schedule.""",
        display_name="learning rate scheduler",
        valid_options=",".join(
            ["MultiStep", "cosine"]
        )
    )
    milestones: List[int] = LIST_FIELD(
        arrList=[88, 96],
        description="""learning rate decay epochs.""",
        display_name="learning rate decay epochs."
    )
    gamma: float = FLOAT_FIELD(
        value=0.1,
        math_cond="> 0.0",
        display_name="gamma",
        description="Multiplicative factor of learning rate decay.",
    )
    warmup_epochs: float = INT_FIELD(
        value=1,
        valid_min=0,
        valid_max=100,
        math_cond=">= 0",
        display_name="Warmup epochs",
        description="Warmup epochs.",
        automl_enabled="TRUE"
    )


@dataclass
class AugmentationConfig:
    """Augmentation configuration template."""

    input_size: int = INT_FIELD(
        value=224, default_value=224,
        display_name="Input size",
        description="Input size.")
    mean: List[float] = LIST_FIELD(
        [0.485, 0.456, 0.406],
        display_name="Mean for the image normalization",
        description="Image mean.")
    std: List[float] = LIST_FIELD(
        [0.229, 0.224, 0.225],
        description="Standard deviation for the image normalization",
        display_name="Image standard deviation")
    min_scale: float = FLOAT_FIELD(
        value=0.1,
        description="Min scale for resizing augmentation",
        display_name="Min scale.")
    max_scale: float = FLOAT_FIELD(
        value=2.0,
        description="Max scale for resizing augmentation",
        display_name="Max scale.")
    min_ratio: float = FLOAT_FIELD(
        value=0.75,
        default_value=0.75,
        description="Min ratio for resizing augmentation",
        display_name="Min ratio.")
    max_ratio: float = FLOAT_FIELD(
        value=1.33,
        default_value=1.33,
        description="Max ratio for resizing augmentation",
        display_name="Max ratio.")
    hflip: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="Horizontal flip probability",
        display_name="Horizontal flip probability.")
    re_prob: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        description="Random erasing probability",
        display_name="Random erasing probability.")
    interpolation: str = STR_FIELD(
        value="bilinear",
        default_value="bilinear",
        description="Interpolation mode during training",
        display_name="Interpolation mode.",
        valid_options=",".join([
            "bilinear", "bicubic", "random"
        ]))
    smoothing: float = FLOAT_FIELD(
        value=0.1, default_value=0.1,
        description="Label smoothing",
        display_name="Label smoothing.")
    color_jitter: float = FLOAT_FIELD(
        value=0.0,
        description="Color jittering",
        display_name="Color jittering.")
    auto_aug: Optional[str] = STR_FIELD(
        value='rand-m9-mstd0.5-inc1',
        default_value='rand-m9-mstd0.5-inc1',
        description="Auto augmentation settings",
        display_name="Auto augmentation settings.")

    # mixup aug
    mixup: float = FLOAT_FIELD(
        value=0.8,
        description="Mixup augmentation",
        display_name="Mixup augmentation.")
    cutmix: float = FLOAT_FIELD(
        value=1.0,
        description="Cutmix augmentation",
        display_name="Cutmix augmentation.")
    cutmix_minmax: Optional[float] = FLOAT_FIELD(
        value=None,
        description="Cutmix minmax augmentation",
        display_name="Cutmix minmax augmentation.")
    mixup_prob: float = FLOAT_FIELD(
        value=1.0,
        description="Mixup probability",
        display_name="Mixup probability.")
    mixup_switch_prob: float = FLOAT_FIELD(
        value=0.5,
        description="Mixup switch probability",
        display_name="Mixup switch probability.")
    mixup_mode: str = STR_FIELD(
        value="batch",
        valid_options=",".join([
            "batch", "pair", "elem"]),
        description="Mixup mode",
        display_name="Mixup mode.")


@dataclass
class MAEDatasetConfig:
    """Data configuration template."""

    batch_size: int = INT_FIELD(
        value=3, default_value=1, valid_min=1, valid_max="inf",
        description="Batch size.",
        display_name="Batch size")
    train_data_sources: str = STR_FIELD(
        value='',
        display_name="Image directory of the training set")
    val_data_sources: str = STR_FIELD(
        value='',
        display_name="Image directory of the validation set")
    test_data_sources: Optional[str] = STR_FIELD(
        value='', default_value=None,
        display_name="Image directory of the test set")
    num_workers_per_gpu: int = INT_FIELD(
        value=2, default_value=2,
        description="Number of workers per GPU",
        display_name="Number of workers per GPU.")
    augmentation: AugmentationConfig = DATACLASS_FIELD(
        AugmentationConfig(),
        description="Configuration parameters for data augmentation",
        display_name="augmentation",
    )


@dataclass
class MAEModelConfig:
    """Model configuration template."""

    arch: str = STR_FIELD(
        value='convnextv2_base', default_value="convnextv2_base",
        valid_options=",".join([
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
            "hiera_large_224",
            "hiera_huge_224",
            "vit_base_patch16",
            "vit_large_patch16",
            "vit_huge_patch14",
        ]),
        description="Model architecture.",
        display_name="Model arch")
    num_classes: int = INT_FIELD(
        value=1000, default_value=1, valid_min=1, valid_max="inf",
        description="Number of classes.",
        display_name="Number of classes")
    drop_path_rate: float = FLOAT_FIELD(
        value=0.1, default_value=0.1,
        description="Drop path rate.",
        display_name="Drop path rate")
    global_pool: bool = BOOL_FIELD(
        value=True, default_value=True,
        description="Whether to use global pooling in ViT or Hiera models.",
        display_name="Global pooling")
    decoder_depth: int = INT_FIELD(
        value=1, default_value=1,
        description="Decoder depth of MAE models.",
        display_name="Decoder depth")
    decoder_embed_dim: int = INT_FIELD(value=512, default_value=512)


@dataclass
class MAETrainExpConfig(TrainConfig):
    """Train configuration template."""

    stage: str = STR_FIELD(
        value="pretrain",
        default_value="pretrain",
        description="Training stage.",
        display_name="Stage",
        valid_options=",".join([
            "pretrain", "finetune",
        ])
    )
    accum_grad_batches: int = INT_FIELD(
        value=1, default_value=1, valid_min=1, valid_max="inf",
        description="Number of accumulated gradient batches",
        display_name="Accum gradient batches")
    pretrained_model_path: Optional[str] = STR_FIELD(
        value=None, default_value="",
        description="Path to the pretrained model",
        display_name="Pretrained model")
    precision: str = STR_FIELD(
        value="fp32",
        default_value="fp32",
        description="Precision to run the training on.",
        display_name="precision",
        valid_options=",".join([
            "fp16", "bf16", "fp32"
        ])
    )
    distributed_strategy: str = STR_FIELD(
        value="ddp",
        valid_options=",".join(
            ["ddp", "fsdp"]
        ),
        display_name="distributed_strategy",
        description="""
        The multi-GPU training strategy.
        DDP (Distributed Data Parallel) and Fully Sharded DDP are supported.""",
    )
    # optim
    optim: OptimConfig = DATACLASS_FIELD(
        OptimConfig(),
        display_name="optimizer",
        description="Hyper parameters to configure the optimizer."
    )
    norm_pix_loss: bool = BOOL_FIELD(
        value=True, default_value=True,
        description="Whether to normalize pixel loss",
        display_name="Normalize pixel loss")
    # freeze
    freeze: Optional[List[str]] = LIST_FIELD(
        arrList=[],
        description="""List of layer names to freeze.""",
        display_name="freeze"
    )
    mask_ratio: float = FLOAT_FIELD(
        value=0.75,
        description="Mask ratio",
        display_name="Mask ratio.")


@dataclass
class MAETRTEngineConfig(TrtConfig):
    """Trt config."""

    data_type: str = STR_FIELD(
        value="fp32",
        default_value="fp32,fp16",
        description="Data type",
        display_name="Data type"
    )


@dataclass
class MAEGenTrtEngineConfig(GenTrtEngineConfig):
    """Gen trt engine config."""

    tensorrt: MAETRTEngineConfig = DATACLASS_FIELD(MAETRTEngineConfig())


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment configuration template."""

    dataset: MAEDatasetConfig = DATACLASS_FIELD(MAEDatasetConfig())
    train: MAETrainExpConfig = DATACLASS_FIELD(MAETrainExpConfig())
    model: MAEModelConfig = DATACLASS_FIELD(MAEModelConfig())
    inference: InferenceConfig = DATACLASS_FIELD(InferenceConfig())
    evaluate: EvaluateConfig = DATACLASS_FIELD(EvaluateConfig())
    gen_trt_engine: MAEGenTrtEngineConfig = DATACLASS_FIELD(MAEGenTrtEngineConfig())
    export: ExportConfig = DATACLASS_FIELD(ExportConfig())

    def __post_init__(self):
        """Set default model name for MAE."""
        if self.model_name is None:
            self.model_name = "mae"
