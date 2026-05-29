# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file"""

from typing import Optional, List
from dataclasses import dataclass
from omegaconf import MISSING

from nvidia_tao_pytorch.config.common.common_config import (
    EvaluateConfig,
    CommonExperimentConfig,
    InferenceConfig,
    TrainConfig
)
from nvidia_tao_pytorch.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    BOOL_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class OptimGenConfig:
    """Optimizer config."""

    optim: str = STR_FIELD(value="Adam")
    lr: float = FLOAT_FIELD(
        value=0.0025,
        valid_min=0,
        valid_max="inf",
        automl_enabled="TRUE"
    )
    eps: float = FLOAT_FIELD(value=1e-8)
    betas: List[float] = LIST_FIELD(arrList=[0, 0.99])


@dataclass
class OptimDiscConfig:
    """Optimizer config."""

    optim: str = STR_FIELD(value="Adam")
    lr: float = FLOAT_FIELD(
        value=0.002,
        valid_min=0,
        valid_max="inf",
        automl_enabled="TRUE"
    )
    eps: float = FLOAT_FIELD(value=1e-8)
    betas: List[float] = LIST_FIELD(arrList=[0, 0.99])


@dataclass
class LossConfig:
    """Configuration parameters for Loss."""

    cls_weight: float = FLOAT_FIELD(0.0)


@dataclass
class StemConfig:
    """Configuration parameters for Generator's stem backbone."""

    fp32: bool = BOOL_FIELD(value=False)
    cbase: int = INT_FIELD(value=32768)
    cmax: int = INT_FIELD(value=512)
    syn_layers: int = INT_FIELD(value=10)
    resolution: int = INT_FIELD(value=128)


@dataclass
class AddedHeadSupperresConfig:
    """Configuration parameters for Generator's super resolution backbone."""

    head_layers: Optional[List[int]] = LIST_FIELD(arrList=[7])
    up_factor: Optional[List[int]] = LIST_FIELD(arrList=[2])
    pretrained_stem_path: Optional[str] = STR_FIELD(value=None)
    reinit_stem_anyway: bool = BOOL_FIELD(value=True)
    train_head_only: bool = BOOL_FIELD(value=True)


@dataclass
class GeneratorConfig:
    """Configuration parameters for Generator (shared with both StyleGAN and BigDatasetGAN)."""

    backbone: str = STR_FIELD(
        value="stylegan3-r",
        display_name="Backbone architectures",
        valid_options="stylegan3-t,stylegan3-r,stylegan2,fastgan"
    )
    superres: bool = BOOL_FIELD(value=False)
    added_head_superres: AddedHeadSupperresConfig = DATACLASS_FIELD(AddedHeadSupperresConfig())
    stem: StemConfig = DATACLASS_FIELD(StemConfig())


@dataclass
class DiscriminatorConfig:
    """Configuration parameters for Discriminator (for StyleGAN)."""

    backbones: List[str] = LIST_FIELD(
        [
            "deit_base_distilled_patch16_224",
            "tf_efficientnet_lite0"
        ]
    )


@dataclass
class MetricsConfig:
    """Configuration parameters for Discriminator (for StyleGAN)."""

    num_fake_imgs: int = INT_FIELD(
        value=50000,
        valid_min=0,
        valid_max=50000
    )
    inception_fid_path: Optional[str] = STR_FIELD(value=None)


@dataclass
class ModelStyleganConfig:
    """StyleGAN Model config."""

    loss: LossConfig = DATACLASS_FIELD(LossConfig())
    discriminator: DiscriminatorConfig = DATACLASS_FIELD(DiscriminatorConfig())
    metrics: MetricsConfig = DATACLASS_FIELD(MetricsConfig())


@dataclass
class FeatureExtractorConfig:
    """Configuration parameters for Feature Extractor (for BigDatasetGAN)."""

    stylegan_checkpoint_path: str = STR_FIELD(value=MISSING, value_type="hidden")
    blocks: List[int] = LIST_FIELD(arrList=[2, 6, 11, 15])


@dataclass
class ModelBigdatasetganConfig:
    """BigDatasetGAN Model config."""

    feature_extractor: FeatureExtractorConfig = DATACLASS_FIELD(FeatureExtractorConfig())


@dataclass
class ModelConfig:
    """Model config."""

    loss: LossConfig = DATACLASS_FIELD(LossConfig())
    generator: GeneratorConfig = DATACLASS_FIELD(GeneratorConfig())
    input_embeddings_path: Optional[str] = STR_FIELD(value=None)
    stylegan: ModelStyleganConfig = DATACLASS_FIELD(ModelStyleganConfig())
    bigdatasetgan: ModelBigdatasetganConfig = DATACLASS_FIELD(ModelBigdatasetganConfig())


@dataclass
class DataPathFormat:
    """Dataset Path experiment config."""

    images_dir: str = STR_FIELD(value=MISSING)


@dataclass
class SeedDatasetConfig:
    """Seed Dataset Config."""

    start_seed: int = INT_FIELD(value=0, valid_min=0, valid_max="inf")
    end_seed: int = INT_FIELD(value=100, valid_min=0, valid_max="inf")


@dataclass
class StyleganDatasetConfig:
    """StyleGAN Dataset Config."""

    train_dataset: DataPathFormat = DATACLASS_FIELD(DataPathFormat())
    validation_dataset: DataPathFormat = DATACLASS_FIELD(DataPathFormat())
    test_dataset: DataPathFormat = DATACLASS_FIELD(DataPathFormat())
    infer_dataset: SeedDatasetConfig = DATACLASS_FIELD(SeedDatasetConfig())
    batch_gpu_size: int = INT_FIELD(value=16, valid_min=1, valid_max="inf")
    mirror: bool = BOOL_FIELD(value=True)


@dataclass
class CommonDatasetConfig:
    """Common Dataset Config."""

    cond: bool = BOOL_FIELD(value=False)
    img_resolution: int = INT_FIELD(value=128)
    img_channels: int = INT_FIELD(value=3)
    num_classes: int = INT_FIELD(value=0)


@dataclass
class BigdatasetganDatasetConfig:
    """BigDatasetGAN Dataset Config."""

    train_dataset: DataPathFormat = DATACLASS_FIELD(DataPathFormat())
    validation_dataset: DataPathFormat = DATACLASS_FIELD(DataPathFormat())
    test_dataset: DataPathFormat = DATACLASS_FIELD(DataPathFormat())
    infer_dataset: SeedDatasetConfig = DATACLASS_FIELD(SeedDatasetConfig())
    class_idx: int = INT_FIELD(value=0, valid_min=0, valid_max="inf")
    seg_classes: int = INT_FIELD(value=2)


@dataclass
class DatasetConfig:
    """Dataset Config."""

    stylegan: StyleganDatasetConfig = DATACLASS_FIELD(StyleganDatasetConfig())
    bigdatasetgan: BigdatasetganDatasetConfig = DATACLASS_FIELD(BigdatasetganDatasetConfig())
    common: CommonDatasetConfig = DATACLASS_FIELD(CommonDatasetConfig())
    batch_size: int = INT_FIELD(value=64, valid_min=1, valid_max="inf")
    pin_memory: bool = BOOL_FIELD(value=True)
    prefetch_factor: int = INT_FIELD(value=2, valid_min=1, valid_max="inf")
    workers: int = INT_FIELD(value=3, valid_min=1, valid_max="inf")


@dataclass
class TensorBoardLogger:
    """Configuration for the tensorboard logger."""

    enabled: bool = BOOL_FIELD(value=False)
    infrequent_logging_frequency: int = INT_FIELD(
        value=2,
        valid_min=0,
        valid_max="inf"
    )  # Defined per epoch


@dataclass
class TrainStyleganConfig:
    """StyleGAN Train Config."""

    gan_seed_offset: int = INT_FIELD(value=0)
    optim_generator: OptimGenConfig = DATACLASS_FIELD(OptimGenConfig())
    optim_discriminator: OptimDiscConfig = DATACLASS_FIELD(OptimDiscConfig())


@dataclass
class TrainBigdatasetganConfig:
    """BigDatasetGAN Train Config."""

    optim_labeller: OptimGenConfig = DATACLASS_FIELD(OptimGenConfig())


@dataclass
class TrainExpConfig(TrainConfig):
    """Train Config."""

    deterministic_all: bool = BOOL_FIELD(value=False)
    pretrained_model_path: Optional[str] = STR_FIELD(value=None)
    stylegan: TrainStyleganConfig = DATACLASS_FIELD(TrainStyleganConfig())
    bigdatasetgan: TrainBigdatasetganConfig = DATACLASS_FIELD(TrainBigdatasetganConfig())
    tensorboard: Optional[TensorBoardLogger] = DATACLASS_FIELD(TensorBoardLogger())


@dataclass
class EvalExpConfig(EvaluateConfig):
    """Evaluation experiment config."""

    vis_after_n_batches: int = INT_FIELD(value=16, valid_min=1, valid_max="inf")
    trt_engine: str = STR_FIELD(value=MISSING)


@dataclass
class InferenceExpConfig(InferenceConfig):
    """Inference experiment config."""

    vis_after_n_batches: int = INT_FIELD(value=1, valid_min=1, valid_max="inf")
    trt_engine: str = STR_FIELD(value=MISSING)
    # Image generation recipe
    truncation_psi: float = FLOAT_FIELD(value=1.0, valid_min=0, valid_max=1.0)
    translate: List[float] = LIST_FIELD(arrList=[0.0, 0.0])
    rotate: float = FLOAT_FIELD(value=0, valid_min=0.0, valid_max=360.0)
    centroids_path: Optional[str] = STR_FIELD(value=None)
    class_idx: int = INT_FIELD(value=0, valid_min=0, valid_max="inf")


@dataclass
class ExportRuntimeConfig:
    """Export experiment config."""

    test_onnxruntime: bool = BOOL_FIELD(value=True)
    sample_result_dir: Optional[str] = STR_FIELD(value=None, value_type="hidden")
    runtime_seed: int = INT_FIELD(value=0)
    runtime_batch_size: int = INT_FIELD(value=1)
    runtime_class_dix: int = INT_FIELD(value=0)


@dataclass
class ExportExpConfig:
    """Export experiment config."""

    results_dir: Optional[str] = STR_FIELD(value=None, value_type="hidden")
    gpu_id: int = INT_FIELD(value=0)
    checkpoint: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="The absolute path to the checkpoint.",
        display="checkpoint"
    )
    onnx_file: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="The absolute path to the onnx file.",
        display="onnx file"
    )
    on_cpu: bool = BOOL_FIELD(value=False)
    opset_version: int = INT_FIELD(value=12)
    batch_size: int = INT_FIELD(value=-1)
    verbose: bool = BOOL_FIELD(value=False)
    onnxruntime: ExportRuntimeConfig = DATACLASS_FIELD(ExportRuntimeConfig())


@dataclass
class CalibrationConfig:
    """Calibration config."""

    cal_image_dir: List[str] = LIST_FIELD(MISSING)
    cal_cache_file: str = STR_FIELD(value=MISSING)
    cal_batch_size: int = INT_FIELD(value=1)
    cal_batches: int = INT_FIELD(value=1)


@dataclass
class TrtConfig:
    """Trt config."""

    data_type: str = STR_FIELD(value="FP32")
    workspace_size: int = INT_FIELD(value=1024)
    min_batch_size: int = INT_FIELD(value=1)
    opt_batch_size: int = INT_FIELD(value=1)
    max_batch_size: int = INT_FIELD(value=1)
    calibration: CalibrationConfig = DATACLASS_FIELD(CalibrationConfig())


@dataclass
class GenTrtEngineExpConfig:
    """Gen TRT Engine experiment config."""

    results_dir: Optional[str] = STR_FIELD(value=None)
    gpu_id: int = INT_FIELD(value=0)
    onnx_file: str = STR_FIELD(value=MISSING)
    trt_engine: Optional[str] = STR_FIELD(value=None)
    input_channel: int = INT_FIELD(value=3)
    input_width: int = INT_FIELD(value=128)
    input_height: int = INT_FIELD(value=128)
    opset_version: int = INT_FIELD(value=12)
    batch_size: int = INT_FIELD(value=-1)
    verbose: bool = BOOL_FIELD(value=False)
    tensorrt: TrtConfig = DATACLASS_FIELD(TrtConfig())


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    task: str = STR_FIELD(value="stylegan", valid_options="stylegan,bigdatasetgan")
    model: ModelConfig = DATACLASS_FIELD(ModelConfig())
    dataset: DatasetConfig = DATACLASS_FIELD(DatasetConfig())
    train: TrainExpConfig = DATACLASS_FIELD(TrainExpConfig())
    evaluate: EvalExpConfig = DATACLASS_FIELD(EvalExpConfig())
    inference: InferenceExpConfig = DATACLASS_FIELD(InferenceExpConfig())
    export: ExportExpConfig = DATACLASS_FIELD(ExportExpConfig())
    gen_trt_engine: GenTrtEngineExpConfig = DATACLASS_FIELD(GenTrtEngineExpConfig())

    def __post_init__(self):
        """Set default model name for StyleGAN XL."""
        if self.model_name is None:
            self.model_name = "stylegan_xl"
