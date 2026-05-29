# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file."""

from typing import Optional, Dict, List
from dataclasses import dataclass
from omegaconf import MISSING

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    DICT_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    STR_FIELD,
    LIST_FIELD
)
from nvidia_tao_pytorch.config.common.common_config import (
    CommonExperimentConfig,
    EvaluateConfig,
    ExportConfig,
    InferenceConfig
)
from nvidia_tao_pytorch.config.common.distillation_config import (
    DistillationBindingConfig,
    DistillationConfig
)
from nvidia_tao_pytorch.config.rtdetr.dataset import (
    RTDatasetConfig
)
from nvidia_tao_pytorch.config.rtdetr.deploy import RTGenTrtEngineExpConfig
from nvidia_tao_pytorch.config.rtdetr.model import RTModelConfig
from nvidia_tao_pytorch.config.rtdetr.train import RTTrainExpConfig
from nvidia_tao_pytorch.config.common.quantization.default_config import (
    ModelQuantizationConfig,
)


@dataclass
class RTDistillationConfig(DistillationConfig):
    """Distillation config"""

    teacher: RTModelConfig = DATACLASS_FIELD(
        RTModelConfig(),
        descripton="Configuration hyper parameters for the RTDETR based teacher model.",
        display_name="teacher"
    )
    pretrained_teacher_model_path: Optional[str] = STR_FIELD(
        value=MISSING,
        display_name="Pretrained teacher model path",
        description="Path to the pre-trained teacher model."
    )
    bindings: List[DistillationBindingConfig] = LIST_FIELD(
        arrList=[],
        default_value=[],
        description=(
            "List of bindings for distillation. Each element is an instance of "
            "RTModelDistillationBindingConfig."
        ),
        display_name="bindings"
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        display_name="Results directory",
        description="""
        Path to where all the assets generated from a task are stored.
        """
    )


@dataclass
class RTInferenceExpConfig(InferenceConfig):
    """Inference experiment config."""

    color_map: Optional[Dict[str, str]] = DICT_FIELD(
        hashMap=None,
        description="Class-wise dictionary with colors to render boxes.",
        display_name="color map"
    )
    conf_threshold: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="""The value of the confidence threshold to be used when
                    filtering out the final list of boxes.""",
        display_name="confidence threshold"
    )
    is_internal: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="is internal",
        description="Flag to render with internal directory structure."
    )
    input_width: Optional[int] = INT_FIELD(
        value=None,
        default_value=640,
        description="Width of the input image tensor.",
        display_name="input width",
        valid_min=32,
    )
    input_height: Optional[int] = INT_FIELD(
        value=None,
        default_value=640,
        description="Height of the input image tensor.",
        display_name="input height",
        valid_min=32,
    )
    outline_width: int = INT_FIELD(
        value=3,
        default_value=3,
        description="Width in pixels of the bounding box outline.",
        display_name="outline width",
        valid_min=1,
    )
    is_quantized: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to indicate if the model is quantized",
        display_name="Flag to indicate if the model is quantized"
    )


@dataclass
class RTEvalExpConfig(EvaluateConfig):
    """Evaluation experiment config."""

    input_width: Optional[int] = INT_FIELD(
        value=None,
        description="Width of the input image tensor.",
        display_name="input width",
        valid_min=1,
    )
    input_height: Optional[int] = INT_FIELD(
        value=None,
        description="Height of the input image tensor.",
        display_name="input height",
        valid_min=1,
    )
    conf_threshold: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        description="""The value of the confidence threshold to be used when
                    filtering out the final list of boxes.""",
        display_name="confidence threshold"
    )
    is_quantized: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to indicate if the model is quantized",
        display_name="Flag to indicate if the model is quantized"
    )


@dataclass
class RTExportExpConfig(ExportConfig):
    """Export configration schema for RT-DETR."""

    serialize_nvdsinfer: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Serialize DeepStream config.",
        description="""Flag to enable serializing the required
                    configs for integrating with DeepStream."""
    )
    is_quantized: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Flag to indicate if the model is quantized",
        display_name="Flag to indicate if the model is quantized"
    )


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: RTModelConfig = DATACLASS_FIELD(
        RTModelConfig(),
        description="Configurable parameters to construct the model for a RT-DETR experiment.",
    )
    dataset: RTDatasetConfig = DATACLASS_FIELD(
        RTDatasetConfig(),
        description="Configurable parameters to construct the dataset for a RT-DETR experiment.",
    )
    train: RTTrainExpConfig = DATACLASS_FIELD(
        RTTrainExpConfig(),
        description="Configurable parameters to construct the trainer for a RT-DETR experiment.",
    )
    evaluate: RTEvalExpConfig = DATACLASS_FIELD(
        RTEvalExpConfig(),
        description="Configurable parameters to construct the evaluator for a RT-DETR experiment.",
    )
    inference: RTInferenceExpConfig = DATACLASS_FIELD(
        RTInferenceExpConfig(),
        description="Configurable parameters to construct the inferencer for a RT-DETR experiment.",
    )
    export: RTExportExpConfig = DATACLASS_FIELD(
        RTExportExpConfig(input_width=640, input_height=640),
        description="Configurable parameters to construct the exporter for a RT-DETR experiment.",
    )
    gen_trt_engine: RTGenTrtEngineExpConfig = DATACLASS_FIELD(
        RTGenTrtEngineExpConfig(),
        description="Configurable parameters to construct the TensorRT engine builder for a RT-DETR experiment.",
    )
    distill: Optional[RTDistillationConfig] = DATACLASS_FIELD(
        None,
        description="Configurable parameters to construct the distiller for a RT-DETR experiment.",
    )
    quantize: ModelQuantizationConfig = DATACLASS_FIELD(
        ModelQuantizationConfig(),
        default_value={},
        description="Configurable parameters to run model quantization for a RT-DETR experiment.",
    )

    def __post_init__(self):
        """Set default model name for RT-DETR."""
        if self.model_name is None:
            self.model_name = "rtdetr"
