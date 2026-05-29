# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file."""

from typing import Optional, List, Dict
from dataclasses import dataclass
from omegaconf import MISSING

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    DICT_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD
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
from nvidia_tao_pytorch.config.dino.dataset import (
    DINODatasetConfig
)
from nvidia_tao_pytorch.config.dino.deploy import DINOGenTrtEngineExpConfig
from nvidia_tao_pytorch.config.dino.model import DINOModelConfig
from nvidia_tao_pytorch.config.dino.train import DINOTrainExpConfig
from nvidia_tao_pytorch.config.common.quantization.default_config import (
    ModelQuantizationConfig,
)


@dataclass
class DINOModelDistillationBindingConfig(DistillationBindingConfig):
    """Distillation binding config."""

    pass


@dataclass
class DINODistillationConfig(DistillationConfig):
    """Distillation config"""

    teacher: DINOModelConfig = DATACLASS_FIELD(
        DINOModelConfig(),
        descripton="Configuration hyper parameters for the DINO based teacher model.",
        display_name="teacher"
    )
    pretrained_teacher_model_path: Optional[str] = STR_FIELD(
        value=MISSING,
        display_name="Pretrained teacher model path",
        description="Path to the pre-trained teacher model."
    )
    bindings: List[DINOModelDistillationBindingConfig] = LIST_FIELD(
        arrList=[],
        default_value=[],
        description=(
            "List of bindings for Distillation. Each element is an instance of DINOModelDistillationBindingConfig."
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
class DINOInferenceExpConfig(InferenceConfig):
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


@dataclass
class DINOEvalExpConfig(EvaluateConfig):
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


@dataclass
class DINOExportExpConfig(ExportConfig):
    """Structured configuration schema for Deformable DETR export."""

    serialize_nvdsinfer: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="Serialize DeepStream config.",
        description="""Flag to enable serializing the required
                    configs for integrating with DeepStream."""
    )


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: DINOModelConfig = DATACLASS_FIELD(
        DINOModelConfig(),
        description="Configurable parameters to construct the model for a DINO experiment.",
    )
    dataset: DINODatasetConfig = DATACLASS_FIELD(
        DINODatasetConfig(),
        description="Configurable parameters to construct the dataset for a DINO experiment.",
    )
    train: DINOTrainExpConfig = DATACLASS_FIELD(
        DINOTrainExpConfig(),
        description="Configurable parameters to construct the trainer for a DINO experiment.",
    )
    evaluate: DINOEvalExpConfig = DATACLASS_FIELD(
        DINOEvalExpConfig(),
        description="Configurable parameters to construct the evaluator for a DINO experiment.",
    )
    inference: DINOInferenceExpConfig = DATACLASS_FIELD(
        DINOInferenceExpConfig(),
        description="Configurable parameters to construct the inferencer for a DINO experiment.",
    )
    export: DINOExportExpConfig = DATACLASS_FIELD(
        DINOExportExpConfig(input_width=640, input_height=640),
        description="Configurable parameters to construct the exporter for a DINO experiment.",
    )
    gen_trt_engine: DINOGenTrtEngineExpConfig = DATACLASS_FIELD(
        DINOGenTrtEngineExpConfig(),
        description="Configurable parameters to construct the TensorRT engine builder for a DINO experiment.",
    )
    distill: Optional[DINODistillationConfig] = DATACLASS_FIELD(
        None,
        description="Configurable parameters to construct the distiller for a DINO experiment.",
    )
    quantize: ModelQuantizationConfig = DATACLASS_FIELD(
        ModelQuantizationConfig(),
        default_value={},
        description="Configurable parameters to run model quantization for a DINO experiment.",
    )

    def __post_init__(self):
        """Set default model name for DINO."""
        if self.model_name is None:
            self.model_name = "dino"
