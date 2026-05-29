# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file."""

from typing import Optional, Dict
from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    DICT_FIELD,
    FLOAT_FIELD,
    INT_FIELD
)
from nvidia_tao_pytorch.config.common.common_config import (
    CommonExperimentConfig,
    EvaluateConfig,
    ExportConfig,
    InferenceConfig
)
from nvidia_tao_pytorch.config.deformable_detr.dataset import (
    DDDatasetConfig
)
from nvidia_tao_pytorch.config.deformable_detr.deploy import DDGenTrtEngineExpConfig
from nvidia_tao_pytorch.config.deformable_detr.model import DDModelConfig
from nvidia_tao_pytorch.config.deformable_detr.train import DDTrainExpConfig
from nvidia_tao_pytorch.config.common.quantization.default_config import (
    ModelQuantizationConfig,
)


@dataclass
class DDInferenceExpConfig(InferenceConfig):
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
class DDEvalExpConfig(EvaluateConfig):
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
class DDExportExpConfig(ExportConfig):
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

    model: DDModelConfig = DATACLASS_FIELD(
        DDModelConfig(),
        description="Configurable parameters to construct the model for a Deformable DETR experiment.",
    )
    dataset: DDDatasetConfig = DATACLASS_FIELD(
        DDDatasetConfig(),
        description="Configurable parameters to construct the dataset for a Deformable DETR experiment.",
    )
    train: DDTrainExpConfig = DATACLASS_FIELD(
        DDTrainExpConfig(),
        description="Configurable parameters to construct the trainer for a Deformable DETR experiment.",
    )
    evaluate: DDEvalExpConfig = DATACLASS_FIELD(
        DDEvalExpConfig(),
        description="Configurable parameters to construct the evaluator for a Deformable DETR experiment.",
    )
    inference: DDInferenceExpConfig = DATACLASS_FIELD(
        DDInferenceExpConfig(),
        description="Configurable parameters to construct the inferencer for a Deformable DETR experiment.",
    )
    export: DDExportExpConfig = DATACLASS_FIELD(
        DDExportExpConfig(input_width=640, input_height=640),
        description="Configurable parameters to construct the exporter for a Deformable DETR experiment.",
    )
    gen_trt_engine: DDGenTrtEngineExpConfig = DATACLASS_FIELD(
        DDGenTrtEngineExpConfig(),
        description=(
            "Configurable parameters to construct the TensorRT engine builder "
            "for a Deformable DETR experiment."
        ),
    )
    quantize: ModelQuantizationConfig = DATACLASS_FIELD(
        ModelQuantizationConfig(),
        default_value={},
        description="Configurable parameters to run model quantization for a Deformable DETR experiment.",
    )

    def __post_init__(self):
        """Set default model name for Deformable DETR."""
        if self.model_name is None:
            self.model_name = "deformable_detr"
