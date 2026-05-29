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
    INT_FIELD,
)
from nvidia_tao_pytorch.config.common.common_config import (
    CommonExperimentConfig,
    EvaluateConfig,
    ExportConfig,
    InferenceConfig
)
from nvidia_tao_pytorch.config.grounding_dino.dataset import (
    GDINODatasetConfig
)
from nvidia_tao_pytorch.config.grounding_dino.deploy import GDINOGenTrtEngineExpConfig
from nvidia_tao_pytorch.config.grounding_dino.model import GDINOModelConfig
from nvidia_tao_pytorch.config.grounding_dino.train import GDINOTrainExpConfig
from nvidia_tao_pytorch.config.common.quantization.default_config import (
    ModelQuantizationConfig,
)


@dataclass
class GDINOInferenceExpConfig(InferenceConfig):
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
        default_value=960,
        description="Width of the input image tensor.",
        display_name="input width",
        valid_min=32,
    )
    input_height: Optional[int] = INT_FIELD(
        value=None,
        default_value=544,
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
class GDINOEvalExpConfig(EvaluateConfig):
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
class GDINOExportExpConfig(ExportConfig):
    """Export experiment config."""

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

    model: GDINOModelConfig = DATACLASS_FIELD(
        GDINOModelConfig(),
        description="Configurable parameters to construct the model for a Grounding DINO experiment.",
    )
    dataset: GDINODatasetConfig = DATACLASS_FIELD(
        GDINODatasetConfig(),
        description="Configurable parameters to construct the dataset for a Grounding DINO experiment.",
    )
    train: GDINOTrainExpConfig = DATACLASS_FIELD(
        GDINOTrainExpConfig(),
        description="Configurable parameters to construct the trainer for a Grounding DINO experiment.",
    )
    evaluate: GDINOEvalExpConfig = DATACLASS_FIELD(
        GDINOEvalExpConfig(),
        description="Configurable parameters to construct the evaluator for a Grounding DINO experiment.",
    )
    inference: GDINOInferenceExpConfig = DATACLASS_FIELD(
        GDINOInferenceExpConfig(),
        description="Configurable parameters to construct the inferencer for a Grounding DINO experiment.",
    )
    export: GDINOExportExpConfig = DATACLASS_FIELD(
        GDINOExportExpConfig(),
        description="Configurable parameters to construct the exporter for a Grounding DINO experiment.",
    )
    gen_trt_engine: GDINOGenTrtEngineExpConfig = DATACLASS_FIELD(
        GDINOGenTrtEngineExpConfig(),
        description="Configurable parameters to construct the TensorRT engine builder for a Grounding DINO experiment.",
    )
    quantize: ModelQuantizationConfig = DATACLASS_FIELD(
        ModelQuantizationConfig(),
        default_value={},
        description="Configurable parameters to run model quantization for a Grounding DINO experiment.",
    )

    def __post_init__(self):
        """Set default model name for Grounding DINO."""
        if self.model_name is None:
            self.model_name = "grounding_dino"
