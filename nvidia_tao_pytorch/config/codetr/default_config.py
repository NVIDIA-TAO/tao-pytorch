# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file for CoDETR."""

from typing import Optional, Dict, List
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
    InferenceConfig,
)
# Reuse DINO's dataset, train, and deploy configs directly
from nvidia_tao_pytorch.config.dino.dataset import DINODatasetConfig
from nvidia_tao_pytorch.config.dino.deploy import DINOGenTrtEngineExpConfig
from nvidia_tao_pytorch.config.dino.train import DINOTrainExpConfig
from nvidia_tao_pytorch.config.codetr.model import CoDETRModelConfig
from nvidia_tao_pytorch.config.common.quantization.default_config import ModelQuantizationConfig


@dataclass
class CoDETRInferenceExpConfig(InferenceConfig):
    """CoDETR inference config."""

    color_map: Optional[Dict[str, str]] = DICT_FIELD(
        hashMap=None,
        description="Class-wise color map for bounding box rendering.",
        display_name="color map",
    )
    conf_threshold: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="Confidence threshold for filtering predictions.",
        display_name="confidence threshold",
    )
    is_internal: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="is internal",
        description="Flag for internal directory structure rendering.",
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
    save_annotated_images: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="save annotated images",
        description="If True, write annotated JPEGs alongside KITTI label files. "
                    "Set to False to write only label files (faster, no image decode/encode).",
    )
    category_mapping: Optional[Dict[str, List[str]]] = DICT_FIELD(
        hashMap=None,
        display_name="category mapping",
        description=(
            "Optional grouping of original classmap categories into output "
            "categories applied AFTER the model forward and BEFORE writing "
            "labels/visualizations. Example: "
            "{'bicycle': ['bicycle', 'motorcycle'], 'car': ['car', 'bus', 'truck']}. "
            "Detections whose original class is not present in any group are "
            "dropped. When soft-NMS is enabled, an additional per-output-category "
            "soft-NMS pass runs so duplicates within a merged group are suppressed."
        ),
    )


@dataclass
class CoDETREvalExpConfig(EvaluateConfig):
    """CoDETR evaluation config."""

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
        description="Confidence threshold for filtering predictions.",
        display_name="confidence threshold",
    )


@dataclass
class CoDETRExportExpConfig(ExportConfig):
    """CoDETR export config."""

    serialize_nvdsinfer: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="serialize DeepStream config",
        description="Flag to serialize configs for DeepStream integration.",
    )
    on_cpu: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="export on CPU",
        description="Run export on CPU instead of GPU.",
    )
    verbose: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="verbose export",
        description="Enable verbose ONNX export logging.",
    )


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """CoDETR experiment config."""

    model: CoDETRModelConfig = DATACLASS_FIELD(
        CoDETRModelConfig(),
        description="CoDETR model hyperparameters.",
    )
    dataset: DINODatasetConfig = DATACLASS_FIELD(
        DINODatasetConfig(),
        description="Dataset configuration (same format as DINO).",
    )
    train: DINOTrainExpConfig = DATACLASS_FIELD(
        DINOTrainExpConfig(),
        description="Training hyperparameters.",
    )
    evaluate: CoDETREvalExpConfig = DATACLASS_FIELD(
        CoDETREvalExpConfig(),
        description="Evaluation parameters.",
    )
    inference: CoDETRInferenceExpConfig = DATACLASS_FIELD(
        CoDETRInferenceExpConfig(),
        description="Inference parameters.",
    )
    export: CoDETRExportExpConfig = DATACLASS_FIELD(
        CoDETRExportExpConfig(input_width=640, input_height=640),
        description="ONNX export parameters.",
    )
    gen_trt_engine: DINOGenTrtEngineExpConfig = DATACLASS_FIELD(
        DINOGenTrtEngineExpConfig(),
        description="TensorRT engine generation parameters.",
    )
    quantize: ModelQuantizationConfig = DATACLASS_FIELD(
        ModelQuantizationConfig(),
        default_value={},
        description="Model quantization parameters.",
    )

    def __post_init__(self):
        """Set default model name."""
        if self.model_name is None:
            self.model_name = "codetr"
