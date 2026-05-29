# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file."""

from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    DATACLASS_FIELD
)
from nvidia_tao_pytorch.config.common.common_config import (
    CommonExperimentConfig
)
from nvidia_tao_pytorch.config.common.quantization import ModelQuantizationConfig

from nvidia_tao_pytorch.config.oneformer.dataset import OneFormerDatasetConfig
from nvidia_tao_pytorch.config.oneformer.model import OneFormerModelConfig
from nvidia_tao_pytorch.config.oneformer.train import OneFormerTrainExpConfig
from nvidia_tao_pytorch.config.oneformer.export import OneFormerExportExpConfig
from nvidia_tao_pytorch.config.oneformer.evaluate import OneFormerEvaluateConfig
from nvidia_tao_pytorch.config.oneformer.inference import OneFormerInferenceConfig
from nvidia_tao_pytorch.config.oneformer.deploy import OneFormerGenTrtEngineExpConfig


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: OneFormerModelConfig = DATACLASS_FIELD(
        OneFormerModelConfig(),
    )
    dataset: OneFormerDatasetConfig = DATACLASS_FIELD(
        OneFormerDatasetConfig(),
    )
    train: OneFormerTrainExpConfig = DATACLASS_FIELD(
        OneFormerTrainExpConfig(),
        description="Configurable parameters to construct the trainer for a OneFormer experiment.",
    )
    evaluate: OneFormerEvaluateConfig = DATACLASS_FIELD(
        OneFormerEvaluateConfig(),
        description="Configurable parameters to construct the evaluator for a OneFormer experiment.",
    )
    inference: OneFormerInferenceConfig = DATACLASS_FIELD(
        OneFormerInferenceConfig(),
        description="Configurable parameters to construct the inference for a OneFormer experiment.",
    )
    export: OneFormerExportExpConfig = DATACLASS_FIELD(
        OneFormerExportExpConfig(),
        description="Configurable parameters to construct the exporter for a OneFormer checkpoint.",
    )
    gen_trt_engine: OneFormerGenTrtEngineExpConfig = DATACLASS_FIELD(
        OneFormerGenTrtEngineExpConfig(),
        description="Configurable parameters to construct the deployer for a OneFormer checkpoint.",
    )
    quantize: ModelQuantizationConfig = DATACLASS_FIELD(
        ModelQuantizationConfig(),
        description="Configurable parameters to run model quantization for a OneFormer experiment.",
    )
