# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file."""

from typing import Optional
from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    DATACLASS_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    BOOL_FIELD,
)

from nvidia_tao_pytorch.config.common.common_config import (
    CommonExperimentConfig,
    EvaluateConfig,
    InferenceConfig,
    ExportConfig
)
from nvidia_tao_pytorch.config.common.quantization import ModelQuantizationConfig
from nvidia_tao_pytorch.config.depth_net.dataset import DepthNetDatasetConfig
from nvidia_tao_pytorch.config.depth_net.model import DepthNetModelConfig
from nvidia_tao_pytorch.config.depth_net.train import DepthNetTrainExpConfig
from nvidia_tao_pytorch.config.depth_net.deploy import DepthNetGenTrtEngineExpConfig
from nvidia_tao_pytorch.config.common.mlops import WandBConfig


@dataclass
class DepthNetInferenceExpConfig(InferenceConfig):
    """Inference experiment config."""

    conf_threshold: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="""The value of the confidence threshold to be used when
                    filtering out the final list of boxes.""",
        display_name="confidence threshold"
    )
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
    save_raw_pfm: Optional[bool] = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Whether to save the raw pfm output during inference.",
        display_name="Save PFM Output"
    )
    save_raw_npy: Optional[bool] = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Whether to save the raw fp32 disparity .npy alongside PNG visualization.",
        display_name="Save NPY Output"
    )


@dataclass
class DepthNetEvalExpConfig(EvaluateConfig):
    """Evaluation experiment config."""

    input_width: Optional[int] = INT_FIELD(
        value=None,
        default_value=736,
        description="Width of the input image tensor.",
        display_name="input width",
        valid_min=1,
    )
    input_height: Optional[int] = INT_FIELD(
        value=None,
        default_value=320,
        description="Height of the input image tensor.",
        display_name="input height",
        valid_min=1,
    )
    native_padded: Optional[bool] = BOOL_FIELD(
        value=False,
        default_value=False,
        description=(
            "If True, pad each image to the nearest 32-multiple and run inference "
            "at native resolution (requires a dynamic-shape TRT engine)."
        ),
        display_name="native padded",
    )


@dataclass
class DepthNetExportExpConfig(ExportConfig):
    """Inference experiment config."""

    valid_iters: Optional[int] = INT_FIELD(
        value=22,
        default_value=22,
        description="Number of GRU iterations to export the model.",
        display_name="Valid Iterations",
        valid_min=1,
    )
    dynamic_hw: Optional[bool] = BOOL_FIELD(
        value=False,
        default_value=False,
        description=(
            "If True, mark the height and width axes as dynamic in the "
            "exported ONNX so the engine can be rebuilt with arbitrary input "
            "shape via gen_trt_engine min/opt/max_height/width. Safe only "
            "for FastFoundationStereo (EdgeNeXt-only backbone, no positional "
            "embeddings); ignored with a warning for FoundationStereo and "
            "mono models because their DINOv2 backbone constant-folds the "
            "trace patch count into pos-embed shape arithmetic, producing a "
            "wrong-shape pos-embed at runtime."
        ),
        display_name="Dynamic H/W axes",
    )


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    dataset: DepthNetDatasetConfig = DATACLASS_FIELD(
        DepthNetDatasetConfig(),
        description="Configurable parameters to construct the dataset for a DepthNet experiment.",
    )
    model: DepthNetModelConfig = DATACLASS_FIELD(
        DepthNetModelConfig(),
        description="Configurable parameters to construct the model for a DepthNet experiment.",
    )
    inference: DepthNetInferenceExpConfig = DATACLASS_FIELD(
        DepthNetInferenceExpConfig(),
        description="Configurable parameters to construct the inferencer for a DepthNet experiment.",
    )
    evaluate: DepthNetEvalExpConfig = DATACLASS_FIELD(
        DepthNetEvalExpConfig(),
        description="Configurable parameters to construct the evaluator for a DepthNet experiment.",
    )
    train: DepthNetTrainExpConfig = DATACLASS_FIELD(
        DepthNetTrainExpConfig(),
        description="Configurable parameters to construct the trainer for a DepthNet experiment.",
    )
    wandb: WandBConfig = DATACLASS_FIELD(
        WandBConfig(),
        description="Configurable parameters to construct the wandb client for a DepthNet experiment.",
    )
    export: DepthNetExportExpConfig = DATACLASS_FIELD(
        DepthNetExportExpConfig(),
        description="Configurable parameters to construct the onnx export for a DepthNet experiment."
    )
    gen_trt_engine: DepthNetGenTrtEngineExpConfig = DATACLASS_FIELD(
        DepthNetGenTrtEngineExpConfig(),
        description="Configurable parameters to construct the TensorRT engine builder for a DepthNet experiment.",
    )
    quantize: ModelQuantizationConfig = DATACLASS_FIELD(
        ModelQuantizationConfig(),
        description="Configurable parameters for model quantization.",
    )

    def __post_init__(self):
        """Set default model name for DepthNet."""
        if self.model_name is None:
            self.model_name = "depth_net"
