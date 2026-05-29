# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file."""

from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD,
    DATACLASS_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    STR_FIELD
)
from nvidia_tao_pytorch.config.common.common_config import (
    CommonExperimentConfig,
    EvaluateConfig,
    InferenceConfig
)
from nvidia_tao_pytorch.config.centerpose.dataset import CenterPoseDatasetConfig
from nvidia_tao_pytorch.config.centerpose.deploy import CenterPoseGenTrtEngineExpConfig
from nvidia_tao_pytorch.config.centerpose.model import CenterPoseModelConfig
from nvidia_tao_pytorch.config.centerpose.train import CenterPoseTrainExpConfig


@dataclass
class CenterPoseInferenceExpConfig(InferenceConfig):
    """Inference experiment config."""

    visualization_threshold: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        valid_min=0.3,
        valid_max=0.3,
        description="Visualization threshold.",
        display_name="Visualization Threshold",
        popular="0.3"
    )
    num_select: int = INT_FIELD(
        value=100,
        default_value=100,
        valid_min=100,
        valid_max=100,
        description="Number of selected objects.",
        display_name="Number of Selected Objects",
        popular="100"
    )
    use_pnp: bool = BOOL_FIELD(
        value=True,
        description="Use PnP.",
        display_name="PnP"
    )
    save_json: bool = BOOL_FIELD(
        value=True,
        description="Save JSON file to local.",
        display_name="Save JSON"
    )
    save_visualization: bool = BOOL_FIELD(
        value=True,
        description="Save visualization image to local.",
        display_name="Save Visualization"
    )
    opencv: bool = BOOL_FIELD(
        value=True,
        description="Use OpenCV for visualization.",
        display_name="UseOpenCV"
    )

    # Camera intrinsic matrix
    principle_point_x: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max="inf",
        description="Intrinsic matrix principle point x.",
        display_name="Principle Point X"
    )
    principle_point_y: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max="inf",
        description="Intrinsic matrix principle point y.",
        display_name="Principle Point Y"
    )
    focal_length_x: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max="inf",
        description="Intrinsic matrix focal length x.",
        display_name="Focal Length X"
    )
    focal_length_y: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max="inf",
        description="Intrinsic matrix focal length y.",
        display_name="Focal Length Y"
    )
    skew: float = FLOAT_FIELD(
        value=0.0,
        default_value=0.0,
        valid_min=0.0,
        valid_max="inf",
        description="Intrinsic matrix Skew.",
        display_name="Skew"
    )
    axis_size: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        valid_min=0.1,
        valid_max="inf",
        description="Axis size setting.",
        display_name="Axis Size",
        popular="0.5"
    )


@dataclass
class CenterPoseExportExpConfig:
    """Export experiment config."""

    results_dir: str = STR_FIELD(
        value="",
        description="Results directory.",
        display_name="Results Directory"
    )
    gpu_id: int = INT_FIELD(
        value=0,
        default_value=0,
        description="GPU ID used for training.",
        display_name="GPU ID"
    )
    checkpoint: str = STR_FIELD(
        value="",
        description="Path to the checkpoint.",
        display_name="Checkpoint"
    )
    onnx_file: str = STR_FIELD(
        value="",
        description="Path to the ONNX file.",
        display_name="ONNX File"
    )
    on_cpu: bool = BOOL_FIELD(
        value=False,
        description="Export the ONNX using CPU only.",
        display_name="ONNX_CPU"
    )
    input_channel: int = INT_FIELD(
        value=3,
        default_value=3,
        valid_min=3,
        valid_max=3,
        description="Input channel.",
        display_name="Input Channel"
    )
    input_width: int = INT_FIELD(
        value=512,
        default_value=512,
        valid_min=512,
        valid_max=512,
        description="Input width.",
        display_name="Input Width"
    )
    input_height: int = INT_FIELD(
        value=512,
        default_value=512,
        valid_min=512,
        valid_max=512,
        description="Input height.",
        display_name="Input Height"
    )
    opset_version: int = INT_FIELD(
        value=16,
        default_value=16,
        valid_min=16,
        valid_max=16,
        description="ONNX opset version.",
        display_name="Opset Version",
        popular="16"
    )
    batch_size: int = INT_FIELD(
        value=-1,
        default_value=-1,
        description="ONNX model batch size (-1: dynamic).",
        display_name="ONNX Batch Size"
    )
    verbose: bool = BOOL_FIELD(
        value=False,
        description="Verbose mode.",
        display_name="Verbose"
    )
    num_select: int = INT_FIELD(
        value=100,
        default_value=100,
        valid_min=100,
        valid_max=100,
        description="Number of selected objects.",
        display_name="Number of Selected Objects"
    )
    do_constant_folding: bool = BOOL_FIELD(
        value=True,
        description="Do constant folding on ONNX model.",
        display_name="Constant Folding"
    )


@dataclass
class CenterPoseEvalExpConfig(EvaluateConfig):
    """Inference experiment config."""

    opencv: bool = BOOL_FIELD(
        value=True,
        description="Use OpenCV for visualization.",
        display_name="UseOpenCV"
    )
    eval_num_symmetry: int = INT_FIELD(
        value=1,
        default_value=1,
        valid_min=3,
        valid_max="inf",
        description="Number of the object symmetries used for evaluation.",
        display_name="Number of Symmetries for Eval",
        popular="1"
    )


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    dataset: CenterPoseDatasetConfig = DATACLASS_FIELD(
        CenterPoseDatasetConfig(),
        description="Configurable parameters to construct the dataset for a CenterPose experiment.",
    )
    train: CenterPoseTrainExpConfig = DATACLASS_FIELD(
        CenterPoseTrainExpConfig(),
        description="Configurable parameters to train the CenterPose model.",
    )
    model: CenterPoseModelConfig = DATACLASS_FIELD(
        CenterPoseModelConfig(),
        description="Configurable parameters to build the CenterPose model.",
    )
    inference: CenterPoseInferenceExpConfig = DATACLASS_FIELD(
        CenterPoseInferenceExpConfig(),
        description="Configurable parameters to run the CenterPose inference.",
    )
    export: CenterPoseExportExpConfig = DATACLASS_FIELD(
        CenterPoseExportExpConfig(),
        description="Configurable parameters to export the CenterPose ONNX model.",
    )
    evaluate: CenterPoseEvalExpConfig = DATACLASS_FIELD(
        CenterPoseEvalExpConfig(),
        description="Configurable parameters to evaluate the CenterPose model.",
    )
    gen_trt_engine: CenterPoseGenTrtEngineExpConfig = DATACLASS_FIELD(
        CenterPoseGenTrtEngineExpConfig(),
        description="Configurable parameters to generate TensorRT engine.",
    )

    def __post_init__(self):
        """Set default model name for CenterPose."""
        if self.model_name is None:
            self.model_name = "centerpose"
