# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file."""

from typing import Optional, Dict, Any
from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    BOOL_FIELD,
    DICT_FIELD,
    DATACLASS_FIELD
)
from nvidia_tao_pytorch.config.common.common_config import (
    CommonExperimentConfig,
    EvaluateConfig,
    InferenceConfig
)
from nvidia_tao_pytorch.config.bevfusion.dataset import BEVFusionDatasetExpConfig
from nvidia_tao_pytorch.config.bevfusion.model import BEVFusionModelConfig
from nvidia_tao_pytorch.config.bevfusion.train import BEVFusionTrainExpConfig


@dataclass
class BEVFusionDataConvertExpConfig:
    """Configuration parameters for Data Converter"""

    dataset: str = STR_FIELD(
        value="kitti",
        default_value="kitti",
        display_name="Dataset Name",
        description="Dataset name for 3D Fusion",
        valid_options=",".join(["kitti", "tao3d"])
    )
    root_dir: str = STR_FIELD(
        value="",
        default_value="",
        display_name="root directory of the dataset",
        description="A path to the root directory of the given dataset."
    )
    results_dir: str = STR_FIELD(
        value="",
        default_value="",
        display_name="results directory",
        description="A directory to save data convert output."
    )
    output_prefix: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="output prefix",
        description="Output prefix to append for output pkl file."
    )
    mode: str = STR_FIELD(
        value="training",
        default_value="training",
        display_name="data convert mode",
        description="Data mode to generate output pkl file",
        valid_options=",".join(["training", "validation", "testing"])
    )
    with_plane: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="is with plane",
        description="Whether to use plane data from kitti."
    )
    per_sequence: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="is per sequence",
        description="Whether to save results in per sequence format."
    )
    is_synthetic: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="is synthetic",
        description="Whether data is generated synthetically from Omniverse or not."
    )
    dimension_order: str = STR_FIELD(
        value='hwl',
        default_value='hwl',
        display_name="3D dimension order",
        description="3D ground truth dimension order."
    )
    merge_only: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="merge only",
        description="Whether to merge only per sequence pkl without generating per seuqence pkl."
    )
    sequence_list: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="sequence list",
        description="Sequence list to process per sequence."
    )


@dataclass
class BEVFusionInferenceExpConfig(InferenceConfig):
    """Inference experiment config."""

    conf_threshold: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        display_name="conf threshold",
        description="Confidence Threshold"
    )
    show: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        display_name="show 3D visualization",
        description="Whether to show the 3D visualizaiton on screen"
    )


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    default_scope: str = STR_FIELD(
        value="mmdet3d",
        default_value="mmdet3d",
        display_name="default scope",
        description="Default scope to use mmdet3d"
    )
    default_hooks: Dict[Any, Any] = DICT_FIELD(
        hashMap={'timer': {'type': 'IterTimerHook'},
                 'logger': {'type': 'LoggerHook', 'interval': 1, 'log_metric_by_epoch': True},
                 'param_scheduler': {'type': 'ParamSchedulerHook'},
                 'checkpoint': {'type': 'CheckpointHook', 'by_epoch': True, 'interval': 1},
                 'sampler_seed': {'type': 'DistSamplerSeedHook'},
                 'visualization': {'type': 'Det3DVisualizationHook'}},
        display_name="default hooks",
        description="Default hooks for mmlabs"
    )
    logger_hook: Optional[str] = STR_FIELD(
        value="TAOBEVFusionLoggerHook",
        default_value="TAOBEVFusionLoggerHook",
        display_name="logger hook",
        description="Default logger hook type"
    )
    manual_seed: Optional[int] = INT_FIELD(
        value=None,
        default_value=None,
        display_name="manual seed",
        description="Optional manual seed. Seed is set when the value is given in spec file."
    )
    input_modality: Dict[str, str] = DICT_FIELD(
        hashMap={"use_lidar": True, "use_camera": True, "use_radar": False,
                 "use_map": False, "use_external": False},
        display_name="input modality",
        description="Input modality for the model. Set True for each modality to use."
    )
    model: BEVFusionModelConfig = DATACLASS_FIELD(
        BEVFusionModelConfig(),
        description="Configurable parameters to construct the model for a BEVFusion experiment."
    )
    dataset: BEVFusionDatasetExpConfig = DATACLASS_FIELD(
        BEVFusionDatasetExpConfig(),
        description="Configurable parameters to construct the dataset for a BEVFusion experiment."
    )
    train: BEVFusionTrainExpConfig = DATACLASS_FIELD(
        BEVFusionTrainExpConfig(),
        description="Configurable parameters to construct the trainer for a BEVFusion experiment."
    )
    evaluate: EvaluateConfig = DATACLASS_FIELD(
        EvaluateConfig(),
        description="Configurable parameters to construct the evaluator for a BEVFusion experiment."
    )
    inference: BEVFusionInferenceExpConfig = DATACLASS_FIELD(
        BEVFusionInferenceExpConfig(),
        description="Configurable parameters to construct the inferencer for a BEVFusion experiment."
    )

    def __post_init__(self):
        """Set default model name for BEVFusion."""
        if self.model_name is None:
            self.model_name = "bevfusion"
