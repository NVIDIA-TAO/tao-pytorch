# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema to deploy the model."""

from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    DATACLASS_FIELD,
    STR_FIELD,
    INT_FIELD,
)
from nvidia_tao_pytorch.config.common.common_config import (
    GenTrtEngineConfig,
    TrtConfig,
    CalibrationConfig
)


@dataclass
class RTTrtConfig(TrtConfig):
    """Trt config."""

    data_type: str = STR_FIELD(
        value="FP32",
        default_value="FP32",
        description="The precision to be set for building the TensorRT engine.",
        display_name="data type",
        valid_options=",".join(["FP32", "FP16", "INT8"])
    )
    max_batch_size: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=1,
        description="""The maximum batch size in the optimization profile for
                    the input tensor of the TensorRT engine.""",
        display_name="Maximum batch size",
    )
    calibration: CalibrationConfig = DATACLASS_FIELD(
        CalibrationConfig(),
        description="""The configuration elements to define the
                    TensorRT calibrator for int8 PTQ.""",
    )


@dataclass
class RTGenTrtEngineExpConfig(GenTrtEngineConfig):
    """Gen TRT Engine experiment config."""

    tensorrt: RTTrtConfig = DATACLASS_FIELD(
        RTTrtConfig(),
        description="Hyper parameters to configure the TensorRT Engine builder.",
        display_name="TensorRT hyper params."
    )
