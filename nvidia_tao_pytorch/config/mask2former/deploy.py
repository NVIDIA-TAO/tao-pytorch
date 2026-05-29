# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema to deploy the model."""

from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    DATACLASS_FIELD,
    STR_FIELD
)
from nvidia_tao_pytorch.config.common.common_config import (
    GenTrtEngineConfig,
    TrtConfig
)


@dataclass
class Mask2FormerTrtConfig(TrtConfig):
    """Trt config."""

    data_type: str = STR_FIELD(
        value="FP32",
        default_value="FP32",
        description="The precision to be set for building the TensorRT engine.",
        display_name="data type",
        valid_options=",".join(["FP32", "FP16"])
    )


@dataclass
class Mask2FormerGenTrtEngineExpConfig(GenTrtEngineConfig):
    """Gen TRT Engine experiment config."""

    tensorrt: Mask2FormerTrtConfig = DATACLASS_FIELD(
        Mask2FormerTrtConfig(),
        description="Hyper parameters to configure the TensorRT Engine builder.",
        display_name="TensorRT hyper params."
    )
