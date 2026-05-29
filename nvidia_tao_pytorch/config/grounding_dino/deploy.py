# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration hyperparameter schema to deploy the model."""

from dataclasses import dataclass

from nvidia_tao_pytorch.config.utils.types import (
    DATACLASS_FIELD,
    STR_FIELD,
    INT_FIELD
)
from nvidia_tao_pytorch.config.common.common_config import (
    GenTrtEngineConfig,
    TrtConfig
)


@dataclass
class GDINOTrtConfig(TrtConfig):
    """Trt config."""

    workspace_size: int = INT_FIELD(
        value=8192,
        default_value=8192,
        valid_min=0,
        description="""The size (in MB) of the workspace TensorRT has
                    to run it's optimization tactics and generate the
                    TensorRT engine.""",
        display_name="Max workspace size",
    )
    max_batch_size: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=1,
        description="""The maximum batch size in the optimization profile for
                    the input tensor of the TensorRT engine.""",
        display_name="Maximum batch size",
    )
    data_type: str = STR_FIELD(
        value="FP32",
        default_value="FP32",
        description="The precision to be set for building the TensorRT engine.",
        display_name="data type",
        valid_options=",".join(["FP32", "FP16"])
    )


@dataclass
class GDINOGenTrtEngineExpConfig(GenTrtEngineConfig):
    """Gen TRT Engine experiment config."""

    tensorrt: GDINOTrtConfig = DATACLASS_FIELD(
        GDINOTrtConfig(),
        description="Hyper parameters to configure the TensorRT Engine builder.",
        display_name="TensorRT hyper params."
    )
