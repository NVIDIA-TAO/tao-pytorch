# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelOpt ONNX backend integration for TAO quantization framework.

This package provides integration with NVIDIA ModelOpt ONNX quantization
capabilities for the TAO Toolkit. It includes the backend implementation
and utility functions for converting TAO configurations to ModelOpt ONNX
parameters.

The ModelOpt ONNX backend supports static post-training quantization (PTQ)
and works exclusively with ONNX model files. It translates TAO quantization
configurations to ModelOpt ONNX parameters and invokes the ModelOpt ONNX
quantization APIs.

Classes
-------
ModelOptONNXBackend
    Main backend class for ModelOpt ONNX quantization integration.

Functions
---------
convert_tao_to_modelopt_onnx_params
    Utility function to convert TAO configuration to ModelOpt ONNX parameters.

Notes
-----
Importing this package automatically registers the ``modelopt.onnx`` backend
via the ``@register_backend`` decorator on ``ModelOptONNXBackend``.

The backend requires the ModelOpt ONNX package to be installed. If not available,
an ImportError will be raised when attempting to use the backend.

Examples
--------
>>> from nvidia_tao_pytorch.core.quantization.backends.modelopt_onnx import ModelOptONNXBackend
>>> from nvidia_tao_pytorch.config.common.quantization.default_config import ModelQuantizationConfig
>>>
>>> # Create and use the backend
>>> backend = ModelOptONNXBackend()
>>> config = ModelQuantizationConfig(backend="modelopt.onnx", model_path="/path/to/model.onnx")
>>> backend.prepare(model=None, config=config)
>>> backend.quantize(model=None, config=config)
"""

from nvidia_tao_pytorch.core.quantization.backends.modelopt_onnx.modelopt_onnx import (
    ModelOptONNXBackend,
)
from nvidia_tao_pytorch.core.quantization.backends.modelopt_onnx.utils import (
    convert_tao_to_modelopt_onnx_params,
)

__all__ = [
    "ModelOptONNXBackend",
    "convert_tao_to_modelopt_onnx_params",
]
