# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
>>> from nvidia_tao_core.config.common.quantization.default_config import ModelQuantizationConfig
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
