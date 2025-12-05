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

"""TorchAO quantization backend package.

This package provides integration with TorchAO weight-only quantization
capabilities for the TAO Toolkit. It includes the backend implementation
for weight-only post-training quantization (PTQ) in INT8 and FP8 formats.

Classes
-------
TorchAOBackend
    Main backend class for TorchAO weight-only quantization integration.

Notes
-----
Importing this package automatically registers the ``torchao`` backend
via the ``@register_backend`` decorator on ``TorchAOBackend``.

The backend requires the TorchAO package to be installed. If not available,
an ImportError will be raised when attempting to use the backend.

Examples
--------
>>> from nvidia_tao_pytorch.core.quantization.backends.torchao import TorchAOBackend
>>> from nvidia_tao_core.config.common.quantization.default_config import ModelQuantizationConfig
>>>
>>> # Create and use the backend
>>> backend = TorchAOBackend()
>>> config = ModelQuantizationConfig(mode="weight_only_ptq")
>>> backend.prepare(model=model, config=config)
>>> quantized_model = backend.quantize(model=model, config=config)
"""

from nvidia_tao_pytorch.core.quantization.backends.torchao.torchao import TorchAOBackend

__all__ = [
    "TorchAOBackend",
]
