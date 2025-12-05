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

"""Quantization backends for TAO Toolkit."""

from nvidia_tao_pytorch.core.quantization.registry import (
    register_backend,
    get_backend_class,
)
from nvidia_tao_pytorch.core.tlt_logging import logger

# Import concrete backends so they self-register via decorators.
# These imports are intentionally placed here to provide a seamless user
# experience – importing the quantization package will make backends
# available in the registry.

# Track which backends are available
_backend_import_status = {}

# Import ModelOpt PyTorch backend
try:  # pragma: no cover - import side-effect only
    from nvidia_tao_pytorch.core.quantization.backends.modelopt_pytorch import ModelOptBackend  # noqa: F401
    _backend_import_status['modelopt.pytorch'] = True
    logger.debug("Successfully loaded backend: modelopt.pytorch")
except ImportError as e:
    _backend_import_status['modelopt.pytorch'] = False
    logger.debug(f"Backend 'modelopt.pytorch' not available: {e}")
except Exception as e:
    _backend_import_status['modelopt.pytorch'] = False
    logger.warning(f"Unexpected error loading backend 'modelopt.pytorch': {e}")

# Import TorchAO backend so it self-registers via decorator
try:  # pragma: no cover - import side-effect only
    from .torchao.torchao import TorchAOBackend  # noqa: F401
    _backend_import_status['torchao'] = True
    logger.debug("Successfully loaded backend: torchao")
except ImportError as e:
    _backend_import_status['torchao'] = False
    logger.debug(f"Backend 'torchao' not available: {e}")
except Exception as e:
    _backend_import_status['torchao'] = False
    logger.warning(f"Unexpected error loading backend 'torchao': {e}")

# Import ModelOpt ONNX backend so it self-registers via decorator
try:  # pragma: no cover - import side-effect only
    from .modelopt_onnx.modelopt_onnx import ModelOptONNXBackend  # noqa: F401
    _backend_import_status['modelopt.onnx'] = True
    logger.debug("Successfully loaded backend: modelopt.onnx")
except ImportError as e:
    _backend_import_status['modelopt.onnx'] = False
    logger.debug(f"Backend 'modelopt.onnx' not available: {e}")
except Exception as e:
    _backend_import_status['modelopt.onnx'] = False
    logger.warning(f"Unexpected error loading backend 'modelopt.onnx': {e}")

# Log summary of available backends
_available = [name for name, status in _backend_import_status.items() if status]
if _available:
    logger.debug(f"Available quantization backends: {_available}")
else:
    logger.warning(
        "No quantization backends are available. "
        "Install backend packages: 'pip install nvidia-modelopt' or 'pip install torchao'"
    )

__all__ = [
    "register_backend",
    "get_backend_class",
]
