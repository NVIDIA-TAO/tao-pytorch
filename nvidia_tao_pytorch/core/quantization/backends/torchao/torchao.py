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

"""TorchAO quantization backend for TAO Toolkit.

Implements a backend that adapts TAO's ``ModelQuantizationConfig`` to
``torchao.quantization`` APIs for weight-only PTQ in INT8 and FP8 formats.

Notes
-----
- Only the weight-only PTQ mode is supported by this backend.
- Activation quantization settings in the TAO configuration are ignored.
- Layerwise configuration is supported via ``module_name`` patterns which are
  matched against qualified module names and module classes.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from nvidia_tao_core.config.common.quantization.default_config import (
    LayerQuantizationConfig,
    ModelQuantizationConfig,
)
from nvidia_tao_pytorch.core.quantization.constants import QuantizationMode
from nvidia_tao_pytorch.core.quantization.quantizer_base import PyTorchQuantizerBase
from nvidia_tao_pytorch.core.quantization.registry import register_backend
from nvidia_tao_pytorch.core.quantization.utils import match_layer
from nvidia_tao_pytorch.core.quantization.validation import (
    assert_supported_dtype,
    validate_backend_mode_compatibility,
    validate_model,
)
from nvidia_tao_pytorch.core.tlt_logging import logger as tlt_logger


try:  # pragma: no cover - import error path is tested via unit test patching
    # TorchAO weight-only APIs
    # mypy/pyright may not know these symbols; keep import local and un-typed.
    from torchao.quantization import (  # type: ignore
        Float8WeightOnlyConfig,
        Int8WeightOnlyConfig,
        AOPerModuleConfig,
        quantize_,
    )
except Exception as exc:  # pragma: no cover - will be simulated in tests
    raise ImportError(
        "torchao is not installed or failed to import. Install torchao to use the 'torchao' backend."
    ) from exc


# Only support weight-only PTQ in this backend
SUPPORTED_MODES = {QuantizationMode.WEIGHT_ONLY_PTQ.value}


def _select_weightonly_cfg(dtype: str):
    """Return the TorchAO weight-only config class instance for a TAO dtype.

    Parameters
    ----------
    dtype : str
        TAO dtype string, one of {"int8", "fp8_e4m3fn", "fp8_e5m2"}.

    Returns
    -------
    object
        Instance of a TorchAO weight-only config matching the dtype.

    Raises
    ------
    ValueError
        If the dtype is not supported by this backend.
    """
    key = str(dtype).lower()
    # Validate dtype against TAO SupportedDtype for helpful messages first
    assert_supported_dtype(key)

    # Warn when e5m2 is used
    if key == "fp8_e5m2":
        tlt_logger.warning(
            "Float8 e5m2 dtype is being used. Note that weight-only PTQ only supports e4m3 and will fall back to it."
        )

    if key == "int8":
        return Int8WeightOnlyConfig()
    # Both FP8 variants map to Float8WeightOnlyConfig in TorchAO
    if key in {"fp8_e4m3fn", "fp8_e5m2"}:
        return Float8WeightOnlyConfig()
    raise ValueError(
        f"Unsupported dtype '{dtype}' for TorchAO weight-only backend."
    )


def _build_module_fqn_to_cfg(
    model: nn.Module, config: ModelQuantizationConfig
) -> Dict[str, object]:
    """Construct the ``module_fqn_to_config`` mapping for ``AOPerModuleConfig``.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose modules will be matched against layer patterns.
    config : ModelQuantizationConfig
        TAO configuration containing layerwise specifications and skip patterns.

    Returns
    -------
    dict[str, object]
        Mapping from module qualified names to TorchAO weight-only config objects.

    Notes
    -----
    - Only ``LayerQuantizationConfig.weights`` is considered. Activation settings
      are ignored by this backend.
    - Later layer specifications override earlier ones for the same module.
    - ``skip_names`` remove any matched modules from the mapping.
    """
    mapping: Dict[str, object] = {}

    named_modules = [(n, m) for n, m in model.named_modules() if n]

    # Apply layerwise rules in order
    for layer in config.layers or []:
        if not isinstance(layer, LayerQuantizationConfig):
            raise TypeError(
                "Items of 'layers' must be LayerQuantizationConfig instances."
            )
        if not layer.module_name:
            raise ValueError("module_name in LayerQuantizationConfig cannot be empty")
        if layer.weights is None:
            # Weight-only backend: ignore layers without weight settings
            continue

        # Support explicit per-layer opt-out via dtype=="native"
        weight_dtype = str(getattr(layer.weights, "dtype", "")).lower()
        if weight_dtype == "native":
            # Remove any existing mapping for matched modules (disable quantization)
            for qual_name, module in named_modules:
                if match_layer(module, qual_name, layer.module_name):
                    mapping.pop(qual_name, None)
            continue

        torchao_cfg = _select_weightonly_cfg(weight_dtype)

        for qual_name, module in named_modules:
            if match_layer(module, qual_name, layer.module_name):
                mapping[qual_name] = torchao_cfg

    # Apply skip patterns (remove from mapping)
    for pattern in config.skip_names or []:
        to_delete = [
            qual_name
            for qual_name, module in named_modules
            if match_layer(module, qual_name, pattern)
        ]
        for qual_name in to_delete:
            mapping.pop(qual_name, None)

    return mapping


@register_backend("torchao")
class TorchAOBackend(PyTorchQuantizerBase):
    """TorchAO weight-only PTQ backend.

    This backend constructs a per-module weight-only quantization configuration
    for TorchAO and invokes ``quantize_`` to perform in-place quantization on a
    deep-copied model instance.

    Parameters
    ----------
    backend_kwargs : dict, optional
        Additional keyword arguments to pass to the TorchAO backend.
        These parameters will be merged with the TorchAO configuration
        before calling TorchAO quantization.

    Attributes
    ----------
    backend_name : str
        Name identifier for this backend ("torchao").
    _logger : Logger
        TAO logging instance for this backend.
    _backend_kwargs : dict
        Additional backend-specific parameters.
    """

    def __init__(self, backend_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the TorchAO backend.

        Parameters
        ----------
        backend_kwargs : dict, optional
            Additional keyword arguments to pass to the TorchAO backend.
            These parameters will be merged with the TorchAO configuration
            before calling TorchAO quantization.

        Returns
        -------
        None
        """
        self._logger = tlt_logger
        self.backend_name = "torchao"
        self._backend_kwargs = backend_kwargs or {}

    def prepare(self, model: nn.Module, config: ModelQuantizationConfig) -> nn.Module:
        """Validate inputs and return the model unchanged.

        Parameters
        ----------
        model : torch.nn.Module
            Model to prepare.
        config : ModelQuantizationConfig
            Quantization configuration.

        Returns
        -------
        torch.nn.Module
            The input model unchanged.
        """
        validate_model(model)
        if not isinstance(config, ModelQuantizationConfig):
            raise TypeError("config must be an instance of ModelQuantizationConfig")

        # Validate mode using centralized validation
        mode_value = (
            config.mode.value if isinstance(config.mode, QuantizationMode) else str(config.mode).lower()
        )
        validate_backend_mode_compatibility(self.backend_name, mode_value, SUPPORTED_MODES)

        return model

    def quantize(self, model: nn.Module, config: ModelQuantizationConfig) -> nn.Module:
        """Quantize a model using TorchAO weight-only APIs.

        Translates the TAO configuration into an ``AOPerModuleConfig`` mapping
        and invokes ``torchao.quantization.quantize_``. Additional backend-specific
        parameters from backend_kwargs are merged with the TorchAO configuration.

        Parameters
        ----------
        model : torch.nn.Module
            Prepared model to quantize.
        config : ModelQuantizationConfig
            Quantization configuration.

        Returns
        -------
        torch.nn.Module
            Quantized model (a deep copy of the input model).

        Notes
        -----
        The backend_kwargs are passed directly to the TorchAO quantize_ function.
        Any parameters in backend_kwargs will be forwarded to the underlying TorchAO call.
        """
        validate_model(model)
        if not isinstance(config, ModelQuantizationConfig):
            raise TypeError("config must be an instance of ModelQuantizationConfig")

        module_map = _build_module_fqn_to_cfg(model, config)

        # Create AOPerModuleConfig with module mapping
        ao_cfg = AOPerModuleConfig(module_fqn_to_config=module_map)

        quantized_model = copy.deepcopy(model)

        quantize_(quantized_model, ao_cfg, **self._backend_kwargs)

        return quantized_model

    def save_model(self, model: nn.Module, path: str) -> None:
        """Save the quantized model to a directory.

        Parameters
        ----------
        model : torch.nn.Module
            Quantized model to save.
        path : str
            Directory where the model is saved as ``quantized_model_torchao.pth``.
        """
        validate_model(model)
        if not isinstance(path, str) or not path:
            raise TypeError("path must be a non-empty string")

        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f"quantized_model_{self.backend_name}.pth")

        # Save state_dict for portability
        torch.save(model.state_dict(), save_path)
        self._logger.info("Quantized model saved to: %s", save_path)


__all__ = ["TorchAOBackend"]
