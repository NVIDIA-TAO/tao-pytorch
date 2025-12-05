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

"""Validation utilities for the quantization framework."""

from typing import List, Optional, Union

import torch.nn as nn

from nvidia_tao_pytorch.core.quantization.constants import (
    DTYPE_NORMALIZATION_MAP,
    QuantizationMode,
    SupportedDtype,
)


def get_valid_dtype_options() -> List[str]:
    """Return valid dtype options derived from the enum values.

    Returns
    -------
    list[str]
        Valid data type strings (e.g., ["int8", "fp8_e4m3fn", "fp8_e5m2"]).
    """
    return [dtype.value for dtype in SupportedDtype]


def normalize_dtype(dtype: str) -> str:
    """Normalize dtype strings to internal representation.

    Converts common variations (e.g., "float8_*" -> "fp8_*") to match internal expectations.

    Parameters
    ----------
    dtype : str
        Input dtype string.

    Returns
    -------
    str
        Normalized dtype string.

    Raises
    ------
    TypeError
        If dtype is not a string.
    """
    if not isinstance(dtype, str):
        raise TypeError(f"dtype must be a string, got {type(dtype).__name__}")
    dtype_lower = dtype.lower()
    return DTYPE_NORMALIZATION_MAP.get(dtype_lower, dtype_lower)


def validate_dtype(dtype: str) -> SupportedDtype:
    """Validate and convert dtype string to enum.

    Parameters
    ----------
    dtype : str
        Dtype string to validate.

    Returns
    -------
    SupportedDtype
        Validated dtype enum value.

    Raises
    ------
    TypeError
        If dtype is None.
    ValueError
        If dtype is not supported.
    """
    if dtype is None:
        raise TypeError("dtype cannot be None")

    normalized = normalize_dtype(dtype)

    try:
        return SupportedDtype(normalized)
    except ValueError:
        valid = get_valid_dtype_options()
        raise ValueError(
            f"Unsupported dtype '{dtype}' (normalized: '{normalized}'). "
            f"Supported dtypes: {valid}. "
            "To extend support, add the dtype to SupportedDtype in constants.py."
        ) from None


def assert_supported_dtype(dtype: str) -> None:
    """Raise a helpful error if dtype is not supported.

    Deprecated: Use validate_dtype() instead for better type safety.

    Parameters
    ----------
    dtype : str
        Dtype string to validate.

    Raises
    ------
    TypeError
        If dtype is None.
    ValueError
        If dtype is not supported.
    """
    validate_dtype(dtype)


def validate_mode(mode: str) -> QuantizationMode:
    """Validate and convert mode string to enum.

    Parameters
    ----------
    mode : str
        Mode string to validate.

    Returns
    -------
    QuantizationMode
        Validated mode enum value.

    Raises
    ------
    TypeError
        If mode is None.
    ValueError
        If mode is not supported.
    """
    if mode is None:
        raise TypeError("mode cannot be None")
    if isinstance(mode, QuantizationMode):
        return mode
    return QuantizationMode.from_string(str(mode))


def validate_backend(backend: str) -> str:
    """Validate backend string.

    Parameters
    ----------
    backend : str
        Backend string to validate.

    Returns
    -------
    str
        Validated backend string.

    Raises
    ------
    ValueError
        If backend is not a string or is empty.

    Notes
    -----
    This function validates format only. Backend availability is checked
    separately by the registry during backend loading.
    """
    if not isinstance(backend, str):
        raise ValueError(f"Backend must be a string, got {type(backend).__name__}")
    if not backend:
        raise ValueError("Backend cannot be empty")
    return backend


def validate_model(model: nn.Module) -> nn.Module:
    """Validate model is a proper nn.Module instance.

    This function requires a non-None model. For optional models, use
    validate_optional_model() instead.

    Parameters
    ----------
    model : torch.nn.Module
        Model to validate.

    Returns
    -------
    torch.nn.Module
        Validated model (same as input).

    Raises
    ------
    TypeError
        If model is None or not an nn.Module instance.

    See Also
    --------
    validate_optional_model : For validating models that can be None
    """
    if model is None:
        raise TypeError(
            "model cannot be None. "
            "If you need to accept None (e.g., for file-based quantization), "
            "use validate_optional_model() instead."
        )

    if not isinstance(model, nn.Module):
        raise TypeError(
            f"model must be an instance of torch.nn.Module, got {type(model).__name__}"
        )

    return model


def validate_optional_model(model: Optional[nn.Module]) -> Optional[nn.Module]:
    """Validate model is a proper nn.Module instance or None.

    This function accepts None, typically used for file-based quantization
    backends (e.g., ONNX) that don't operate on PyTorch models directly.

    Parameters
    ----------
    model : torch.nn.Module or None
        Model to validate, or None.

    Returns
    -------
    torch.nn.Module or None
        Validated model, or None if input was None.

    Raises
    ------
    TypeError
        If model is not None and not an nn.Module instance.

    See Also
    --------
    validate_model : For validating models that must not be None
    """
    if model is None:
        return None

    if not isinstance(model, nn.Module):
        raise TypeError(
            f"model must be an instance of torch.nn.Module or None, got {type(model).__name__}"
        )

    return model


def validate_backend_mode_compatibility(backend: str, mode: Union[str, QuantizationMode], supported_modes: set) -> None:
    """Validate that a backend supports the specified mode.

    Parameters
    ----------
    backend : str
        Backend name.
    mode : str or QuantizationMode
        Mode string or enum to validate.
    supported_modes : set
        Set of mode strings supported by the backend.

    Raises
    ------
    ValueError
        If mode is not supported by the backend.
    """
    mode_value = mode.value if isinstance(mode, QuantizationMode) else str(mode).lower()

    if mode_value not in supported_modes:
        raise ValueError(
            f"Unsupported mode '{mode}' for backend '{backend}'. "
            f"Supported modes: {sorted(supported_modes)}"
        )
