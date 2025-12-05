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

"""Utility functions for the ModelOpt ONNX backend.

This module provides helpers to translate TAO Toolkit quantization
configuration dataclasses into parameters that are accepted by the
ModelOpt ONNX quantization API.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from nvidia_tao_pytorch.core.quantization import (
    LayerQuantizationConfig,
    ModelQuantizationConfig,
)
from nvidia_tao_pytorch.core.quantization.constants import QuantizationMode
from nvidia_tao_pytorch.core.quantization.validation import assert_supported_dtype
from nvidia_tao_pytorch.core.tlt_logging import logger as tlt_logger

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False


def format_params_for_logging(params: Dict[str, Any]) -> str:
    """Format ModelOpt parameters for clean logging without large arrays.

    Parameters
    ----------
    params : dict
        ModelOpt parameters dictionary

    Returns
    -------
    str
        Formatted string with array shapes instead of full arrays
    """
    formatted = {}
    for key, value in params.items():
        if hasattr(value, 'shape'):
            # NumPy array or similar - show shape and dtype
            dtype = getattr(value, 'dtype', 'unknown')
            formatted[key] = f"<array: shape={value.shape}, dtype={dtype}>"
        elif isinstance(value, (list, tuple)) and value and hasattr(value[0], 'shape'):
            # List of arrays
            formatted[key] = f"<list of {len(value)} arrays>"
        elif isinstance(value, dict):
            # Dictionary - check if it contains arrays
            formatted[key] = {k: f"<array: shape={v.shape}>" if hasattr(v, 'shape') else v
                              for k, v in value.items()}
        else:
            formatted[key] = value
    return str(formatted)


__all__ = [
    "convert_tao_to_modelopt_onnx_params",
    "format_params_for_logging",
]


def _dtype_to_quantize_mode(dtype: str) -> str:
    """Convert TAO dtype to ModelOpt ONNX quantize_mode.

    This function maps TAO Toolkit data types to ModelOpt ONNX quantization
    modes. It validates the input dtype and returns the corresponding
    ModelOpt ONNX quantize_mode.

    Parameters
    ----------
    dtype : str
        TAO dtype string. Supported values:
        - "int8": 8-bit integer quantization
        - "fp8_e4m3fn": FP8 with E4M3 format
        - "fp8_e5m2": FP8 with E5M2 format

    Returns
    -------
    str
        ModelOpt ONNX quantize_mode:
        - "int8" for int8 quantization
        - "fp8" for FP8 quantization

    Raises
    ------
    ValueError
        If the dtype is not supported by ModelOpt ONNX backend.

    Notes
    -----
    The function first validates the dtype using the TAO validation
    framework, then maps it to the appropriate ModelOpt ONNX mode.
    """
    key = dtype.lower()

    # Validate dtype first
    assert_supported_dtype(key)

    if key == "int8":
        return "int8"
    elif key in {"fp8_e4m3fn", "fp8_e5m2"}:
        return "fp8"
    else:
        raise ValueError(f"Unsupported dtype '{dtype}' for ModelOpt ONNX backend")


def _extract_op_types_to_quantize(config: ModelQuantizationConfig) -> List[str]:
    """Extract ONNX operator types to quantize from TAO configuration.

    This function analyzes the TAO quantization configuration and maps
    PyTorch module names to corresponding ONNX operator types that should
    be quantized. It examines the layer configurations and determines
    which ONNX operators need quantization.

    Parameters
    ----------
    config : ModelQuantizationConfig
        TAO quantization configuration containing layer specifications.

    Returns
    -------
    List[str]
        List of ONNX operator types to quantize. Returns empty list if no layers
        are configured for quantization. Common operator types include:
        - "Gemm": General matrix multiplication
        - "Conv": Convolution operations
        - "MatMul": Matrix multiplication
        - "Add": Element-wise addition
        - "Mul": Element-wise multiplication
        - "AveragePool", "MaxPool": Pooling operations
        - "BatchNormalization": Batch normalization
        - "Clip": Clipping operations
        - "GlobalAveragePool": Global average pooling
        - "ConvTranspose": Transposed convolution

    Notes
    -----
    The function maps PyTorch module names to ONNX operator types based
    on common naming patterns. The mapping is:
    - "linear" -> "Gemm"
    - "conv" -> "Conv"
    - "matmul" -> "MatMul"
    - "add" -> "Add"
    - "mul" -> "Mul"
    - "pool" -> "AveragePool", "MaxPool"
    - "batchnorm"/"bn" -> "BatchNormalization"
    - "clip" -> "Clip"
    - "global" -> "GlobalAveragePool"
    - "transpose" -> "ConvTranspose"
    """
    op_types = set()

    for layer in config.layers or []:
        if not isinstance(layer, LayerQuantizationConfig):
            continue

        # Map TAO module names to ONNX op types
        module_name = layer.module_name.lower()

        if "linear" in module_name:
            op_types.add("Gemm")
        elif "conv" in module_name:
            op_types.add("Conv")
        elif "matmul" in module_name:
            op_types.add("MatMul")
        elif "add" in module_name:
            op_types.add("Add")
        elif "mul" in module_name:
            op_types.add("Mul")
        elif "pool" in module_name:
            op_types.add("AveragePool")
            op_types.add("MaxPool")
        elif "batchnorm" in module_name or "bn" in module_name:
            op_types.add("BatchNormalization")
        elif "clip" in module_name:
            op_types.add("Clip")
        elif "global" in module_name:
            op_types.add("GlobalAveragePool")
        elif "transpose" in module_name:
            op_types.add("ConvTranspose")

    return list(op_types)


def _extract_nodes_to_exclude(config: ModelQuantizationConfig) -> List[str]:
    """Extract ONNX node names to exclude from quantization.

    This function extracts node names to exclude from the configuration's
    skip_names list. For ONNX backends, skip_names should contain actual
    ONNX node names (not PyTorch module names).

    Parameters
    ----------
    config : ModelQuantizationConfig
        TAO quantization configuration containing skip_names list with ONNX node names.

    Returns
    -------
    List[str]
        List of ONNX node names to exclude from quantization.

    Notes
    -----
    Since this is an ONNX backend, users should provide ONNX node names directly
    in the skip_names field. To find ONNX node names, users can:
    1. Use Netron to visualize the ONNX model
    2. Load the ONNX model with onnx.load() and inspect node names
    3. Use onnxruntime tools to analyze the model structure
    """
    if not config.skip_names:
        return []

    # For ONNX backend, skip_names should already be ONNX node names
    return list(config.skip_names)


def _determine_calibration_method(config: ModelQuantizationConfig) -> str:
    """Determine calibration method from TAO configuration.

    This function analyzes the TAO quantization configuration to determine
    the appropriate calibration method for ModelOpt ONNX quantization.
    It checks for algorithm specifications and falls back to mode-based defaults.

    Parameters
    ----------
    config : ModelQuantizationConfig
        TAO quantization configuration containing algorithm and mode settings.

    Returns
    -------
    str
        Calibration method for ModelOpt ONNX. Supported methods:
        - "entropy": Entropy-based calibration
        - "max": Maximum value calibration
        - "awq_clip": AWQ with clipping
        - "awq_lite": Lightweight AWQ
        - "awq_full": Full AWQ
        - "rtn_dq": RTN with dynamic quantization

    Notes
    -----
    The function first checks for an explicit algorithm specification in
    the configuration. If not found, it defaults to "max" for static PTQ
    mode. The algorithm string is converted to lowercase for matching.

    For static post-training quantization (PTQ), the default method is "max"
    which uses the maximum absolute value for calibration.
    """
    # Check if algorithm is specified in config
    algorithm = getattr(config, "algorithm", None)
    if algorithm:
        algorithm_str = str(algorithm).lower()
        if algorithm_str in ["entropy", "max", "awq_clip", "awq_lite", "awq_full", "rtn_dq"]:
            return algorithm_str

    # Default calibration methods based on mode
    mode = config.mode
    if isinstance(mode, QuantizationMode):
        mode_str = mode.name.lower()
    else:
        mode_str = str(mode).lower()

    if mode_str == "static_ptq":
        return "max"  # Default for PTQ
    else:
        return "max"  # Fallback


def _determine_execution_providers(device_str: str, logger) -> List[str]:
    """Determine ONNX Runtime execution providers based on device configuration.

    This function maps TAO device configuration to ONNX Runtime execution providers.
    It handles various device types including CPU, CUDA, and TensorRT. All available
    providers are included in the list, with the preferred provider first.

    Parameters
    ----------
    device_str : str
        Device string from configuration. Valid values:
        - "cpu": Use CPU execution provider
        - "cuda", "cuda:0", etc.: Use CUDA execution provider as preferred
        - "trt": Use TensorRT execution provider as preferred
    logger : Logger
        Logger instance for warnings and information

    Returns
    -------
    List[str]
        List of execution providers in priority order with all available providers.
        The preferred provider is listed first, followed by other available providers:
        - CPU device: ["CPUExecutionProvider"]
        - CUDA device: ["CUDAExecutionProvider", "CPUExecutionProvider"] (if CUDA available)
          or ["CPUExecutionProvider"] (if CUDA not available)
        - TRT device: ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
          (with all available providers included)

    Notes
    -----
    The function queries ONNX Runtime for available providers and includes all of them
    in the returned list, with the preferred provider based on device_str listed first.
    This allows ONNX Runtime to automatically fall back to alternative providers if needed.
    """
    providers = []
    preferred_provider = None

    # Determine preferred provider based on device string
    if device_str == 'trt':
        preferred_provider = 'TensorrtExecutionProvider'
    elif device_str.startswith('cuda'):
        preferred_provider = 'CUDAExecutionProvider'
    else:  # cpu or any other value
        preferred_provider = 'CPUExecutionProvider'

    # Get available providers from ONNX Runtime
    if not ONNXRUNTIME_AVAILABLE:
        logger.warning(
            "ONNX Runtime not available for provider detection. "
            "Using CPUExecutionProvider as fallback."
        )
        return ['CPUExecutionProvider']

    available_providers = ort.get_available_providers()

    # Additional validation for CUDA
    if preferred_provider == 'CUDAExecutionProvider':
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available on this system. "
                "Will use CPU for calibration."
            )
            preferred_provider = 'CPUExecutionProvider'

    # Build provider list: preferred first, then all other available providers
    if preferred_provider in available_providers:
        providers.append(preferred_provider)
        logger.info(f"Using {preferred_provider} as preferred execution provider for ONNX quantization")

        # Add all other available providers
        for provider in available_providers:
            if provider != preferred_provider and provider not in providers:
                providers.append(provider)
    else:
        # Preferred provider not available, warn and use all available providers
        logger.warning(
            f"{preferred_provider} requested but not available in ONNX Runtime. "
            f"Available providers: {available_providers}. "
            f"Using available providers with automatic fallback."
        )
        providers = list(available_providers)

    # Ensure at least CPUExecutionProvider is present
    if 'CPUExecutionProvider' not in providers:
        providers.append('CPUExecutionProvider')

    logger.debug(f"ONNX Runtime execution providers (in priority order): {providers}")

    return providers


def convert_tao_to_modelopt_onnx_params(
    config: ModelQuantizationConfig,
    model: Optional[nn.Module],
    onnx_path: str,
    output_path: Optional[str] = None,
    calibration_data: Optional[Any] = None,
    backend_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert TAO configuration to ModelOpt ONNX parameters.

    This function translates TAO Toolkit quantization configuration into
    parameters that are compatible with the ModelOpt ONNX quantization API.
    It extracts quantization settings, operator types, calibration methods,
    and other parameters needed for ONNX model quantization.

    Parameters
    ----------
    config : ModelQuantizationConfig
        TAO quantization configuration containing layer specifications,
        quantization modes, and other settings.
    model : torch.nn.Module, optional
        PyTorch model (ignored for ONNX backend, kept for API compatibility).
    onnx_path : str
        Path to the input ONNX model file to be quantized.
    output_path : str, optional
        Path where the quantized ONNX model will be saved. If not provided,
        defaults to config.results_dir + "/quantized_model.onnx".
    calibration_data : Any, optional
        Calibration data as numpy array or other format compatible with
        ModelOpt ONNX. Used for static quantization calibration.
    backend_kwargs : dict, optional
        Additional keyword arguments to merge with the converted parameters.
        These can override default settings or provide additional ModelOpt
        ONNX options like per_channel, reduce_range, use_external_data_format, etc.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing ModelOpt ONNX parameters:
        - "onnx_path": Path to input ONNX model
        - "quantize_mode": Quantization mode ("int8" or "fp8")
        - "calibration_data": Calibration data for quantization
        - "calibration_method": Method for calibration ("max", "entropy", etc.)
        - "op_types_to_quantize": List of ONNX operator types to quantize
        - "nodes_to_exclude": List of node names to exclude from quantization
        - "output_path": Path for saving quantized model
        - Additional parameters from backend_kwargs

    Raises
    ------
    TypeError
        If config is None.

    Notes
    -----
    The function determines the quantization mode from the first layer's
    dtype specification. It extracts operator types to quantize by mapping
    PyTorch module names to ONNX operator types. Node exclusions are taken
    from config.skip_names (should be actual ONNX node names).

    The calibration method is determined from the configuration's algorithm
    setting or defaults to "max" for static PTQ. Additional backend-specific
    parameters from backend_kwargs are merged and can override defaults.

    Users can pass additional ModelOpt ONNX parameters via backend_kwargs:
    - per_channel: bool - Enable per-channel quantization
    - reduce_range: bool - Reduce quantization range for better accuracy
    - use_external_data_format: bool - For large models (>2GB)
    - And any other ModelOpt ONNX-specific parameters

    We default the dq_only parameter on the ModelOpt ONNX call to False,
    but it can be overridden via backend_kwargs. This was done in order to ensure parity
    with the modelopt onnx CLI behavior and to allow for compatibility with TRT downstream,
    which requires a Q-DQ node pair.
    """
    if config is None:
        raise TypeError("config cannot be None")

    # Validate dtype consistency across layers and between weights/activations
    # ModelOpt ONNX backend uses a single global quantize_mode for the entire model
    weight_dtypes = set()
    activation_dtypes = set()
    layer_mismatches = []

    if config.layers:
        for layer in config.layers:
            if isinstance(layer, LayerQuantizationConfig):
                w_dtype = None
                a_dtype = None

                if layer.weights and layer.weights.dtype:
                    w_dtype = str(layer.weights.dtype).lower()
                    weight_dtypes.add(w_dtype)

                if layer.activations and layer.activations.dtype:
                    a_dtype = str(layer.activations.dtype).lower()
                    activation_dtypes.add(a_dtype)

                # Check if weights and activations differ within the same layer
                if w_dtype and a_dtype and w_dtype != a_dtype:
                    layer_mismatches.append({
                        'module': layer.module_name,
                        'weights': w_dtype,
                        'activations': a_dtype
                    })

    # Warning 1: Different dtypes between weights and activations in the same layer
    if layer_mismatches:
        tlt_logger.warning(
            "ModelOpt ONNX backend detected different dtypes for weights and activations within layers. "
            "The ONNX backend uses a single 'quantize_mode' for the entire model and cannot apply "
            "different dtypes to weights vs activations per layer. The first layer's dtype will be used. "
            "Mismatched layers:"
        )
        for mismatch in layer_mismatches:
            tlt_logger.warning(
                f"  - Layer '{mismatch['module']}': weights={mismatch['weights']}, "
                f"activations={mismatch['activations']}"
            )
        tlt_logger.warning(
            "Using different dtypes for weights and activations is currently unsupported in the ONNX backend. "
            "Please use the same dtype for both weights and activations."
        )

    # Warning 2: Different dtypes across different layers
    all_dtypes = weight_dtypes | activation_dtypes
    if len(all_dtypes) > 1:
        tlt_logger.warning(
            f"ModelOpt ONNX backend detected multiple different dtypes across layers: {sorted(all_dtypes)}. "
            "The ONNX backend uses a single global 'quantize_mode' for the entire model. "
            "Only the first layer's dtype will be applied to all layers. "
            "If you need per-layer quantization control, consider using the 'modelopt' (PyTorch) backend instead, "
            "or specify operators to quantize using 'op_types_to_quantize' in backend_kwargs."
        )

    # Determine quantization mode from the first layer's dtype
    quantize_mode = "int8"  # Default
    if config.layers:
        for layer in config.layers:
            if isinstance(layer, LayerQuantizationConfig):
                if layer.weights and layer.weights.dtype:
                    quantize_mode = _dtype_to_quantize_mode(layer.weights.dtype)
                    break
                if layer.activations and layer.activations.dtype:
                    quantize_mode = _dtype_to_quantize_mode(layer.activations.dtype)
                    break

    # Extract op types to quantize
    op_types_to_quantize = _extract_op_types_to_quantize(config)

    # Extract nodes to exclude from config.skip_names (ONNX node names)
    nodes_to_exclude = _extract_nodes_to_exclude(config)

    # Determine calibration method
    calibration_method = _determine_calibration_method(config)

    # Get device from config and determine execution providers
    device_str = getattr(config, 'device', 'cuda')
    execution_providers = _determine_execution_providers(device_str, tlt_logger)

    # Build parameters dict with core parameters
    params = {
        "onnx_path": onnx_path,
        "quantize_mode": quantize_mode,
        "calibration_data": calibration_data,
        "calibration_method": calibration_method,
        "op_types_to_quantize": op_types_to_quantize or None,  # None means quantize all supported ops
        "nodes_to_exclude": nodes_to_exclude or None,
        "output_path": output_path if output_path is not None else os.path.join(config.results_dir, "quantized_model.onnx"),
        "dq_only": False,  # Default to False, can be overridden via backend_kwargs
    }

    # Add execution providers if not already specified in backend_kwargs
    # This allows backend_kwargs to override the device-based provider selection if needed
    if backend_kwargs and 'execution_providers' not in backend_kwargs:
        params['execution_providers'] = execution_providers
    elif not backend_kwargs:
        params['execution_providers'] = execution_providers

    # Merge backend_kwargs (allows advanced ModelOpt ONNX parameters)
    # backend_kwargs can include: per_channel, reduce_range, activation_type,
    # weight_type, use_external_data_format, extra_options, execution_providers, etc.
    # Note: If execution_providers is in backend_kwargs, it will override our device-based selection
    if backend_kwargs:
        params.update(backend_kwargs)

    return params
