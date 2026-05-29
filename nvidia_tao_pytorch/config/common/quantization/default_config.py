# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for quantization framework.

This module defines dataclasses for specifying quantization configuration at various levels of
granularity, including model, layer, weight, and activation. These configuration classes are used to
control quantization behavior in the TAO Toolkit.

The configuration hierarchy is as follows:

- ``ModelQuantizationConfig``: Top-level configuration that orchestrates the entire quantization process
- ``LayerQuantizationConfig``: Configuration for individual layers or groups of layers
- ``WeightQuantizationConfig``: Configuration specifically for weight quantization
- ``ActivationQuantizationConfig``: Configuration specifically for activation quantization
- ``BaseQuantizationConfig``: Base class providing common quantization parameters

Supported backends include "torchao", "modelopt.pytorch", and "modelopt.onnx", with quantization modes
including "static_ptq" and "weight_only_ptq". The framework supports various data types including
"int8", "fp8_e4m3fn", "fp8_e5m2", and "native" precision.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from nvidia_tao_pytorch.config.utils.types import (
    DICT_FIELD,
    DATACLASS_FIELD,
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD,
)


@dataclass
class BaseQuantizationConfig:
    """Base configuration for quantization.

    Parameters
    ----------
    dtype : str
        Data type to use for quantization. Valid values include "int8", "fp8_e4m3fn",
        "fp8_e5m2", and "native". Defaults to an empty string.
    observer_or_fake_quant : str
        Name of the observer (PTQ) or fake quant (QAT) to use for collecting statistics (must be
        registered). Defaults to an empty string.
    quant_axis : int or None, optional
        Axis along which to apply per-channel quantization. If ``None``, per-tensor quantization is
        used. Defaults to None.
    observer_or_fake_quant_kwargs : dict or None, optional
        Additional keyword arguments to pass to the observer or fake quant constructor. Defaults to
        an empty dictionary.

    Notes
    -----
    This is the base class for all quantization configurations. It provides the common parameters
    needed for both weight and activation quantization.
    """

    dtype: str = STR_FIELD(  # type: ignore
        "",
        default_value="",
        description="Data type to use for quantization (e.g., 'int8', 'fp8_e4m3fn', 'fp8_e5m2')",
        display_name="Quantization data type",
        required="yes",
    )
    observer_or_fake_quant: str = STR_FIELD(  # type: ignore
        "",
        default_value="",
        description="Name of the observer (PTQ) or fake quant (QAT) to use for collecting statistics",
        display_name="Observer or fake quant",
    )
    quant_axis: Optional[int] = INT_FIELD(  # type: ignore
        None,
        description="Axis along which to apply per-channel quantization. If None, per-tensor quantization is used",
        display_name="Quantization axis",
    )
    observer_or_fake_quant_kwargs: Optional[Dict[str, Any]] = DICT_FIELD(
        {},
        description="Additional keyword arguments to pass to the observer or fake quant constructor",
        display_name="Observer kwargs",
    )


@dataclass
class WeightQuantizationConfig(BaseQuantizationConfig):
    """Configuration for weight quantization.

    Inherits all parameters from ``BaseQuantizationConfig``. This configuration is specifically for
    quantizing model weights.

    Parameters
    ----------
    dtype : str
        Data type to use for weight quantization. Valid values include "int8", "fp8_e4m3fn",
        "fp8_e5m2", and "native". Defaults to an empty string.
    observer_or_fake_quant : str
        Name of the observer (PTQ) or fake quant (QAT) to use for collecting weight statistics.
        Must be registered. Defaults to an empty string.
    quant_axis : int or None, optional
        Axis along which to apply per-channel quantization for weights. If ``None``, per-tensor
        quantization is used. Defaults to None.
    observer_or_fake_quant_kwargs : dict or None, optional
        Additional keyword arguments to pass to the weight observer or fake quant constructor.
        Defaults to an empty dictionary.

    Notes
    -----
    Weight quantization typically uses per-channel quantization (quant_axis=0 for most layers)
    to preserve accuracy while reducing model size and improving inference speed.

    See Also
    --------
    BaseQuantizationConfig
        Base class for quantization configurations.
    ActivationQuantizationConfig
        Configuration for activation quantization.
    """
    # No additional parameters for now; all inherited from BaseQuantizationConfig.


@dataclass
class ActivationQuantizationConfig(BaseQuantizationConfig):
    """Configuration for activation quantization.

    Inherits all parameters from ``BaseQuantizationConfig``. This configuration is specifically for
    quantizing model activations.

    Parameters
    ----------
    dtype : str
        Data type to use for activation quantization. Valid values include "int8", "fp8_e4m3fn",
        "fp8_e5m2", and "native". Defaults to an empty string.
    observer_or_fake_quant : str
        Name of the observer (PTQ) or fake quant (QAT) to use for collecting activation statistics.
        Must be registered. Defaults to an empty string.
    quant_axis : int or None, optional
        Axis along which to apply per-channel quantization for activations. If ``None``, per-tensor
        quantization is used. Defaults to None.
    observer_or_fake_quant_kwargs : dict or None, optional
        Additional keyword arguments to pass to the activation observer or fake quant constructor.
        Defaults to an empty dictionary.

    Notes
    -----
    Activation quantization typically uses per-tensor quantization (quant_axis=None) as activations
    are more sensitive to quantization errors and per-channel quantization may not provide
    significant benefits.

    See Also
    --------
    BaseQuantizationConfig
        Base class for quantization configurations.
    WeightQuantizationConfig
        Configuration for weight quantization.
    """
    # No additional parameters for now; all inherited from BaseQuantizationConfig.


@dataclass
class LayerQuantizationConfig:
    """Configuration for quantizing a single module or set of modules.

    Parameters
    ----------
    module_name : str
        Name or pattern specifying the module(s) to quantize. Can be:

        - Exact module name (e.g., "layers.0.conv1")
        - Layer type (e.g., "Linear", "Conv2d")
        - Wildcard pattern (e.g., "conv*", "*.linear")

        The pattern matching is case-sensitive and uses Python's ``fnmatch.fnmatch``.
    weights : WeightQuantizationConfig or None, optional
        Weight quantization configuration. If ``None``, weights are not quantized and left in the
        original precision. Defaults to None.
    activations : ActivationQuantizationConfig or None, optional
        Activation quantization configuration. If ``None``, activations are not quantized and left in
        the original precision. Defaults to None.

    Notes
    -----
    This configuration allows fine-grained control over quantization at the layer level. You can
    specify different quantization strategies for weights and activations of the same layer.

    When both weights and activations are None, the layer will not be quantized and will retain
    its original precision. This is useful for excluding specific layers from quantization.

    The module_name pattern matching is applied first to the module's qualified graph name,
    then to the module's class name if no match is found.

    Examples
    --------
    Quantize all convolutional layers with int8 weights and activations::

        LayerQuantizationConfig(
            module_name="Conv*",
            weights=WeightQuantizationConfig(dtype="int8", observer_or_fake_quant="minmax"),
            activations=ActivationQuantizationConfig(dtype="int8", observer_or_fake_quant="minmax")
        )

    Quantize only weights of linear layers::

        LayerQuantizationConfig(
            module_name="Linear",
            weights=WeightQuantizationConfig(dtype="fp8_e4m3fn", observer_or_fake_quant="entropy")
        )

    Exclude a specific layer from quantization::

        LayerQuantizationConfig(
            module_name="final_layer",
            weights=None,
            activations=None
        )
    """

    module_name: str = STR_FIELD(  # type: ignore
        "",
        default_value="",
        description="Name or pattern specifying the module(s) to quantize",
        display_name="Module name",
        required="yes",
    )
    weights: Optional[WeightQuantizationConfig] = DATACLASS_FIELD(
        None,
        description="Weight quantization configuration. If None, weights are not quantized",
        display_name="Weight quantization",
    )
    activations: Optional[ActivationQuantizationConfig] = DATACLASS_FIELD(
        None,
        description="Activation quantization configuration. If None, activations are not quantized",
        display_name="Activation quantization",
    )


@dataclass
class ModelQuantizationConfig:
    """Top-level configuration for model quantization.

    Parameters
    ----------
    backend : str or None, optional
        The quantization backend to use. Valid options are "modelopt.pytorch", "torchao", and "modelopt.onnx".
        Defaults to "torchao".
    mode : str or None, optional
        The quantization mode to use. Valid options are "static_ptq" and "weight_only_ptq".
        Defaults to "weight_only_ptq".
    algorithm : str or None, optional
        Calibration/optimization algorithm name. Used by ModelOpt backends (both PyTorch and ONNX).
        Valid options:

        - For ``modelopt.pytorch``: "minmax", "max", "entropy"
        - For ``modelopt.onnx``: "max", "entropy", "awq_clip", "awq_lite", "awq_full", "rtn_dq"
        - For ``torchao``: Ignored

        Defaults to "minmax".
    layers : list of LayerQuantizationConfig, optional
        List of per-module quantization configurations. Each entry specifies how a particular module
        or set of modules should be quantized. This is the primary way to configure quantization.
        Defaults to an empty list.
    skip_names : list of str, optional
        List of module names or patterns to exclude from quantization. Each entry can be:

        - An exact module name (e.g., "layers.0.conv1")
        - A layer type (e.g., "Linear", "Conv2d")
        - A wildcard pattern (e.g., "conv*", "*.linear")

        Any module whose name matches any pattern in this list will be excluded from quantization.
        Defaults to an empty list.
    model_path : str, optional
        Path to the model to be quantized. For ONNX backend, this should be the path to the ONNX file.
        For PyTorch backends, this is typically set programmatically. Defaults to an empty string.
    results_dir : str, optional
        Path to where all the assets generated from a quantization task are stored. Defaults to an
        empty string.
    backend_kwargs : dict, optional
        Additional keyword arguments to pass to the backend. Backend-specific parameters can be
        provided here. Defaults to an empty dictionary.
    device : str, optional
        Device to use for calibration during quantization. Valid options include:

        - ``"cuda"``: Use the default GPU (automatically falls back to CPU if no GPU available)
        - ``"cpu"``: Force CPU execution
        - ``"cuda:0"``, ``"cuda:1"``, etc.: Use a specific GPU device
        - ``"trt"``: Use TensorRT execution provider (ONNX backend only)

        Defaults to "cuda".

        .. note::
           - For ``modelopt.pytorch`` backend: TRT falls back to CUDA for calibration
           - For ``modelopt.onnx`` backend: TRT uses TensorRT execution provider if available
           - For ``torchao`` backend: Device parameter is not used (weight-only quantization)

    Notes
    -----
    This is the main configuration class that orchestrates the entire quantization process.

    **Quantization Strategy:**

    Quantization is configured primarily through the ``layers`` parameter. Each
    ``LayerQuantizationConfig`` specifies which modules to quantize and their data types.
    Global defaults are not supported; you must explicitly configure each layer pattern.

    **Pattern Matching:**

    The ``module_name`` in ``LayerQuantizationConfig`` and entries in ``skip_names`` accept
    wildcard patterns interpreted using Python's ``fnmatch.fnmatch``. Wildcards ``*`` and ``?``
    are supported and matching is case-sensitive. Patterns are first applied to the module's
    qualified graph name (e.g., ``backbone.layer1.0.conv1``). If no match, the module's class
    name (e.g., ``Conv2d``) is checked.

    See ``nvidia_tao_pytorch.core.quantization.utils.match_layer`` for implementation details.

    Examples
    --------
    Basic weight-only INT8 quantization with TorchAO::

        config = ModelQuantizationConfig(
            backend="torchao",
            mode="weight_only_ptq",
            layers=[
                LayerQuantizationConfig(
                    module_name="*",  # All layers
                    weights=WeightQuantizationConfig(dtype="int8")
                )
            ],
            skip_names=["head", "final_layer"]
        )

    FP8 quantization with ModelOpt (weights + activations)::

        config = ModelQuantizationConfig(
            backend="modelopt.pytorch",
            mode="static_ptq",
            algorithm="max",
            layers=[
                LayerQuantizationConfig(
                    module_name="*",
                    weights=WeightQuantizationConfig(dtype="fp8_e4m3fn"),
                    activations=ActivationQuantizationConfig(dtype="fp8_e4m3fn")
                )
            ],
            skip_names=["BatchNorm*", "LayerNorm", "*head*"]
        )

    Mixed precision quantization::

        config = ModelQuantizationConfig(
            backend="modelopt.pytorch",
            mode="static_ptq",
            algorithm="entropy",
            layers=[
                # FP8 for most layers
                LayerQuantizationConfig(
                    module_name="*",
                    weights=WeightQuantizationConfig(dtype="fp8_e4m3fn"),
                    activations=ActivationQuantizationConfig(dtype="fp8_e4m3fn")
                ),
                # Keep specific layers in native precision
                LayerQuantizationConfig(
                    module_name="embedding",
                    weights=WeightQuantizationConfig(dtype="native")
                )
            ]
        )

    See Also
    --------
    LayerQuantizationConfig
        Configuration for individual layer quantization.
    WeightQuantizationConfig
        Configuration for weight quantization.
    ActivationQuantizationConfig
        Configuration for activation quantization.
    """

    backend: Optional[str] = STR_FIELD(  # type: ignore
        "torchao",
        default_value="torchao",
        valid_options="modelopt.pytorch,torchao,modelopt.onnx",
        description="The quantization backend to use",
        display_name="Quantization backend",
    )
    mode: Optional[str] = STR_FIELD(
        "weight_only_ptq",
        default_value="weight_only_ptq",
        valid_options="static_ptq,weight_only_ptq",
        description="The quantization mode to use",
        display_name="Quantization mode",
    )
    algorithm: Optional[str] = STR_FIELD(  # type: ignore
        "minmax",
        default_value="minmax",
        valid_options="minmax,max,entropy,awq_clip,awq_lite,awq_full,rtn_dq",
        description=(
            "Calibration/optimization algorithm. Used by ModelOpt backends "
            "(modelopt.pytorch and modelopt.onnx). Ignored by torchao backend."
        ),
        display_name="Calibration algorithm",
    )
    layers: Optional[List[LayerQuantizationConfig]] = LIST_FIELD(
        [],
        description=(
            "List of per-module quantization configurations. Each entry specifies which modules "
            "to quantize and their data types. This is the primary way to configure quantization."
        ),
        display_name="Layer quantization configs",
    )
    skip_names: Optional[List[str]] = LIST_FIELD(
        [],
        description="List of module/layer names or patterns to exclude from quantization",
        display_name="Skip names",
    )
    model_path: Optional[str] = STR_FIELD(
        "",
        default_value="",
        display_name="Model path",
        description="Path to the model to be quantized. For ONNX backend, path to ONNX file.",
        required="yes",
    )
    results_dir: Optional[str] = STR_FIELD(
        "",
        default_value="",
        display_name="Results directory",
        description="Path to where all the assets generated from a task are stored.",
        required="yes",
    )
    backend_kwargs: Optional[Dict[str, Any]] = DICT_FIELD(
        {},
        description="Additional keyword arguments to pass to the backend",
        display_name="Backend kwargs",
    )
    device: str = STR_FIELD(
        "cuda",
        default_value="cuda",
        regex=r"^(cuda|cpu|trt|cuda:[0-9]+)$",
        description=(
            "Device to use for calibration. Accepts 'cuda' (uses default GPU), 'cpu', "
            "'trt' (TensorRT for ONNX backend), or specific GPU device like 'cuda:0', 'cuda:1', etc. "
            "If 'cuda' is specified but no GPU is available, the framework will automatically fall back to 'cpu'. "
            "Note: 'trt' is only supported by the modelopt.onnx backend for ONNX Runtime with TensorRT "
            "execution provider."
        ),
        display_name="Calibration device",
    )


@dataclass
class QuantCalibrationDataset:
    """Quantization calibration dataset config."""

    images_dir: str = STR_FIELD(
        value="",
        default_value="",
        display_name="images directory",
        description="Path to images directory for quantization calibration",
    )
