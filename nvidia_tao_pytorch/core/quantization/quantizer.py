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
"""Core quantization orchestration utilities for TAO Toolkit.

Exposes the :class:`ModelQuantizer` entry point that coordinates prepare,
optional calibration, quantization, and saving using a pluggable backend.
"""

from __future__ import annotations

import importlib
from typing import Any, Optional, Type, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nvidia_tao_core.config.common.quantization.default_config import ModelQuantizationConfig
from nvidia_tao_pytorch.core.quantization.calibratable import Calibratable
from nvidia_tao_pytorch.core.quantization.constants import QuantizationState
from nvidia_tao_pytorch.core.quantization.quantizer_base import (
    QuantizerBase,
    PyTorchQuantizerBase,
    FileBasedQuantizerBase,
)
from nvidia_tao_pytorch.core.quantization.registry import get_available_backends, get_backend_class
from nvidia_tao_pytorch.core.quantization.utils import build_model_quant_config_from_omegaconf
from nvidia_tao_pytorch.core.quantization.validation import validate_model
from nvidia_tao_pytorch.core.tlt_logging import logger

# Import backends package for side-effect registration of available backends
try:
    importlib.import_module("nvidia_tao_pytorch.core.quantization.backends")
except Exception:
    # Backends are optional; ignore if unavailable at import time
    pass


def _coerce_to_model_quant_config(cfg_like: Any) -> ModelQuantizationConfig:
    if isinstance(cfg_like, ModelQuantizationConfig):
        return cfg_like
    # Accept OmegaConf DictConfig or plain dict
    return build_model_quant_config_from_omegaconf(cfg_like)


class ModelQuantizer:
    """High-level interface to run the quantization pipeline.

    This class maintains quantization context (backend, config, etc.) and exposes
    methods to perform the full pipeline: prepare, optionally calibrate, quantize,
    and save.

    The workflow follows a state machine:
    INITIALIZED -> PREPARED -> [CALIBRATED] -> QUANTIZED

    Examples
    --------
    Basic usage:
        >>> from nvidia_tao_pytorch.core.quantization import ModelQuantizer
        >>>
        >>> # Create quantizer with config
        >>> config = {"backend": "modelopt.pytorch", "layers": [...]}
        >>> quantizer = ModelQuantizer(config)
        >>>
        >>> # Full pipeline
        >>> quantized_model = quantizer.quantize_model(model)
        >>> quantizer.save_model(path="quantized_model.pth")

    Step-by-step usage with calibration:
        >>> # Prepare model
        >>> prepared_model = quantizer.prepare(model)
        >>>
        >>> # Calibrate (if backend supports it)
        >>> quantizer.calibrate(prepared_model, calibration_dataloader)
        >>>
        >>> # Quantize
        >>> quantized_model = quantizer.quantize()
        >>>
        >>> # Save
        >>> quantizer.save_model(path="quantized_model.pth")
    """

    def __init__(self, cfg_like: Any):
        """Initialize the quantizer with configuration.

        Parameters
        ----------
        cfg_like : Any
            Configuration for quantization. Accepts a ``ModelQuantizationConfig``,
            an OmegaConf ``DictConfig``, or a plain ``dict``.

        Raises
        ------
        ValueError
            If backend is not specified or not available.
        TypeError
            If backend is not a subclass of QuantizerBase.
        """
        self.config = _coerce_to_model_quant_config(cfg_like)

        if not self.config.backend:
            raise ValueError("Quantization backend must be specified in the config (e.g., 'modelopt').")

        # Check backend availability before attempting to use it
        available_backends = get_available_backends()
        if self.config.backend not in available_backends:
            raise ValueError(
                f"Backend '{self.config.backend}' is not available. "
                f"Available backends: {available_backends}. "
                f"Make sure the required backend package is installed "
                f"(e.g., 'pip install nvidia-modelopt' for modelopt backends, "
                f"'pip install torchao' for torchao backend)."
            )

        try:
            self.backend_class: Type[QuantizerBase] = get_backend_class(self.config.backend)
        except ValueError as e:
            # Provide more context in error message
            raise ValueError(
                f"Failed to load backend '{self.config.backend}': {e}. "
                f"Available backends: {available_backends}"
            ) from e

        if not issubclass(self.backend_class, QuantizerBase):
            raise TypeError(
                f"Backend '{self.config.backend}' must be a subclass of QuantizerBase, "
                f"but got {self.backend_class}"
            )
        self.quantizer: QuantizerBase = self.backend_class(self.config.backend_kwargs)

        # Initialize state machine
        self._state: QuantizationState = QuantizationState.INITIALIZED

        logger.info(f"Initialized quantization with backend: {self.config.backend}")

    def prepare(self, model: Optional[nn.Module]) -> Optional[nn.Module]:
        """Prepare the model for quantization.

        Parameters
        ----------
        model : torch.nn.Module or None
            Model to prepare for quantization. Can be None for backends that
            work with file paths (e.g., ONNX backend).

        Returns
        -------
        torch.nn.Module or None
            Prepared model (e.g., after observer/fake-quant insertion depending on backend).
            Returns None for file-based backends.

        Raises
        ------
        TypeError
            If model type doesn't match backend expectations.
        RuntimeError
            If called in an invalid state (must be INITIALIZED).
        ValueError
            If model is provided but backend doesn't accept PyTorch models, or vice versa.
        """
        # State validation
        if self._state != QuantizationState.INITIALIZED:
            raise RuntimeError(
                f"prepare() called in invalid state '{self._state.name}'. "
                f"Expected state: INITIALIZED. "
                f"Hint: Create a new ModelQuantizer instance for a new quantization workflow."
            )

        # Validate model matches backend expectations
        if isinstance(self.quantizer, PyTorchQuantizerBase):
            if model is None:
                raise ValueError(
                    f"Backend '{self.config.backend}' requires a PyTorch model, but model=None was provided. "
                    f"Expected: PyTorch nn.Module instance"
                )
        elif isinstance(self.quantizer, FileBasedQuantizerBase):
            if model is not None:
                raise ValueError(
                    f"Backend '{self.config.backend}' is file-based and does not accept PyTorch models. "
                    f"Expected: model=None, specify file path in config.model_path. "
                    f"Hint: For ONNX quantization, export your model first, then set config.model_path"
                )

        # Validate model type if provided
        if model is not None:
            validate_model(model)

        prepared_model = self.quantizer.prepare(model, self.config)
        self._state = QuantizationState.PREPARED
        logger.debug("State transition: INITIALIZED -> PREPARED")
        return prepared_model

    def calibrate(self, model: Optional[nn.Module], dataloader: DataLoader) -> None:
        """Calibrate a prepared model using the provided data loader.

        Parameters
        ----------
        model : torch.nn.Module or None
            Prepared model to calibrate. Can be None for file-based backends.
        dataloader : torch.utils.data.DataLoader
            Data loader providing calibration samples.

        Raises
        ------
        RuntimeError
            If called in an invalid state (must be PREPARED).

        Notes
        -----
        If the selected backend does not support calibration, this call is a no-op with a warning.
        """
        # State validation
        if self._state not in (QuantizationState.PREPARED,):
            raise RuntimeError(
                f"calibrate() called in invalid state '{self._state.name}'. "
                f"Expected state: PREPARED. "
                f"Hint: Call prepare() before calibrate()."
            )

        if isinstance(self.quantizer, Calibratable):
            self.quantizer.calibrate(model, dataloader)
            self._state = QuantizationState.CALIBRATED
            logger.debug("State transition: PREPARED -> CALIBRATED")
        else:
            logger.warning(
                f"Backend '{self.config.backend}' does not support calibration; skipping calibration phase"
            )
            # Stay in PREPARED state if backend doesn't support calibration

    def quantize(self, model: Optional[nn.Module]) -> Union[nn.Module, str, None]:
        """Quantize the prepared model.

        Parameters
        ----------
        model : torch.nn.Module or None
            Prepared model to quantize. Must match backend expectations:
            - PyTorch backends (modelopt.pytorch, torchao): Require nn.Module
            - File-based backends (modelopt.onnx): Require None

        Returns
        -------
        torch.nn.Module or str or None
            The return type depends on the backend:
            - PyTorch backends: Return quantized nn.Module
            - File-based backends: Return output file path (str) or None

        Raises
        ------
        RuntimeError
            If called in an invalid state (must be PREPARED or CALIBRATED).
        ValueError
            If model parameter doesn't match backend requirements.
        """
        # State validation
        if self._state not in (QuantizationState.PREPARED, QuantizationState.CALIBRATED):
            raise RuntimeError(
                f"quantize() called in invalid state '{self._state.name}'. "
                f"Expected state: PREPARED or CALIBRATED. "
                f"Hint: Call prepare() (and optionally calibrate()) before quantize()."
            )

        quantized_model = self.quantizer.quantize(model, self.config)
        self._state = QuantizationState.QUANTIZED
        logger.debug("State transition: PREPARED/CALIBRATED -> QUANTIZED")
        return quantized_model

    def save_model(self, model: Optional[nn.Module], path: str) -> None:
        """Save a quantized model to disk.

        Parameters
        ----------
        model : torch.nn.Module or None
            Model instance to save. Can be None for backends that manage output internally.
        path : str
            Directory path where the model should be saved.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If called before quantize().
        ValueError
            If model is None for backends that require it, or path is empty.
        """
        # State validation
        if self._state != QuantizationState.QUANTIZED:
            raise RuntimeError(
                f"save_model() called in invalid state '{self._state.name}'. "
                f"Expected state: QUANTIZED. "
                f"Hint: Call quantize() before save_model()."
            )

        if not path:
            raise ValueError("path cannot be empty")

        # PyTorch backends require a model for saving
        if isinstance(self.quantizer, PyTorchQuantizerBase) and model is None:
            raise ValueError(
                f"PyTorch backend '{self.config.backend}' requires a model for saving, but model=None was provided. "
                "Pass the quantized model returned from quantize()."
            )

        if hasattr(self.quantizer, 'save_model'):
            self.quantizer.save_model(model, path)
        else:
            # Fallback to torch.save if backend doesn't provide save_model
            if model is None:
                raise ValueError("Cannot save: backend has no save_model method and model is None")
            torch.save(model.state_dict(), path)

        logger.debug(f"Model saved successfully in state: {self._state.name}")

    def quantize_model(
        self,
        model: Optional[nn.Module],
        calibration_loader: Optional[DataLoader] = None
    ) -> Union[nn.Module, str, None]:
        """Run the end-to-end quantization pipeline and return the quantized model.

        This convenience method automatically handles the complete workflow:
        prepare -> [calibrate] -> quantize

        The quantization process modifies the model in-place to minimize memory usage.
        State transitions are managed automatically.

        Parameters
        ----------
        model : torch.nn.Module or None
            Model to quantize:
            - PyTorch backends (modelopt.pytorch, torchao): Pass nn.Module
            - File-based backends (modelopt.onnx): Pass None, use config.model_path
        calibration_loader : torch.utils.data.DataLoader, optional
            Calibration data loader. If provided and the backend supports calibration,
            calibration is performed.

        Returns
        -------
        torch.nn.Module or str or None
            Return type depends on backend:
            - PyTorch backends: quantized nn.Module (modified in-place)
            - File-based backends: output file path (str) or None

        Raises
        ------
        RuntimeError
            If called in an invalid state (must be INITIALIZED).
        ValueError
            If model parameter doesn't match backend requirements.

        Notes
        -----
        Different backends have different requirements:
        - PyTorch backends (modelopt.pytorch, torchao): Require nn.Module, return quantized nn.Module
        - ONNX backend (modelopt.onnx): Require model=None, use config.model_path for input

        The input model is modified in-place during quantization to minimize peak memory usage.
        This is particularly beneficial for large models (multi-GB) as it avoids creating
        intermediate copies that would otherwise triple memory consumption.

        Examples
        --------
        Basic usage:
            >>> quantizer = ModelQuantizer(config)
            >>> quantized_model = quantizer.quantize_model(model)
            >>> # Note: 'model' has been modified in-place

        With calibration:
            >>> quantizer = ModelQuantizer(config)
            >>> quantized_model = quantizer.quantize_model(model, calibration_loader=loader)
        """
        # Ensure we're starting from the right state
        if self._state != QuantizationState.INITIALIZED:
            raise RuntimeError(
                f"quantize_model() called in invalid state '{self._state.name}'. "
                f"Expected state: INITIALIZED. "
                f"Hint: Create a new ModelQuantizer instance for each model."
            )

        # Validate model matches backend expectations
        if isinstance(self.quantizer, PyTorchQuantizerBase):
            if model is None:
                raise ValueError(
                    f"Backend '{self.config.backend}' requires a PyTorch model, but model=None was provided. "
                    f"Expected: PyTorch nn.Module instance"
                )
        elif isinstance(self.quantizer, FileBasedQuantizerBase):
            if model is not None:
                raise ValueError(
                    f"Backend '{self.config.backend}' is file-based and does not accept PyTorch models. "
                    f"You provided a PyTorch model of type {type(model).__name__}.\n"
                    f"For file-based quantization (e.g., ONNX):\n"
                    f"  1. Export your model to the required format first\n"
                    f"  2. Set config.model_path to the file path\n"
                    f"  3. Call quantize_model(model=None, calibration_loader=...)\n"
                    f"Current config.model_path: {getattr(self.config, 'model_path', 'NOT SET')}"
                )

        # Standard workflow that works for all backends
        # Models are modified in-place to minimize memory usage
        prepared_model = self.prepare(model)

        # Calibrate if supported and requested
        if calibration_loader is not None and isinstance(self.quantizer, Calibratable):
            self.calibrate(prepared_model, calibration_loader)

        quantized_model = self.quantize(prepared_model)

        return quantized_model

    @property
    def state(self) -> QuantizationState:
        """Get the current state of the quantization workflow.

        Returns
        -------
        QuantizationState
            Current state (INITIALIZED, PREPARED, CALIBRATED, or QUANTIZED).
        """
        return self._state
