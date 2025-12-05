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

"""ModelOpt ONNX quantization backend for TAO Toolkit.

Implements a backend that translates TAO quantization configuration to the
ModelOpt ONNX configuration and invokes ModelOpt ONNX APIs for quantization.
This backend works exclusively with ONNX files specified by file path.
"""

from __future__ import annotations

import os
import platform
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from nvidia_tao_core.config.common.quantization.default_config import ModelQuantizationConfig
from nvidia_tao_pytorch.core.quantization.backends.modelopt_onnx.utils import (
    convert_tao_to_modelopt_onnx_params,
    format_params_for_logging,
)
from nvidia_tao_pytorch.core.quantization.calibratable import Calibratable
from nvidia_tao_pytorch.core.quantization.constants import QuantizationMode
from nvidia_tao_pytorch.core.quantization.quantizer_base import FileBasedQuantizerBase
from nvidia_tao_pytorch.core.quantization.registry import register_backend
from nvidia_tao_pytorch.core.quantization.validation import validate_backend_mode_compatibility
from nvidia_tao_pytorch.core.tlt_logging import logger as tlt_logger

try:
    from modelopt.onnx.quantization import quantize as modelopt_onnx_quantize  # type: ignore
except ImportError:
    # ModelOpt ONNX is optional; provide a dummy function if unavailable
    def modelopt_onnx_quantize(*args, **kwargs):
        raise ImportError("modelopt.onnx.quantization is not available. Please install modelopt with ONNX support.")


SUPPORTED_MODES = {QuantizationMode.STATIC_PTQ.value}


def _ensure_cudnn_in_library_path():
    """Ensure cuDNN library path is in LD_LIBRARY_PATH for ModelOpt ONNX backend.

    ModelOpt ONNX performs strict filesystem checks for cuDNN in LD_LIBRARY_PATH,
    even though libraries may be accessible via ldconfig. This function adds the
    standard library path if not already present to prevent unnecessary fallback to CPU.

    This is particularly important for models with custom TensorRT plugins that
    require CUDA/TensorRT execution providers.

    Supports both x86_64 and ARM (aarch64) architectures.
    """
    arch = platform.machine()  # Returns 'x86_64' or 'aarch64'
    cudnn_path = f'/usr/lib/{arch}-linux-gnu'
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')

    if cudnn_path not in ld_library_path:
        tlt_logger.info(f"Adding {cudnn_path} to LD_LIBRARY_PATH for ModelOpt ONNX backend")
        os.environ['LD_LIBRARY_PATH'] = f"{cudnn_path}:{ld_library_path}" if ld_library_path else cudnn_path


@register_backend("modelopt.onnx")
class ModelOptONNXBackend(FileBasedQuantizerBase, Calibratable):
    """ModelOpt ONNX quantization backend for TAO Toolkit.

    This backend provides integration with NVIDIA ModelOpt ONNX quantization
    capabilities. It works exclusively with ONNX model files and translates
    TAO quantization configurations to ModelOpt ONNX parameters.

    The backend supports static post-training quantization (PTQ) and requires
    ONNX model files to be specified via file paths rather than PyTorch models.

    Parameters
    ----------
    backend_kwargs : dict, optional
        Additional keyword arguments to pass to the ModelOpt ONNX backend.
        These parameters will be merged with the converted TAO configuration
        parameters before calling ModelOpt ONNX quantization.

    Attributes
    ----------
    backend_name : str
        Name identifier for this backend ("modelopt_onnx").
    _logger : Logger
        TAO logging instance for this backend.
    _onnx_path : str, optional
        Path to the ONNX model file to be quantized.
    _backend_kwargs : dict
        Additional backend-specific parameters.
    _calibration_data : Any, optional
        Calibration data extracted from the provided data loader.

    Notes
    -----
    This backend requires the ModelOpt ONNX package to be installed. If not
    available, an ImportError will be raised when attempting to quantize.

    The backend only supports static post-training quantization (PTQ) mode.
    Dynamic quantization and quantization-aware training are not supported.

    Examples
    --------
    >>> from nvidia_tao_pytorch.core.quantization.backends.modelopt_onnx import ModelOptONNXBackend
    >>> from nvidia_tao_core.config.common.quantization.default_config import ModelQuantizationConfig
    >>>
    >>> # Create backend instance
    >>> backend = ModelOptONNXBackend()
    >>>
    >>> # Prepare for quantization (model must be None for ONNX backend)
    >>> config = ModelQuantizationConfig(model_path="/path/to/model.onnx")
    >>> backend.prepare(model=None, config=config)
    >>>
    >>> # Set calibration data
    >>> backend.set_calibration_data(calibration_data)
    >>>
    >>> # Quantize the model
    >>> backend.quantize(model=None, config=config)
    """

    def __init__(self, backend_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ModelOpt ONNX backend.

        Parameters
        ----------
        backend_kwargs : dict, optional
            Additional keyword arguments to pass to the ModelOpt ONNX backend.
            These parameters will be merged with the converted TAO configuration
            parameters before calling ModelOpt ONNX quantization.

        Returns
        -------
        None
        """
        self.backend_name = "modelopt.onnx"
        self._logger = tlt_logger

        self._onnx_path: Optional[str] = None
        self._backend_kwargs = backend_kwargs or {}
        self._calibration_data: Any = None
        self._config: Optional[ModelQuantizationConfig] = None

    def prepare(self, model: Optional[nn.Module], config: ModelQuantizationConfig) -> Optional[nn.Module]:
        """Prepare the backend for quantization using ONNX file from config.

        This method validates the configuration, checks the quantization mode,
        and validates the ONNX file path. The model parameter must be None
        for this backend as it works exclusively with ONNX files.

        Parameters
        ----------
        model : torch.nn.Module, optional
            PyTorch model (must be None for ONNX backend).
        config : ModelQuantizationConfig
            TAO quantization configuration containing ONNX file path and
            quantization parameters.

        Returns
        -------
        torch.nn.Module, optional
            Always returns None for ONNX backend.

        Raises
        ------
        TypeError
            If config is not an instance of ModelQuantizationConfig.
        ValueError
            If model is not None (ONNX backend requires model=None).
        ValueError
            If quantization mode is not supported (only static PTQ supported).
        ValueError
            If ONNX file path is not specified in config.
        FileNotFoundError
            If the specified ONNX file does not exist.
        ValueError
            If the file extension is not .onnx.

        Notes
        -----
        The ONNX file path must be specified in the 'model_path' field of the config.
        The file must exist and have a valid ONNX extension.
        """
        if not isinstance(config, ModelQuantizationConfig):
            raise TypeError("config must be an instance of ModelQuantizationConfig")

        if model is not None:
            raise ValueError("ONNX backend requires model=None. Provide ONNX file path via config.model_path instead.")

        # Validate mode using centralized validation
        mode_value = (
            config.mode.value
            if isinstance(config.mode, QuantizationMode)
            else str(config.mode).lower()
        )
        validate_backend_mode_compatibility(self.backend_name, mode_value, SUPPORTED_MODES)

        # Get and validate ONNX file path from model_path
        onnx_path = getattr(config, 'model_path', None)
        if not onnx_path:  # Check for None or empty string
            raise ValueError("ONNX file path must be specified in config.model_path")

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")

        if not onnx_path.lower().endswith(".onnx"):
            raise ValueError(f"Invalid file extension: {onnx_path}. Expected .onnx")

        self._onnx_path = onnx_path
        self._config = config

    def calibrate(self, model: Optional[nn.Module], data_loader) -> None:
        """Extract calibration data from dataloader for quantization.

        This method processes the provided data loader to extract calibration
        data that will be used during quantization. The calibration data is
        stored internally and used during the quantize() call.

        Parameters
        ----------
        model : torch.nn.Module, optional
            PyTorch model (must be None for ONNX backend).
        data_loader : DataLoader, optional
            PyTorch DataLoader containing calibration data. Can be None if
            no calibration data is needed.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If model is not None (ONNX backend requires model=None).

        Notes
        -----
        The calibration data is extracted from all batches in the data loader,
        converted to numpy arrays, and concatenated along the batch dimension.
        """
        if model is not None:
            raise ValueError("ONNX backend requires model=None. Provide ONNX file path via config.model_path instead.")

        if data_loader is None:
            self._calibration_data = None
        else:
            self._calibration_data = self._extract_calibration_data_from_dataloader(data_loader)

    def set_calibration_data(self, calibration_data) -> None:
        """Set calibration data directly without using a data loader.

        This method allows setting calibration data directly, bypassing
        the data loader extraction process. This is useful when calibration
        data is already available in the desired format.

        Parameters
        ----------
        calibration_data : numpy.ndarray or Dict[str, numpy.ndarray] or list
            Calibration data compatible with ModelOpt ONNX quantization.
            - For single-input models: numpy.ndarray with shape (N, C, H, W) or similar
            - For multi-input models: Dict[str, numpy.ndarray] mapping input names to arrays
            - Alternative: List of numpy.ndarray for sequential inputs

        Returns
        -------
        None

        Notes
        -----
        The calibration data should be in a format compatible with ModelOpt ONNX
        quantization requirements:
        - Single-input ONNX models: Pass numpy array directly
        - Multi-input ONNX models: Pass dict mapping ONNX input names to numpy arrays

        Examples
        --------
        Single-input model:
        >>> calibration_data = np.random.randn(100, 3, 224, 224)
        >>> backend.set_calibration_data(calibration_data)

        Multi-input model:
        >>> calibration_data = {
        ...     "input_ids": np.random.randint(0, 1000, (100, 128)),
        ...     "attention_mask": np.ones((100, 128))
        ... }
        >>> backend.set_calibration_data(calibration_data)
        """
        self._calibration_data = calibration_data

    def _extract_calibration_data_from_dataloader(self, data_loader):
        """Extract calibration data from PyTorch DataLoader.

        This private method processes a PyTorch DataLoader to extract
        calibration data in the format required by ModelOpt ONNX.
        It handles various batch formats and converts tensors to numpy arrays.

        Parameters
        ----------
        data_loader : DataLoader
            PyTorch DataLoader containing calibration data.

        Returns
        -------
        numpy.ndarray or None
            Extracted calibration data as numpy array, or None if no data found.

        Raises
        ------
        TypeError
            If input data is not a tensor or numpy array.
        ValueError
            If calibration data shapes are inconsistent.

        Notes
        -----
        Handles various batch formats:
        - List/tuple: Uses first element as input
        - Dict: Looks for common keys ('input', 'data', 'x', 'images',
                'images_left', 'images_right'), or uses first value
        - Tensor: Uses directly

        The method converts PyTorch tensors to numpy arrays and moves them
        to CPU before processing.
        """
        calibration_data = []
        total_images = 0
        num_batches = 0

        for batch in data_loader:
            num_batches += 1

            # Extract input data from batch
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            elif isinstance(batch, dict):
                # Try to extract input data from common keys
                inputs = None
                for key in ["input", "data", "x", "images", "images_left", "images_right"]:
                    if key in batch and batch[key] is not None:
                        inputs = batch[key]
                        break
                if inputs is None:
                    inputs = next(iter(batch.values()))
            else:
                inputs = batch

            # Convert to numpy with validation
            if torch.is_tensor(inputs):
                inputs_np = inputs.detach().cpu().numpy()
            elif isinstance(inputs, np.ndarray):
                inputs_np = inputs
            else:
                raise TypeError(
                    f"Unsupported calibration data type: {type(inputs)}. "
                    f"Expected torch.Tensor or numpy.ndarray."
                )

            calibration_data.append(inputs_np)

            # Count and log progress (every 10 batches)
            batch_size = inputs_np.shape[0] if len(inputs_np.shape) > 0 else 1
            total_images += batch_size

            if num_batches % 10 == 0:
                self._logger.info(f"  Processed {num_batches} batches ({total_images} images)...")

        self._logger.info(f"Extracted {total_images} calibration images from {num_batches} batches")

        # Convert list of arrays to a single concatenated array with error handling
        if len(calibration_data) == 1:
            return calibration_data[0]
        elif len(calibration_data) > 1:
            try:
                return np.concatenate(calibration_data, axis=0)
            except ValueError as e:
                shapes = [arr.shape for arr in calibration_data]
                raise ValueError(
                    f"Cannot concatenate calibration data due to inconsistent shapes: {shapes}. "
                    f"All batches must have the same shape except for batch dimension. "
                    f"Original error: {e}"
                ) from e
        else:
            return None

    def quantize(self, model: Optional[nn.Module], config: ModelQuantizationConfig) -> Optional[str]:
        """Quantize ONNX model using ModelOpt ONNX APIs.

        This method performs the actual quantization of the ONNX model using
        the ModelOpt ONNX quantization library. It converts the TAO configuration
        to ModelOpt ONNX parameters and calls the quantization function.

        Parameters
        ----------
        model : torch.nn.Module, optional
            PyTorch model (must be None for ONNX backend).
        config : ModelQuantizationConfig
            TAO quantization configuration containing quantization parameters.

        Returns
        -------
        str, optional
            Path to the quantized ONNX model file.

        Raises
        ------
        ValueError
            If model is not None (ONNX backend requires model=None).
        TypeError
            If config is not an instance of ModelQuantizationConfig.
        FileNotFoundError
            If the ONNX model file is not found.
            If the output file was not created after quantization.
        ImportError
            If ModelOpt ONNX package is not available.
        RuntimeError
            If ModelOpt quantization fails.

        Notes
        -----
        The quantized model is automatically saved to the output path specified
        in the configuration. The quantization process uses the calibration data
        that was set via calibrate() or set_calibration_data() methods.

        The method converts TAO configuration parameters to ModelOpt ONNX format
        and merges any additional backend-specific parameters before calling
        the ModelOpt ONNX quantization function.
        """
        if model is not None:
            raise ValueError("ONNX backend requires model=None. Provide ONNX file path via config.model_path instead.")

        if not isinstance(config, ModelQuantizationConfig):
            raise TypeError("config must be an instance of ModelQuantizationConfig")

        if self._onnx_path is None:
            raise RuntimeError(
                "No ONNX model path available. "
                "Call prepare() before quantize() to set up the ONNX file path."
            )

        if not os.path.exists(self._onnx_path):
            raise FileNotFoundError(f"ONNX model file not found: {self._onnx_path}")

        # Ensure cuDNN path is in LD_LIBRARY_PATH to avoid ModelOpt validation issues
        _ensure_cudnn_in_library_path()

        # Log calibration data status
        if self._calibration_data is None:
            self._logger.warning(
                "No calibration data provided. ModelOpt will generate dummy data which may reduce accuracy."
            )
        else:
            shape_info = self._calibration_data.shape if hasattr(self._calibration_data, 'shape') else type(self._calibration_data).__name__
            self._logger.info(f"Using calibration data with shape: {shape_info}")

        # Convert TAO config to ModelOpt ONNX parameters
        modelopt_params = convert_tao_to_modelopt_onnx_params(
            config, None, self._onnx_path, calibration_data=self._calibration_data, backend_kwargs=self._backend_kwargs
        )

        output_path = modelopt_params.get("output_path")

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            self._logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        # Log ModelOpt parameters for debugging (without full arrays)
        formatted_params = format_params_for_logging(modelopt_params)
        self._logger.debug(f"Calling ModelOpt ONNX quantization with parameters:\n{formatted_params}")

        # Log key quantization settings
        self._logger.info(
            f"Quantization configuration - Mode: {modelopt_params.get('quantize_mode')}, "
            f"Method: {modelopt_params.get('calibration_method')}, "
            f"Op types: {len(modelopt_params.get('op_types_to_quantize') or [])}"
        )

        # Call ModelOpt ONNX quantization with error handling
        try:
            self._logger.info("Starting ModelOpt ONNX quantization...")
            result = modelopt_onnx_quantize(**modelopt_params)

            # Log return value if any
            if result is not None:
                self._logger.debug(f"ModelOpt quantization returned: {result}")

        except Exception as e:
            error_msg = (
                f"ModelOpt ONNX quantization failed\n"
                f"  Input model: {self._onnx_path}\n"
                f"  Output path: {output_path}\n"
                f"  Error: {e}"
            )
            self._logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Verify output file was created
        if not os.path.exists(output_path):
            error_msg = (
                f"Quantized model was not created at expected path: {output_path}\n"
                f"ModelOpt quantization completed but output file is missing."
            )
            self._logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        file_size = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
        self._logger.info(f"Quantization complete. Model saved to: {output_path} ({file_size:.2f} MB)")

        return output_path

    def save_model(self, model: Optional[nn.Module], path: str) -> None:
        """Save model method (informational for ONNX backend).

        This method is provided for interface compatibility but does not
        perform any actual saving. The quantized ONNX model is automatically
        saved by the ModelOpt ONNX quantization process to the output path
        specified in the configuration.

        Parameters
        ----------
        model : torch.nn.Module or str or None
            For ONNX backend, this can be:
            - None: No-op, model already saved
            - str: Path returned from quantize(), no-op
            - nn.Module: Error, PyTorch models not supported
        path : str
            Path where the model would be saved (not used for ONNX backend).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If model is a PyTorch nn.Module (not supported by ONNX backend).

        Notes
        -----
        This method is provided for interface compatibility with the
        QuantizerBase interface. The actual model saving is handled
        automatically by the ModelOpt ONNX quantization process during
        the quantize() call.

        If the quantize() method's return value (a string path) is passed
        as the model argument, this method will simply log and return since
        the model is already saved.
        """
        # If model is a string (path returned from quantize), just log and return
        if isinstance(model, str):
            self._logger.info(f"Model already saved at: {model}")
            return

        # If model is None, that's fine - just a no-op
        if model is None:
            self._logger.debug("save_model() called with model=None (already saved during quantize)")
            return

        # If model is a PyTorch module, that's an error
        if isinstance(model, nn.Module):
            raise ValueError(
                "ONNX backend does not support saving PyTorch models. "
                "The quantized ONNX model is automatically saved during quantize()."
            )


__all__ = [
    "ModelOptONNXBackend",
]
