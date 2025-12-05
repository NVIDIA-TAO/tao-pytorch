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

"""Quantizer core interfaces for TAO Toolkit.

This module defines the base interfaces for quantization backends. It provides
two specialized interfaces for different quantization paradigms:

1. PyTorchQuantizerBase: For backends that operate on PyTorch nn.Module objects
2. FileBasedQuantizerBase: For backends that operate on serialized model files (e.g., ONNX)

The split allows for type-safe interfaces that accurately reflect the requirements
of each backend type.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
import torch.nn as nn

from nvidia_tao_core.config.common.quantization.default_config import (
    ModelQuantizationConfig,
)


class QuantizerBase(ABC):
    """Base interface for all quantization backends.

    This is the root interface that all quantization backends must implement.
    Subclasses should inherit from one of the specialized interfaces:
    - PyTorchQuantizerBase for PyTorch model-based backends
    - FileBasedQuantizerBase for file-based backends (e.g., ONNX)

    The methods accept and return Union types to accommodate both paradigms,
    but subclasses use more specific types.

    See Also
    --------
    PyTorchQuantizerBase : For PyTorch nn.Module-based quantization
    FileBasedQuantizerBase : For file-based quantization (ONNX, etc.)
    Calibratable : Mix-in interface for PTQ calibration support
    """

    @abstractmethod
    def prepare(
        self,
        model: Optional[nn.Module],
        config: ModelQuantizationConfig
    ) -> Optional[nn.Module]:
        """Prepare a model or file for quantization.

        Parameters
        ----------
        model : torch.nn.Module or None
            Model to prepare. PyTorch backends require nn.Module,
            file-based backends require None.
        config : ModelQuantizationConfig
            Quantization configuration.

        Returns
        -------
        torch.nn.Module or None
            Prepared model for PyTorch backends, None for file-based backends.

        Notes
        -----
        Subclasses should use more specific type signatures. See
        PyTorchQuantizerBase and FileBasedQuantizerBase for examples.
        """
        pass

    @abstractmethod
    def quantize(
        self,
        model: Optional[nn.Module],
        config: ModelQuantizationConfig
    ) -> Union[nn.Module, str, None]:
        """Quantize a prepared model or file.

        Parameters
        ----------
        model : torch.nn.Module or None
            Prepared model to quantize. PyTorch backends require nn.Module,
            file-based backends require None.
        config : ModelQuantizationConfig
            Quantization configuration.

        Returns
        -------
        torch.nn.Module or str or None
            - PyTorch backends: quantized nn.Module
            - File-based backends: output file path (str) or None

        Notes
        -----
        Subclasses should use more specific type signatures. See
        PyTorchQuantizerBase and FileBasedQuantizerBase for examples.
        """
        pass


class PyTorchQuantizerBase(QuantizerBase):
    """Base interface for PyTorch model-based quantization backends.

    This interface is for backends that operate directly on PyTorch nn.Module
    objects, such as ModelOpt PyTorch and TorchAO. It provides type-safe
    signatures that require nn.Module inputs and outputs.

    Examples
    --------
    >>> class MyPyTorchBackend(PyTorchQuantizerBase):
    ...     def prepare(self, model: nn.Module, config: ModelQuantizationConfig) -> nn.Module:
    ...         # Insert observers/fake quantizers
    ...         return model
    ...
    ...     def quantize(self, model: nn.Module, config: ModelQuantizationConfig) -> nn.Module:
    ...         # Convert to quantized model
    ...         return quantized_model

    See Also
    --------
    FileBasedQuantizerBase : For file-based quantization backends
    Calibratable : Mix-in for calibration support
    """

    @abstractmethod
    def prepare(self, model: nn.Module, config: ModelQuantizationConfig) -> nn.Module:
        """Prepare a PyTorch model for quantization.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model to prepare for quantization.
        config : ModelQuantizationConfig
            Quantization configuration.

        Returns
        -------
        torch.nn.Module
            Prepared model with observers/fake-quant modules inserted.
        """
        pass

    @abstractmethod
    def quantize(self, model: nn.Module, config: ModelQuantizationConfig) -> nn.Module:
        """Convert a prepared PyTorch model to its quantized form.

        Parameters
        ----------
        model : torch.nn.Module
            Prepared PyTorch model to quantize.
        config : ModelQuantizationConfig
            Quantization configuration.

        Returns
        -------
        torch.nn.Module
            Quantized model.
        """
        pass


class FileBasedQuantizerBase(QuantizerBase):
    """Base interface for file-based quantization backends.

    This interface is for backends that operate on serialized model files
    (e.g., ONNX) rather than PyTorch nn.Module objects. These backends
    accept None for model parameters and work with file paths specified
    in the configuration.

    Examples
    --------
    >>> class MyONNXBackend(FileBasedQuantizerBase):
    ...     def prepare(self, model: None, config: ModelQuantizationConfig) -> None:
    ...         # Validate config.model_path points to valid ONNX file
    ...         return None
    ...
    ...     def quantize(self, model: None, config: ModelQuantizationConfig) -> str:
    ...         # Quantize ONNX file and return output path
    ...         return "/path/to/quantized_model.onnx"

    See Also
    --------
    PyTorchQuantizerBase : For PyTorch model-based backends
    Calibratable : Mix-in for calibration support
    """

    @abstractmethod
    def prepare(self, model: None, config: ModelQuantizationConfig) -> None:
        """Prepare for file-based quantization.

        This method typically validates that config.model_path points to a
        valid file and performs any necessary preprocessing.

        Parameters
        ----------
        model : None
            Must be None for file-based backends.
        config : ModelQuantizationConfig
            Quantization configuration. Must contain valid model_path.

        Returns
        -------
        None
            File-based backends don't return a model object.

        Raises
        ------
        ValueError
            If model is not None.
        ValueError
            If config.model_path is not set or invalid.
        """
        pass

    @abstractmethod
    def quantize(self, model: None, config: ModelQuantizationConfig) -> Optional[str]:
        """Quantize a model file.

        Parameters
        ----------
        model : None
            Must be None for file-based backends.
        config : ModelQuantizationConfig
            Quantization configuration containing model_path and output settings.

        Returns
        -------
        str or None
            Path to the quantized output file, or None if output path is
            determined by configuration.

        Raises
        ------
        ValueError
            If model is not None.
        """
        pass
