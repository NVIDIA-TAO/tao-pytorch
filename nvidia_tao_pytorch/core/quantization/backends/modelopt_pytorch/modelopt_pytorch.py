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

"""ModelOpt quantization backend for TAO Toolkit.

Implements a backend that translates TAO quantization configuration to the
ModelOpt configuration dict and invokes ``modelopt.torch.quantization``
APIs for calibration and quantization.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from nvidia_tao_core.config.common.quantization.default_config import ModelQuantizationConfig
from nvidia_tao_pytorch.core.quantization.backends.modelopt_pytorch.utils import convert_tao_to_modelopt_config
from nvidia_tao_pytorch.core.quantization.calibratable import Calibratable
from nvidia_tao_pytorch.core.quantization.constants import QuantizationMode
from nvidia_tao_pytorch.core.quantization.quantizer_base import PyTorchQuantizerBase
from nvidia_tao_pytorch.core.quantization.registry import register_backend
from nvidia_tao_pytorch.core.quantization.validation import validate_model, validate_backend_mode_compatibility
from nvidia_tao_pytorch.core.tlt_logging import logger as tlt_logger

try:
    import modelopt.torch.opt as mto
    import modelopt.torch.quantization as mtq  # type: ignore
except Exception as exc:  # pragma: no cover - import error path
    raise ImportError(
        "modelopt is not installed or failed to import. Install ModelOpt (pip install nvidia-modelopt) "
        "to use the 'modelopt' backend."
    ) from exc


SUPPORTED_MODES = {QuantizationMode.STATIC_PTQ.value}


def _default_forward_loop(model: nn.Module) -> None:
    """No-op forward loop used when no calibration data is provided.

    Parameters
    ----------
    model : torch.nn.Module
        Model to run through a dummy evaluation pass.
    """
    model.eval()
    with torch.no_grad():
        return


@register_backend("modelopt.pytorch")
class ModelOptBackend(PyTorchQuantizerBase, Calibratable):
    """ModelOpt quantization backend.

    Adapts the TAO quantization configuration to the ModelOpt configuration
    dictionary and delegates quantizer insertion and calibration to
    ``modelopt.torch.quantization`` APIs.
    """

    def __init__(self, backend_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self._forward_loop: Optional[Callable[[nn.Module], None]] = None
        self.backend_name = "modelopt.pytorch"  # Store the backend name as an instance attribute
        self._logger = tlt_logger
        self._backend_kwargs = backend_kwargs or {}
        self._config: Optional[ModelQuantizationConfig] = None

    def prepare(self, model: nn.Module, config: ModelQuantizationConfig) -> nn.Module:
        """Validate inputs and return the model unchanged.

        ModelOpt handles quantizer insertion inside its ``quantize`` API, so
        ``prepare`` is effectively a no-op for this backend.

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

        # Store config for use in calibrate
        self._config = config
        return model

    def calibrate(self, model: nn.Module, data_loader) -> None:
        """Build and store a forward loop for later calibration.

        Parameters
        ----------
        model : torch.nn.Module
            Model to be calibrated.
        data_loader : torch.utils.data.DataLoader
            Iterator producing inputs. Each batch can be a tensor, a tuple where the first
            element is the tensor, or a dict with key "input".
        """
        validate_model(model)

        # Get device from config
        device_str = getattr(self._config, 'device', 'cuda') if self._config else 'cuda'

        # Handle device validation and fallback
        if device_str == 'trt':
            self._logger.warning(
                "TensorRT (TRT) device is not supported for ModelOpt PyTorch calibration. "
                "Falling back to CUDA for calibration. TensorRT will be used during inference/export."
            )
            device_str = 'cuda'

        # Validate and set device with automatic CPU fallback for CUDA
        if device_str.startswith('cuda'):
            if not torch.cuda.is_available():
                self._logger.warning(
                    f"CUDA device '{device_str}' requested but CUDA is not available. "
                    f"Falling back to CPU for calibration."
                )
                device = torch.device('cpu')
            else:
                try:
                    device = torch.device(device_str)
                    # Verify the specific GPU is available if specified
                    if ':' in device_str:
                        try:
                            gpu_id = int(device_str.split(':')[1])
                        except ValueError:
                            raise ValueError(f"Invalid GPU ID in device string '{device_str}'")
                        if gpu_id >= torch.cuda.device_count():
                            self._logger.warning(
                                f"GPU {gpu_id} specified but only {torch.cuda.device_count()} GPU(s) available. "
                                f"Falling back to default CUDA device."
                            )
                            device = torch.device('cuda:0')
                except Exception as e:
                    self._logger.warning(f"Invalid device '{device_str}': {e}. Falling back to CPU.")
                    device = torch.device('cpu')
        else:
            device = torch.device(device_str)

        self._logger.info(f"Using device '{device}' for calibration")

        def _extract_input(batch):
            if torch.is_tensor(batch):
                return batch
            if isinstance(batch, (list, tuple)) and len(batch) > 0:
                return batch[0]
            if isinstance(batch, dict):
                # Common convention
                return batch.get("input", next(iter(batch.values())))
            raise TypeError(
                f"Unsupported batch format for calibration. Expected a Tensor, (Tensor, ...), or dict,"
                f" but got type {type(batch).__name__}."
            )

        def forward_loop(m: nn.Module) -> None:
            if len(data_loader) == 0:
                self._logger.warning(
                    "No calibration data present in the calibration dataset. "
                    "Accuracy of the quantized model may degrade if activations are being quantized. "
                    "Please check the calibration dataset `quant_calibration_dataset` if this is a problem. "
                    "Weight only quantization will not be affected. "
                    "Disregard this warning if you are running evaluation, inference, or other non-quantization tasks."
                )
            m.eval().to(device)
            with torch.no_grad():
                for batch in tqdm(data_loader, desc="Calibrating model"):
                    x = _extract_input(batch).to(device)
                    m(x)
        self._forward_loop = forward_loop

    def quantize(self, model: nn.Module, config: ModelQuantizationConfig) -> nn.Module:
        """Quantize a model using ModelOpt APIs.

        Translates the TAO configuration into a ModelOpt configuration dict and
        invokes ``modelopt.torch.quantization.quantize`` with the stored forward
        loop if available.

        Parameters
        ----------
        model : torch.nn.Module
            Prepared model to quantize.
        config : ModelQuantizationConfig
            Quantization configuration.

        Returns
        -------
        torch.nn.Module
            Quantized model.
        """
        validate_model(model)
        if not isinstance(config, ModelQuantizationConfig):
            raise TypeError("config must be an instance of ModelQuantizationConfig")

        modelopt_cfg = convert_tao_to_modelopt_config(config, model)

        # ModelOpt configuration prepared; proceed to quantization

        if self._forward_loop is None:
            self._logger.warning(
                "No calibration dataloader provided; using a no-op forward loop. "
                "Accuracy of the quantized model may degrade if activations are being quantized"
                "and no calibration data is provided. Weight only quantization will not be affected. "
                "Disregard this warning if you are running evaluation, inference, or other non-quantization tasks."
            )
            forward_loop = _default_forward_loop
        else:
            forward_loop = self._forward_loop

        quantized_model = mtq.quantize(model, modelopt_cfg, forward_loop)
        return quantized_model

    def save_model(self, model: nn.Module, path: str) -> None:
        """Save the quantized model to a directory.

        Parameters
        ----------
        model : torch.nn.Module
            Quantized model to save.
        path : str
            Directory where the model is saved as ``quantized_model_modelopt.pth``.

        Raises
        ------
        TypeError
            If model is not a torch.nn.Module or path is not a string.
        ValueError
            If path is empty.
        """
        validate_model(model)
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        if not path:
            raise ValueError("path cannot be empty")

        os.makedirs(path, exist_ok=True)
        quantized_model_path = os.path.join(path, f"quantized_model_{self.backend_name}.pth")

        try:
            mto.save(model, quantized_model_path)
            self._logger.info(f"Quantized model saved to: {quantized_model_path}")
        except Exception as e:
            error_msg = f"Failed to save model to {quantized_model_path}: {e}"
            self._logger.error(error_msg)
            raise RuntimeError(error_msg) from e


__all__ = [
    "ModelOptBackend",
]
