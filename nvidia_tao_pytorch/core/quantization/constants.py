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

"""Constants for the quantization framework."""

from enum import Enum, auto


class SupportedDtype(Enum):
    """Supported data types for quantization."""

    INT8 = "int8"
    FP8_E4M3FN = "fp8_e4m3fn"
    FP8_E5M2 = "fp8_e5m2"
    NATIVE = "native"  # Special value to disable quantization for a layer


class QuantizationMode(Enum):
    """Supported quantization modes."""

    WEIGHT_ONLY_PTQ = "weight_only_ptq"
    STATIC_PTQ = "static_ptq"

    @classmethod
    def from_string(cls, mode: str) -> "QuantizationMode":
        """Convert string to QuantizationMode enum.

        Parameters
        ----------
        mode : str
            Mode string (case-insensitive)

        Returns
        -------
        QuantizationMode
            Corresponding enum value

        Raises
        ------
        ValueError
            If mode string is not recognized
        """
        mode_lower = str(mode).lower()
        for member in cls:
            if member.value == mode_lower:
                return member
        raise ValueError(
            f"Unsupported mode '{mode}'. "
            f"Supported modes: {[m.value for m in cls]}"
        )


class QuantizationState(Enum):
    """States in the quantization workflow.

    The quantization workflow follows this state machine:
    INITIALIZED -> PREPARED -> [CALIBRATED] -> QUANTIZED

    States:
        INITIALIZED: Quantizer created with configuration
        PREPARED: Model prepared for quantization (observers/fake-quants inserted)
        CALIBRATED: Model calibrated with representative data (optional)
        QUANTIZED: Model quantized and ready for use
    """

    INITIALIZED = auto()
    PREPARED = auto()
    CALIBRATED = auto()
    QUANTIZED = auto()


class BackendType(Enum):
    """Registered quantization backend types."""

    MODELOPT_PYTORCH = "modelopt.pytorch"
    MODELOPT_ONNX = "modelopt.onnx"
    TORCHAO = "torchao"

    @classmethod
    def from_string(cls, backend: str) -> "BackendType":
        """Convert string to BackendType enum.

        Parameters
        ----------
        backend : str
            Backend string (case-sensitive)

        Returns
        -------
        BackendType
            Corresponding enum value

        Raises
        ------
        ValueError
            If backend string is not recognized
        """
        for member in cls:
            if member.value == backend:
                return member
        raise ValueError(
            f"Unsupported backend '{backend}'. "
            f"Supported backends: {[b.value for b in cls]}"
        )


# Constant strings for common use
DTYPE_NORMALIZATION_MAP = {
    "float8_e4m3fn": "fp8_e4m3fn",
    "float8_e5m2": "fp8_e5m2",
}
