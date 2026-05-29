# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibratable interface for PTQ calibration."""

from abc import ABC, abstractmethod
import torch.nn as nn
from torch.utils.data import DataLoader


class Calibratable(ABC):
    """Abstract interface for PTQ calibration.

    Provides the method signature for post-training quantization calibration.
    Subclasses must implement ``calibrate``.
    """

    @abstractmethod
    def calibrate(self, model: nn.Module, data_loader: DataLoader) -> None:
        """Collect statistics or perform PTQ-style calibration.

        Parameters
        ----------
        model : torch.nn.Module
            Model to calibrate.
        data_loader : torch.utils.data.DataLoader
            Data loader providing calibration data.

        Returns
        -------
        None
            This method modifies the model or backend state in-place.
        """
        raise NotImplementedError("Subclasses must implement calibrate()")
