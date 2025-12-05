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
