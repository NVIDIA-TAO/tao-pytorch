# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RADIO model wrapper for CRADIO or ERADIO
"""

from torch import nn


class RADIOWrapper(nn.Module):
    """RADIO modwl wrapper for C-RADIO and E-RADIO
    """

    def __init__(self, model: nn.Module, resolution: tuple = None):
        """RADIO modwl wrapper for C-RADIO and E-RADIO.

        Args:
            model (nn.Module): RADIO model
            resolution (tuple, optional): input resolution. Defaults to None.
        """
        super().__init__()
        self.radio = model

        if resolution is not None:
            self._validate_input(resolution)

    @property
    def num_summary_tokens(self) -> int:
        """Total number of extra tokens (class tokens + register tokens)
        """
        return self.radio.num_summary_tokens

    @property
    def patch_size(self) -> int:
        """Patch size
        """
        return self.radio.patch_size

    @property
    def window_size(self) -> int:
        """Window size
        """
        return self.radio.window_size

    @property
    def min_resolution_step(self) -> int:
        """Minimum acceptable patch size
        """
        res = self.radio.patch_size
        if self.radio.window_size is not None:
            res *= self.radio.window_size

        return res

    def _get_nearest_supported_resolution(self, height: int, width: int):
        height = int(round(height / self.min_resolution_step) * self.min_resolution_step)
        width = int(round(width / self.min_resolution_step) * self.min_resolution_step)

        height = max(height, self.min_resolution_step)
        width = max(width, self.min_resolution_step)

        return height, width

    def _validate_input(self, resolution):
        res_step = self.min_resolution_step
        if res_step is not None and (resolution[0] % res_step != 0 or resolution[1] % res_step != 0):
            raise ValueError('The input resolution must be a multiple of `self.min_resolution_step`. '
                             f'Input: {resolution}, Nearest: {self._get_nearest_supported_resolution(resolution[0], resolution[1])}')

    def forward(self, x):
        """
        Forward function and return the features
        """
        return self.radio(x)
