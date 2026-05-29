# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AdamW optimizer with step."""

from torch.optim import AdamW


class AdamWwStep(AdamW):
    """AdamW optimizer with step."""

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)

        for param_group in self.param_groups:
            param_group['step'] = 0
            param_group['epoch'] = 0

    def step(self, closure=None):
        """Step."""
        super().step(closure)
        for param_group in self.param_groups:
            param_group['step'] = param_group['step'] + 1

    def next_epoch(self):
        """Next epoch."""
        for param_group in self.param_groups:
            param_group['epoch'] += 1
