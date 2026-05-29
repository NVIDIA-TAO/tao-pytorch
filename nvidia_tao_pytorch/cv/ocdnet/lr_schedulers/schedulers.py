#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Popular Learning Rate Schedulers"""
from __future__ import division

import torch


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    """WarmupPolyLR.

    Decays the learning rate via a polynomial function. The learning rate increases to initial value
    during warmup stage and is reduced from the initial value to zero during the training stage.
    """

    def __init__(self, optimizer, target_lr=0, max_iters=0, power=0.9, warmup_factor=1.0 / 3, warmup_iters=500,
                 warmup_method='linear', last_epoch=-1, warmup_epochs=0, epochs=0,  **kwargs):
        """Initialize."""
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted "
                "got {}".format(warmup_method))

        self.target_lr = target_lr
        self.max_iters = max_iters
        self.power = power
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.warmup_epochs = warmup_epochs
        self.total_epochs = epochs

        super(WarmupPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate."""
        N = self.max_iters - self.warmup_iters
        T = self.last_epoch * self.max_iters / self.total_epochs - self.warmup_iters
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = float(self.last_epoch) / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError("Unknown warmup type.")
            return [self.target_lr + (base_lr - self.target_lr) * warmup_factor for base_lr in self.base_lrs]
        factor = pow(1 - T / N, self.power)
        return [self.target_lr + (base_lr - self.target_lr) * factor for base_lr in self.base_lrs]
