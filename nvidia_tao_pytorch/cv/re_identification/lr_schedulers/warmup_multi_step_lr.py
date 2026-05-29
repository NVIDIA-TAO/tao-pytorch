# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warmup Multi-Step LR Scheduler Module for Re-Identification."""
from bisect import bisect_right
import torch


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """Custom learning rate scheduler with initial warm-up phase.

    This scheduler adjusts the learning rate according to the schedule defined by the `milestones`.
    It also supports a warm-up phase at the start of training, where the learning rate is initially smaller
    and gradually ramps up to its initial value.

    Inherits from PyTorch's torch.optim.lr_scheduler._LRScheduler class.

    Attributes:
        milestones (list): List of epoch indices. The learning rate is decreased at these epochs.
        gamma (float): Multiplicative factor of learning rate decay.
        warmup_factor (float): Multiplicative factor of learning rate applied during the warm-up phase.
        warmup_iters (int): Number of epochs for the warm-up phase.
        warmup_method (str): The method for the warm-up phase, either 'constant' or 'linear'.
    """

    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        """Initialize the learning rate scheduler.

        Args:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            milestones (list of int): List of epoch indices. Must be increasing.
            gamma (float, optional): Factor by which the learning rate is reduced. Defaults to 0.1.
            warmup_factor (float, optional): Factor for computing the starting warmup learning rate. Defaults to 1/3.
            warmup_iters (int, optional): Number of warmup epochs at the start of training. Defaults to 500.
            warmup_method (str, optional): Warmup method to use, either 'constant' or 'linear'. Defaults to 'linear'.
            last_epoch (int, optional): The index of the last epoch. Defaults to -1.

        Raises:
            ValueError: If `milestones` are not in increasing order, or `warmup_method` is not 'constant' or 'linear'.
        """
        if not list(milestones) == sorted(milestones):
            raise ValueError("Milestones should be a list of" " increasing integers. Got {}".format(milestones))

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute the learning rate at the current epoch.

        Returns:
            list of float: Learning rates for each parameter group.
        """
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
