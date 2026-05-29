# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warmup and Cosine Scheduler"""

import math
import torch


class LambdaWarmUpCosineScheduler:
    """
    note: use with a base_lr of 1.0
    """

    def __init__(
        self,
        *,
        val_base,
        val_final,
        max_decay_steps,
        val_start=0,
        warm_up_steps=0,
        freeze_steps=0,
    ):
        """Warmup cosine scheduler

        Args:
            val_base (float): the val after warmup
            val_final (float): the val at the end of the schedule
            max_decay_steps (int): number of steps to decay from val_base to val_final (after warmup)
            val_start (float, optional): learning rate at the start of the schedule. Defaults to 0.
            warm_up_steps (int, optional): number of steps for the warmup phase. Defaults to 0.
            freeze_steps (int, optional): number of steps to freeze the learning rate. Defaults to 0.
        """
        self.val_final = val_final
        self.val_base = val_base
        self.warm_up_steps = warm_up_steps
        self.freeze_steps = freeze_steps
        self.val_start = val_start
        self.val_base_decay_steps = max_decay_steps
        self.last_lr = 0.0

    def schedule(self, n: int):
        """Schedule function

        Args:
            n (int): step

        Returns:
            _type_: the scheduled value based on the step
        """
        if n < self.freeze_steps:
            return 0.0

        if n < self.warm_up_steps:
            lr = (
                self.val_base - self.val_start
            ) / self.warm_up_steps * n + self.val_start
            self.last_lr = lr

            return lr

        if self.val_base_decay_steps == self.warm_up_steps:
            # no cosine decay, warmup only
            return self.val_final

        t = (n - self.warm_up_steps) / (self.val_base_decay_steps - self.warm_up_steps)
        t = min(t, 1.0)
        lr = self.val_final + 0.5 * (self.val_base - self.val_final) * (
            1 + math.cos(t * torch.pi)
        )
        self.last_lr = lr

        return lr

    def __call__(self, n):
        """Return the scheduled value based on the step"""
        return self.schedule(n)
