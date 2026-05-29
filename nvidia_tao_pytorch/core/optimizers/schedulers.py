# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Custom schedulers for TAO training workflows. """

import torch


# TODO: @scha to add the schedulder to RTDETR PL module for the next release
class LinearWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Linear Warmup scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int = 1000,
    ):
        if num_warmup_steps > 0:
            msg = f"num_warmup_steps should be > 0, got {num_warmup_steps}"
            ValueError(msg)
        self.num_warmup_steps = num_warmup_steps
        super().__init__(optimizer, lambda step: min(step / num_warmup_steps, 1.0))
