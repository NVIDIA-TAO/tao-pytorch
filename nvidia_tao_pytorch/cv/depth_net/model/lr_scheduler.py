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

"""DepthNet lr scheduler module."""

from torch.optim.lr_scheduler import (MultiStepLR, StepLR, LambdaLR,
                                      CosineAnnealingLR, PolynomialLR, OneCycleLR,)


def build_lr_scheduler(optimizer, scheduler_type, train_config, trainer):
    """Build learning rate scheduler given the scheduler type and training configuration.

    This function creates and configures a PyTorch learning rate scheduler based on the
    specified scheduler type and training configuration parameters. It supports multiple
    scheduler types including MultiStepLR, StepLR, and LambdaLR.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be scheduled.
        scheduler_type (str): Type of learning rate scheduler to create. Supported types:
            - "MultiStep": Multi-step learning rate scheduler
            - "StepLR": Step learning rate scheduler
            - "LambdaLR": Lambda learning rate scheduler
            - "CosineAnnealingLR": CosineAnnealing learning rate scheduler
            - "OneCycleLR": Onecycle learning rate scheduler
            - "PolynomailLR":Polynomial learning rate scheduler
        train_config (object): Training configuration object containing scheduler parameters.
            Must have the following attributes:
            - optim.lr_steps: List of step numbers for MultiStepLR
            - optim.lr_decay: Learning rate decay factor (gamma)
            - optim.lr_step_size: Step size for StepLR
            - num_epochs: Number of training epochs (for LambdaLR)
        data_loader_length (int, optional): Length of the training data loader. Required
            for LambdaLR scheduler. Defaults to None.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Configured learning rate scheduler instance.

    Raises:
        ValueError: If data_loader_length is None when scheduler_type is "LambdaLR".
        NotImplementedError: If the specified scheduler_type is not supported.
    """
    if scheduler_type == "MultiStepLR":
        lr_scheduler = MultiStepLR(
            optimizer=optimizer,
            milestones=train_config['optim']["lr_steps"],
            gamma=train_config['optim']["lr_decay"],
        )
    elif scheduler_type == "StepLR":
        lr_scheduler = StepLR(
            optimizer=optimizer,
            step_size=train_config['optim']["lr_step_size"],
            gamma=train_config['optim']["lr_decay"],
        )
    elif scheduler_type == "LambdaLR":
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda x: 1 - x / (train_config['num_epochs'] * len(trainer.datamodule.train_dataloader())),
        )
    elif scheduler_type == "OneCycleLR":
        lr_scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=train_config["optim"]["lr"],
            total_steps=trainer.estimated_stepping_batches,
        )
    elif scheduler_type == "PolynomialLR":
        lr_scheduler = PolynomialLR(
            optimizer=optimizer,
            total_iters=trainer.estimated_stepping_batches,
            power=1,
        )
    elif scheduler_type == "CosineAnnealingLR":
        lr_scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=trainer.estimated_stepping_batches,
            eta_min=train_config["optim"]["min_lr"],
            last_epoch=-1,
        )
    else:
        raise NotImplementedError("LR Scheduler {} is not implemented".format(scheduler_type))
    return lr_scheduler
