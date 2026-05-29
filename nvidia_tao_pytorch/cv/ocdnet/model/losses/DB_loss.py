#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DB loss module."""
from torch import nn

from nvidia_tao_pytorch.cv.ocdnet.model.losses.basic_loss import BalanceCrossEntropyLoss, MaskL1Loss, DiceLoss


class DBLoss(nn.Module):
    """Differentiable Binarization (DB) Loss."""

    def __init__(self, alpha=1.0, beta=10, ohem_ratio=3, reduction='mean', eps=1e-6):
        """Implement loss based on PSE(Progressive Scale Expansion) network.

        The total loss contains three parts:
            - The loss of approximate binary map. It is calculated by Dice Loss.
            - The loss of probability(shrink) map. It is calculated by Balance Cross Entropy Loss.
            - The loss of threshold map. It is calculated by MaskL1 Loss.

        Args:
            alpha: coefficient for shrink_map loss
            beta: coefficient for threshold_map loss
            ohem_ratio: the ratio of negative samples to positive samples
            reduction: calculate loss via 'mean' or 'sum'
        """
        super().__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"

        self.alpha = alpha
        self.beta = beta
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio)
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    def forward(self, pred, batch):
        """Forward."""
        shrink_logits = pred[:, 0, :, :]
        threshold_maps = pred[:, 1, :, :]
        binary_maps = pred[:, 2, :, :]

        loss_shrink_maps = self.bce_loss(shrink_logits, batch['shrink_map'], batch['shrink_mask'])
        loss_threshold_maps = self.l1_loss(threshold_maps, batch['threshold_map'], batch['threshold_mask'])
        metrics = dict(loss_shrink_maps=loss_shrink_maps, loss_threshold_maps=loss_threshold_maps)
        if pred.size()[1] > 2:
            loss_binary_maps = self.dice_loss(binary_maps, batch['shrink_map'], batch['shrink_mask'])
            metrics['loss_binary_maps'] = loss_binary_maps
            loss_all = self.alpha * loss_shrink_maps + self.beta * loss_threshold_maps + loss_binary_maps
            metrics['loss'] = loss_all
        else:
            metrics['loss'] = loss_shrink_maps
        return metrics
