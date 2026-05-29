#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Basic loss module."""
import torch
import torch.nn as nn


class BalanceCrossEntropyLoss(nn.Module):
    """Balanced cross entropy loss.

    This loss measures the Binary Cross Entropy between the target and the input probabilities.


    Examples::
        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    """

    def __init__(self, negative_ratio=3.0, eps=1e-6):
        """Initialize."""
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                mask: torch.Tensor,
                return_origin=False):
        """Forward.

        Args:
            pred: Prediction of network. The shape is (N, 1, H, W).
            gt: Groudtruth. The shape is (N, 1, H, W).
            mask: The mask indicates positive regions. The shape is (N, H, W).
        """
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy_with_logits(pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = negative_loss.view(-1).topk(negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)

        if return_origin:
            return balance_loss, loss
        return balance_loss


class DiceLoss(nn.Module):
    """DiceLoss.

    Loss function from https://arxiv.org/abs/1707.03237,
    where iou computation is introduced heatmap manner to measure the
    diversity bwtween two heatmaps.
    """

    def __init__(self, eps=1e-6):
        """Initialize."""
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask, weights=None):
        """Forward.

        Args:
            pred: One or two heatmaps of shape (N, 1, H, W),
                the losses of tow heatmaps are added together.
            gt: (N, 1, H, W)
            mask: (N, H, W)
        """
        return self._compute(pred, gt, mask, weights)

    def _compute(self, pred, gt, mask, weights):
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
            gt = gt[:, 0, :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = (pred * gt * mask).sum()

        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1, f'unions {union}, intersection {intersection} \n \
            pred {pred.dtype} \n {pred[0]} \n {torch.max(pred)} \n {torch.min(pred)} \
            gt {gt.dtype} \n {gt[0]} \n {torch.max(gt)} \n {torch.min(gt)} '
        return loss


class MaskL1Loss(nn.Module):
    """Mask L1 Loss."""

    def __init__(self, eps=1e-6):
        """Initialize."""
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask):
        """Forward."""
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        return loss
