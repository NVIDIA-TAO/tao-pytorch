# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Module to compute losses for stereo depth network"""

import torch
import torch.nn.functional as F
from torch import nn


def reduction_batch_based(image_loss, M):
    """Reduces the loss based on the batch size.
    """
    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


class SequenceLoss(nn.Module):
    """Loss function defined over sequence of disparity predictions."""

    def __init__(self, loss_gamma=0.9, max_disparity=192):
        """Initializes the SequenceLoss.

        Args:
            loss_gamma: Loss gamma for weighting predictions.
            max_disparity: Maximum disparity value.
            is_interpolated: Whether disparity ground truth is interpolated.
        """
        super().__init__()
        self.loss_gamma = loss_gamma
        self.max_disparity = max_disparity

    def forward(self, disp_preds, disp_init_pred, disp_gt, valid, is_interpolated=True):
        """Forward pass of SequenceLoss.

        Args:
            disp_preds: List of predicted disparities.
            disp_init_pred: Initial predicted disparity.
            disp_gt: Ground truth disparity.
            valid: Valid mask.

        Returns:
            Tuple of disparity loss and metrics.
        """
        n_predictions = len(disp_preds)
        assert n_predictions >= 1, "there should be at least one model prediction (iter=1)!"
        disp_loss = 0.0
        mag = torch.sum(disp_gt**2, dim=1, keepdim=True).sqrt()
        valid = (valid.bool() & (mag < self.max_disparity))
        if len(valid.shape) == 3:
            valid = valid.unsqueeze(1)
        assert valid.shape == disp_gt.shape, \
            f"Validity mask and disparity GT shape should be the same {valid.shape} vs {disp_gt.shape}"

        if is_interpolated:
            disp_gt_04 = F.interpolate(disp_gt, size=None,
                                       scale_factor=1 / 4.0, mode='nearest') / 4.0
            valid_04 = F.interpolate(valid.float(), size=None,
                                     scale_factor=1 / 4.0, mode='nearest').round().bool()
        else:
            valid_04 = valid
            disp_gt_04 = disp_gt

        disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[valid_04.bool()],
                                            disp_gt_04[valid_04.bool()], reduction='mean')
        init_disp_loss = disp_loss.item()
        if n_predictions > 1:
            for i in range(n_predictions):
                adjusted_loss_gamma = self.loss_gamma**(15 / (n_predictions - 1))
                i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
                i_loss = (disp_preds[i] - disp_gt).abs()
                assert i_loss.shape == valid.shape, \
                    f"[loss map and Validity mask shapes must be the same: {i_loss.shape} vs {valid.shape}]"
                disp_loss += i_weight * i_loss[valid.bool()].mean()
        else:
            disp_loss = init_disp_loss

        epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
        epe = epe.reshape(-1)[valid.reshape(-1)]

        metrics = {
            'init_disp_loss': init_disp_loss,
            'epe': epe.mean().item(),
            'bp1': (epe < 1).float().mean().item(),
            'bp2': (epe > 2).float().mean().item(),
            'bp3': (epe < 3).float().mean().item(),
        }
        return disp_loss, metrics


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):
    """Helper function to apply mse_loss reduction to the mse_loss
    """
    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


class MSELoss(nn.Module):
    """Computes the masked Mean Squared Error loss."""

    def __init__(self):
        """Initializes the MSELoss module.

        Args:
            reduction: The reduction method to use ('batch-based' or
                'image-based'). Defaults to 'batch-based'.
        """
        super().__init__()

        self.__reduction = reduction_batch_based

    def forward(self, prediction, target, mask):
        """Computes the masked MSE loss.

        Args:
            prediction: The predicted values.
            target: The ground truth target values.
            mask: A binary mask indicating valid regions.

        Returns:
            The computed masked MSE loss.
        """
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


LOSS = {
    'sequence_loss': SequenceLoss,
    'mse_loss': MSELoss
}
