# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DepthNet Loss module."""

import torch
from torch import nn


def compute_scale_and_shift(prediction, target, mask):
    """
    This function computes the optimal scale and shift parameters that minimize the
    mean squared error between the scaled and shifted prediction and the target,
    considering only the pixels where the mask is valid. This is used in scale-
    and shift-invariant loss functions for depth prediction.

    Args:
        prediction (torch.Tensor): Predicted depth values of shape (B, H, W).
        target (torch.Tensor): Ground truth depth values of shape (B, H, W).
        mask (torch.Tensor): Binary mask indicating valid pixels of shape (B, H, W).

    Returns:
        tuple: A tuple containing:
            - scale (torch.Tensor): Optimal scale factor for each batch item (B,).
            - shift (torch.Tensor): Optimal shift value for each batch item (B,).

    Note:
        The function solves the linear system Ax = b where:
        A = [[sum(pred²), sum(pred)], [sum(pred), sum(1)]]
        b = [sum(pred * target), sum(target)]
        x = [scale, shift]
    """
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    """
    This function computes the global average of all valid pixels across all images
    in the batch. It handles the case where there might be no valid pixels in the batch.

    Args:
        image_loss (torch.Tensor): Per-image loss values of shape (B,).
        M (torch.Tensor): Number of valid pixels per image of shape (B,).

    Returns:
        torch.Tensor: Scalar tensor representing the average loss across all valid pixels.

    Note:
        If there are no valid pixels in the entire batch (sum(M) = 0), returns 0.
    """
    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    """
    This function first computes the average loss per image (considering only valid pixels),
    then takes the mean across all images. This gives equal weight to each image regardless
    of the number of valid pixels it contains.

    Args:
        image_loss (torch.Tensor): Per-image loss values of shape (B,).
        M (torch.Tensor): Number of valid pixels per image of shape (B,).

    Returns:
        torch.Tensor: Scalar tensor representing the mean of per-image averages.

    Note:
        Images with no valid pixels (M[i] = 0) will have their loss set to 0.
    """
    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):
    """
    This function computes the MSE loss between predicted and target depth values,
    considering only the pixels where the mask is valid. The loss can be reduced
    using different strategies (batch-based or image-based).

    Args:
        prediction (torch.Tensor): Predicted depth values of shape (B, H, W).
        target (torch.Tensor): Ground truth depth values of shape (B, H, W).
        mask (torch.Tensor): Binary mask indicating valid pixels of shape (B, H, W).
        reduction (callable, optional): Reduction function to apply. Defaults to
            reduction_batch_based.

    Returns:
        torch.Tensor: Scalar tensor representing the computed MSE loss.
    """
    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):
    """
    This function computes the gradient loss by comparing the spatial gradients
    of the predicted and target depth maps. It encourages the model to preserve
    edge information and smoothness in the depth predictions.

    Args:
        prediction (torch.Tensor): Predicted depth values of shape (B, H, W).
        target (torch.Tensor): Ground truth depth values of shape (B, H, W).
        mask (torch.Tensor): Binary mask indicating valid pixels of shape (B, H, W).
        reduction (callable, optional): Reduction function to apply. Defaults to
            reduction_batch_based.

    Returns:
        torch.Tensor: Scalar tensor representing the computed gradient loss.

    Note:
        The gradient loss is computed as the sum of absolute differences in
        horizontal and vertical gradients between prediction and target.
    """
    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    """
    This module implements the MSE(Mean Squared Error) loss for depth prediction tasks. It computes
    the squared difference between predicted and target depth values, considering
    only valid pixels as indicated by the mask.

    Attributes:
        __reduction (callable): The reduction function to apply to the computed loss.
    """

    def __init__(self, reduction='batch-based'):
        """Initialize MSELoss.

        Args:
            reduction (str, optional): Reduction strategy to use. Options:
                - 'batch-based': Average across all valid pixels in the batch
                - 'image-based': Average per image, then mean across images
                Defaults to 'batch-based'.
        """
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        """Forward pass for MSELoss.

        Args:
            prediction (torch.Tensor): Predicted depth values of shape (B, H, W).
            target (torch.Tensor): Ground truth depth values of shape (B, H, W).
            mask (torch.Tensor): Binary mask indicating valid pixels of shape (B, H, W).

        Returns:
            torch.Tensor: Scalar tensor representing the computed MSE loss.
        """
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    """Gradient-based regularization loss for depth prediction.

    This module implements a multi-scale gradient loss that encourages the model
    to preserve edge information and smoothness in depth predictions across
    different spatial scales.

    Attributes:
        __reduction (callable): The reduction function to apply to the computed loss.
        __scales (int): Number of scales to compute gradient loss at.
    """

    def __init__(self, scales=4, reduction='batch-based'):
        """Initialize GradientLoss.

        Args:
            scales (int, optional): Number of scales to compute gradient loss at.
                The loss is computed at scales 2^0, 2^1, ..., 2^(scales-1).
                Defaults to 4.
            reduction (str, optional): Reduction strategy to use. Options:
                - 'batch-based': Average across all valid pixels in the batch
                - 'image-based': Average per image, then mean across images
                Defaults to 'batch-based'.
        """
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        """Forward pass for GradientLoss.

        Args:
            prediction (torch.Tensor): Predicted depth values of shape (B, H, W).
            target (torch.Tensor): Ground truth depth values of shape (B, H, W).
            mask (torch.Tensor): Binary mask indicating valid pixels of shape (B, H, W).

        Returns:
            torch.Tensor: Scalar tensor representing the computed multi-scale gradient loss.
        """
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    """
    This module implements a scale and shift invariant loss that combines MSE loss
    with gradient regularization. The loss is invariant to global scale and shift
    transformations, making it suitable for relative depth prediction tasks.

    Attributes:
        __data_loss (MSELoss): MSE loss component.
        __regularization_loss (GradientLoss): Gradient regularization component.
        __alpha (float): Weight for the regularization loss.
        __prediction_ssi (torch.Tensor): Last computed scale-and-shift-invariant prediction.
    """

    def __init__(self, alpha=2.0, scales=4, reduction='batch-based'):
        """Initialize ScaleAndShiftInvariantLoss.

        Args:
            alpha (float, optional): Weight for the gradient regularization loss.
                Defaults to 2.0.
            scales (int, optional): Number of scales for gradient loss computation.
                Defaults to 4.
            reduction (str, optional): Reduction strategy to use. Options:
                - 'batch-based': Average across all valid pixels in the batch
                - 'image-based': Average per image, then mean across images
                Defaults to 'batch-based'.
        """
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        """Forward pass for ScaleAndShiftInvariantLoss.

        Args:
            prediction (torch.Tensor): Predicted depth values of shape (B, H, W).
            target (torch.Tensor): Ground truth depth values of shape (B, H, W).
            mask (torch.Tensor): Binary mask indicating valid pixels of shape (B, H, W).

        Returns:
            torch.Tensor: Scalar tensor representing the computed scale-and-shift-invariant loss.

        Note:
            The function first computes optimal scale and shift parameters, then applies
            them to the prediction before computing the combined loss.
        """
        if not torch.isfinite(target[mask.bool()]).all():
            print('target is infinite for image: prediction: ', prediction, 'target: ', target, 'mask: ', mask)

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if total.isnan():
            print('data loss is nan for image: scale', scale, 'shift', shift, 'self.__prediction_ssi: ', self.__prediction_ssi, 'prediction: ', prediction, 'target: ', target, 'mask: ', mask)
        total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)
        if total.isnan():
            print('total loss is nan for image: prediction: ', self.__prediction_ssi, 'target: ', target, 'mask: ', mask)
        return total

    def __get_prediction_ssi(self):
        """Get the last computed scale-and-shift-invariant prediction.

        Returns:
            torch.Tensor: The last computed scale-and-shift-invariant prediction,
                or None if forward has not been called yet.
        """
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


class SiLogLoss(nn.Module):
    """Scale-invariant logarithmic loss for depth prediction.

    This module implements a scale-invariant logarithmic loss that is invariant
    to global scale transformations. It computes the variance of the logarithmic
    difference between predicted and target depths.

    Attributes:
        lambd (float): Weight for the mean term in the loss computation.
    """

    def __init__(self, lambd=0.5):
        """Initialize SiLogLoss.

        Args:
            lambd (float, optional): Weight for the mean term in the loss computation.
                The loss is computed as var(log(pred/target)) + λ * mean(log(pred/target))².
                Defaults to 0.5.
        """
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        """Forward pass for SiLogLoss.

        Args:
            pred (torch.Tensor): Predicted depth values of shape (B, H, W).
            target (torch.Tensor): Ground truth depth values of shape (B, H, W).
            valid_mask (torch.Tensor): Boolean mask indicating valid pixels of shape (B, H, W).

        Returns:
            torch.Tensor: Scalar tensor representing the computed scale-invariant logarithmic loss.

        Note:
            The loss is computed as: var(log(pred/target)) + λ * mean(log(pred/target))²
            where only valid pixels are considered.
        """
        # pred, target: (B, H, W)
        # valid_mask: (B, H, W)
        pred = pred[valid_mask]
        target = target[valid_mask]
        g = torch.log(pred) - torch.log(target)
        Dg = torch.var(g) + self.lambd * torch.pow(torch.mean(g), 2)
        return Dg
