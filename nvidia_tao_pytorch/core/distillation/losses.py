# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common losses for distilling TAO Toolkit models."""

from math import exp
from typing import Dict, Callable, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class Criterion(nn.Module):
    """Base class for all loss functions."""

    def forward(self, output, target):
        """Forward pass of the criterion.

        Args:
            output: The output tensor from the model.
            target: The target tensor.

        Returns:
            The loss value.
        """
        raise NotImplementedError


class KLDivCriterion(Criterion):
    """KLDivCriterion class to calculate KL Divergence loss."""

    def forward(self, output, target):
        """Calculates KL Divergence loss.

        Args:
            output: The output tensor from the model.
            target: The target tensor.

        Returns:
            The KL Divergence loss value.
        """
        output = F.log_softmax(output, dim=-1)
        target = F.log_softmax(target, dim=-1)
        return F.kl_div(output, target, log_target=True)


class LPCriterion(Criterion):
    """LPCriterion class to calculate Lp loss."""

    def __init__(self, p: float = 2):
        """Initializes the LPCriterion.

        Args:
            p: The p value for order in Lp loss.
        """
        super().__init__()
        self.p = p

    def forward(self, output, target):
        """Calculates Lp loss.

        Args:
            output: The output tensor from the model.
            target: The target tensor.

        Returns:
            The Lp loss value.
        """
        if self.p == 2:
            return F.mse_loss(output, target)
        if self.p == 1:
            return F.l1_loss(output, target)

        return torch.mean(torch.abs((output - target)**(self.p)))


class ProjCriterion2d(Criterion):
    """ProjCriterion2d class to project output to target shape and apply criterion."""

    def __init__(self, in_channels: int, out_channels: int, base_binding: Criterion):
        """Initializes the ProjCriterion2d.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            base_binding: The base criterion to apply.
        """
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1)
        self.base_binding = base_binding

    def forward(self, output, target):
        """Projects output to target shape and applies criterion."""
        output = self.proj(output)
        return self.base_binding(output, target)


class WeightedCriterion(Criterion):
    """WeightedCriterion class to apply weight to a criterion."""

    def __init__(self, weight: float, criterion: Criterion):
        """Initializes the WeightedCriterion.

        Args:
            weight: The weight to apply to the criterion.
            criterion: The base criterion to apply.
        """
        super().__init__()
        self.weight = weight
        self.criterion = criterion

    def forward(self, output, target):
        """Applies weight to the criterion.

        Args:
            output: The output tensor from the model.
            target: The target tensor.

        Returns:
            The weighted loss value.
        """
        return self.criterion(output, target) * self.weight


class L1ProjCriterion2d(ProjCriterion2d):
    """L1ProjCriterion2d class to project output to target shape and apply L1 criterion."""

    def __init__(self, in_channels: int, out_channels: int):
        """Initializes the L1ProjCriterion2d.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
        """
        super().__init__(in_channels, out_channels, LPCriterion(p=1))


class L2ProjCriterion2d(ProjCriterion2d):
    """L2ProjCriterion2d class to project output to target shape and apply L2 criterion."""

    def __init__(self, in_channels: int, out_channels: int):
        """Initializes the L2ProjCriterion2d.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
        """
        super().__init__(in_channels, out_channels, LPCriterion(p=2))


class DictCriterion(Criterion):
    """Criterion class to apply criterion on a dictionary of tensor_name:tensor."""

    def __init__(self, children: Dict[str, Criterion]):
        """Initializes the DictCriterion.

        Args:
            children: The dictionary of tensor_name:Criterion.
        """
        super().__init__()
        self.children_modules = nn.ModuleDict(children)

    def forward(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
        """Applies criterion on a dictionary of tensor_name:tensor.

        Args:
            output: The output dictionary from the model.
            target: The target dictionary.

        Returns:
            The loss value.
        """
        loss = 0.
        # print(output.keys(), target.keys())
        for key in self.children_modules:
            loss = loss + self.children_modules[key](output[key], target[key])
        return loss


class FeatureDictCriterion(Criterion):
    """Criterion class to apply criterion on a dictionary of feature_name:tensor."""

    def __init__(self, criterion: Criterion):
        """Initializes the FeatureDictCriterion.

        Args:
            criterion: The base criterion to apply.
        """
        super().__init__()
        self.criterion = criterion

    def forward(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
        """Applies criterion on a dictionary of feature_name:tensor.

        Args:
            output: The output dictionary from the model.
            target: The target dictionary.

        Returns:
            The loss value.
        """
        loss = 0.
        for out_key, target_key in zip(output.keys(), target.keys()):
            loss = loss + self.criterion(output[out_key], target[target_key])
        # Use only the last layer
        # loss = loss + self.criterion(output[list(output.keys())[-1]], target[list(target.keys())[-1]])
        return loss


class FeatureMapCriterion(Criterion):
    """Criterion class to apply criterion on a list of feature maps."""

    def __init__(self, criterion: Criterion):
        """Initializes the FeatureMapCriterion.

        Args:
            criterion: The base criterion to apply.
        """
        super().__init__()
        self.criterion = criterion

    def forward(self, output: List[torch.Tensor], target: List[torch.Tensor]):
        """Applies criterion on a dictionary of feature_name:tensor.

        Args:
            output: The output dictionary from the model.
            target: The target dictionary.

        Returns:
            The loss value.
        """
        loss = 0.
        for out, tgt in zip(output, target):
            loss = loss + self.criterion(out, tgt)
        # Use only the last layer
        # loss = loss + self.criterion(output[list(output.keys())[-1]], target[list(target.keys())[-1]])
        return loss


def gaussian(window_size, sigma):
    """Gaussian function to calculate gaussian weights.

    Args:
        window_size: The size of the window.
        sigma: The sigma value for gaussian.

    Returns:
        The gaussian weights.
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """Create window for SSIM.

    Args:
        window_size: The size of the window.
        channel: The number of channels.

    Returns:
        The window for SSIM.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """Calculate SSIM.

    Args:
        img1: The first image tensor.
        img2: The second image tensor.
        window: The window for SSIM.
        window_size: The size of the window.
        channel: The number of channels.
        size_average: The flag to take average of SSIM.

    Returns:
        The SSIM value.
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map


class SSIM(nn.Module):
    """SSIM class to calculate SSIM loss."""

    def __init__(self, window_size=11, size_average=True):
        """Initializes the SSIM.

        Args:
            window_size: The size of the window.
            size_average: The flag to take average of SSIM.
        """
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        """Calculate SSIM.

        Args:
            img1: The first image tensor.
            img2: The second image tensor.

        Returns:
            The SSIM value.
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    """Function to calculate SSIM.

    Args:
        img1: The first image tensor.
        img2: The second image tensor.
        window_size: The size of the window.
        size_average: The flag to take average of SSIM.

    Returns:
        The SSIM value.
    """
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def weighted_loss(inputs: Dict[str, torch.Tensor],
                  target: Dict[str, torch.Tensor],
                  criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                  weights: Optional[Dict[str, torch.Tensor]] = None,
                  masks: Optional[Dict[str, torch.Tensor]] = None
                  ):
    """Function to calculate weighted loss.

    Args:
        inputs: The input dictionary from the model.
        target: The target dictionary.
        criterion: The criterion to apply.
        weights: The dictionary of weights for each key.
        masks: The dictionary of masks for each key.

    Returns:
        The weighted loss value.
    """
    if weights is None:
        weights = {}
    if masks is None:
        masks = {}

    # Check if input and target have same keys
    assert set(inputs.keys()) == set(target.keys()), "Input and target keys do not match"
    # Check if input has atleast one key
    assert len(inputs.keys()) > 0, "Input should have atleast one key"

    reference = next(iter(inputs.values()))

    loss = torch.tensor(
        0.,
        dtype=reference.dtype,
        device=reference.device
    )

    for key in inputs.keys():
        # Compute criterion for each key
        loss_for_key = criterion(inputs[key], target[key])

        # If mask provided for the key, apply mask to the loss
        if key in masks:
            loss_for_key = masks[key] * loss_for_key

        # If weight provided for the key, apply weight to the loss
        if key in weights:
            loss_for_key = weights[key] * loss_for_key

        # Take mean of the loss
        loss_for_key = torch.mean(loss_for_key)

        # Add loss for the key to the total loss
        loss += loss_for_key

    return loss


def weighted_l1_loss(inputs: Dict[str, torch.Tensor],
                     target: Dict[str, torch.Tensor],
                     weights: Optional[Dict[str, torch.Tensor]] = None,
                     masks: Optional[Dict[str, torch.Tensor]] = None
                     ):
    """Function to calculate weighted l1 loss.

    Args:
        inputs: The input dictionary from the model.
        target: The target dictionary.
        weights: The dictionary of weights for each key.
        masks: The dictionary of masks for each key.

    Returns:
        The weighted loss value.
    """
    return weighted_loss(inputs, target, lambda x, y: F.l1_loss(x, y, reduction='none'), weights, masks)


def weighted_l2_loss(inputs: Dict[str, torch.Tensor],
                     target: Dict[str, torch.Tensor],
                     weights: Optional[Dict[str, torch.Tensor]] = None,
                     masks: Optional[Dict[str, torch.Tensor]] = None
                     ):
    """Function to calculate weighted l2 loss.

    Args:
        inputs: The input dictionary from the model.
        target: The target dictionary.
        weights: The dictionary of weights for each key.
        masks: The dictionary of masks for each key.

    Returns:
        The weighted l2 loss value.
    """
    return weighted_loss(inputs, target, lambda x, y: F.mse_loss(x, y, reduction='none'), weights, masks)


def weighted_mse_loss(inputs: Dict[str, torch.Tensor],
                      target: Dict[str, torch.Tensor],
                      weights: Optional[Dict[str, torch.Tensor]] = None,
                      masks: Optional[Dict[str, torch.Tensor]] = None
                      ):
    """Function to calculate weighted mse loss.

    Args:
        inputs: The input dictionary from the model.
        target: The target dictionary.
        weights: The dictionary of weights for each key.
        masks: The dictionary of masks for each key.

    Returns:
        The weighted mse loss value.
    """
    return weighted_l2_loss(inputs, target, weights, masks)


def weighted_huber_loss(inputs: Dict[str, torch.Tensor],
                        target: Dict[str, torch.Tensor],
                        delta: float = 1.0,
                        weights: Optional[Dict[str, torch.Tensor]] = None,
                        masks: Optional[Dict[str, torch.Tensor]] = None
                        ):
    """Function to calculate weighted huber loss.

    Args:
        inputs: The input dictionary from the model.
        target: The target dictionary.
        delta: The delta value for huber loss.
        weights: The dictionary of weights for each key.
        masks: The dictionary of masks for each key.

    Returns:
        The weighted huber loss value.
    """
    return weighted_loss(inputs, target, lambda x, y: F.huber_loss(x, y, delta=delta, reduction='none'), weights, masks)
