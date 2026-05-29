# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Depth Net Evaluator in distributed mode. """

from typing import Tuple

import torch
from torch import Tensor, tensor

from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_same_shape


def _rmse_update(preds: Tensor, target: Tensor, max_disparity: int = None) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute Mean Absolute Error.

    Check for same shape of input tensors.

    Args:
        preds (torch.Tensor): Predicted tensor
        target (torch.Tensor): Ground truth tensor

    Returns:
        sum_abs_error (torch.Tensor): Sum of root mean squared error
        num_obs (int): Number of observations
    """
    _check_same_shape(preds, target)
    if max_disparity is not None:
        mask = (target > 0.) & (target < max_disparity)
    else:
        mask = (target > 0.)

    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo
    mse = torch.nn.functional.mse_loss(preds[mask], target[mask])
    rmse = torch.sqrt(mse)
    return rmse, target.shape[0]


def _rmse_log_update(preds: Tensor, target: Tensor, max_disparity: int = None) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute Mean Absolute Error.

    Check for same shape of input tensors.

    Args:
        preds (torch.Tensor): Predicted tensor
        target (torch.Tensor): Ground truth tensor

    Returns:
        sum_log_error (torch.Tensor): Sum of log error
        num_obs (int): Number of observations
    """
    _check_same_shape(preds, target)
    if max_disparity is not None:
        mask = (target > 0.) & (target < max_disparity)
    else:
        mask = (target > 0.)

    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo
    mse_log = torch.nn.functional.mse_loss(torch.log(preds[mask]), torch.log(target[mask]))
    rmse_log = torch.sqrt(mse_log)
    return rmse_log, target.shape[0]


def _abs_rel_update(preds: Tensor, target: Tensor, max_disparity: int = None) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute Mean Absolute Error.

    Check for same shape of input tensors.

    Args:
        preds (torch.Tensor): Predicted tensor
        target (torch.Tensor): Ground truth tensor

    Returns:
        sum_abs_error (torch.Tensor): Sum of absolute relative error
        num_obs (int): Number of observations
    """
    _check_same_shape(preds, target)
    if max_disparity is not None:
        mask = (target > 0.) & (target < max_disparity)
    else:
        mask = (target > 0.)

    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo

    sum_abs_error = torch.sum(torch.abs(preds[mask] - target[mask]) / target[mask], dim=0)
    return sum_abs_error, target.shape[0]


def _sq_rel_update(preds: Tensor, target: Tensor, max_disparity: int = None) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute Mean Absolute Error.

    Check for same shape of input tensors.

    Args:
        preds (torch.Tensor): Predicted tensor
        target (torch.Tensor): Ground truth tensor

    Returns:
        sum_sq_error (torch.Tensor): Sum of squared relative error
        num_obs (int): Number of observations
    """
    _check_same_shape(preds, target)
    if max_disparity is not None:
        mask = (target > 0.) & (target < max_disparity)
    else:
        mask = (target > 0.)

    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo
    rel_sq_error = torch.sum(torch.pow(
        (preds[mask] - target[mask]), 2)) / torch.sum(torch.abs(target[mask] - torch.mean(target[mask])), dim=0, keepdims=True)
    return rel_sq_error, target.shape[0]


def _epe_error(preds: Tensor, target: Tensor, max_disparity: int = None) -> Tuple[Tensor, int]:
    """Calculates and returns EPE error and other related stereo metrics.

    This private helper function computes several key metrics for stereo
    matching, including End-Point Error (EPE), D1-metric, and bad-pixel
    ratios at different thresholds. It handles batch processing and
    converts input tensors to floating point numbers.

    Args:
        preds (torch.Tensor): The predicted disparity maps. Expected shape is `(N, C, H, W)`.
        target (torch.Tensor): The ground truth disparity maps. Expected shape
            is the same as `preds`.
        max_disparity (int, optional): The maximum possible disparity value.
            Used to create a valid mask for the ground truth. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]: A tuple containing the
            sums of the following metrics across the batch:
            - sum_d1 (torch.Tensor): The sum of the D1-metric values.
            - sum_bp1 (torch.Tensor): The sum of the bad-pixel ratios with a threshold of 1.
            - sum_bp2 (torch.Tensor): The sum of the bad-pixel ratios with a threshold of 2.
            - sum_bp3 (torch.Tensor): The sum of the bad-pixel ratios with a threshold of 3.
            - sum_epe_val (torch.Tensor): The sum of the mean EPE values.
            - num_obs (int): The number of observations (i.e., the batch size).
    """
    _check_same_shape(preds, target)
    mask = (target > 0.) & (target < max_disparity)
    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo
    epe = torch.abs(preds - target)
    B = target.shape[0]
    epe_mean = (epe[mask]).reshape(B, -1).sum(dim=-1) / (mask.reshape(B, -1).sum(dim=-1) + 1e-8)
    d1, bp1, bp2, bp3, epe_val = 0.0, 0.0, 0.0, 0.0, 0.0
    # assuming batch is not 1
    for i in range(B):
        d1 += (((epe[i][mask[i]] > 3) & (epe[i][mask[i]] / target[i][mask[i]] > 0.05)) + 0.0).mean()
        bp1 += (((epe[i] > 1)[mask[i]]) + 0.0).mean()
        bp2 += (((epe[i] > 2)[mask[i]]) + 0.0).mean()
        bp3 += (((epe[i] > 3)[mask[i]]) + 0.0).mean()
        epe_val += epe_mean
    return d1, bp1, bp2, bp3, epe_val, target.shape[0]


class StereoDepthEvaluator(Metric):
    """Depth Evaluation Metric Class."""

    def __init__(self, max_disparity=416, **kwargs):
        """Initialize for Depth Metric Class.
        Args:
            max_disparity (float): Maximum disparity value.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.max_disparity = max_disparity
        num_outputs = 1
        self.add_state("sum_abs_rel", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_sq_rel", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_rmse", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_rmse_log", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_d1", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_d2", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_d3", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_bp1", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_bp2", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_bp3", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_epe", default=torch.zeros(num_outputs), dist_reduce_fx="sum")

        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
        # track static values
        self.d1 = 0.
        self.d2 = 0.
        self.d3 = 0.
        self.bp1 = 0.
        self.bp2 = 0.
        self.bp3 = 0.
        self.epe = 0.
        self.kwargs = kwargs

    def update(self, preds: Tensor = None, target: Tensor = None):
        """Updates the metric results for a stereo estimation model.

        This function calculates various stereo metrics such as D1-metric, EPE,
        and different relative and squared errors. It accumulates these values
        into the class attributes for later aggregation.

        Args:
            preds (torch.Tensor): The predicted disparity maps. The tensor is
                expected to have a shape of `(N, C, H, W)`, where N is the batch
                size, C is the number of output channels (usually 1 for disparity),
                and H and W are the height and width of the maps. The values
                should represent the estimated disparity.
            target (torch.Tensor): The ground truth disparity maps. The tensor
                should have the same shape as `preds` and contain the true
                disparity values. It's important that this tensor is correctly
                aligned with the `preds` tensor.

        Returns:
            None: This function does not return any value. It updates the internal
                state of the object by accumulating the calculated metrics.
        """
        sum_d1, sum_bp1, sum_bp2, sum_bp3, sum_epe_val, num_obs = _epe_error(
            preds, target, max_disparity=self.max_disparity)
        sum_sq_rel, _ = _sq_rel_update(preds, target, max_disparity=self.max_disparity)
        sum_rmse, _ = _rmse_update(preds, target, max_disparity=self.max_disparity)
        sum_rmse_log, _ = _rmse_log_update(preds, target, max_disparity=self.max_disparity)
        sum_abs_rel, _ = _abs_rel_update(preds, target, max_disparity=self.max_disparity)
        self.sum_abs_rel += sum_abs_rel
        self.sum_sq_rel += sum_sq_rel
        self.sum_rmse += sum_rmse
        self.sum_rmse_log += sum_rmse_log
        self.sum_d1 += sum_d1
        self.sum_bp1 += sum_bp1
        self.sum_bp2 += sum_bp2
        self.sum_bp3 += sum_bp3
        self.sum_epe += sum_epe_val
        self.total += num_obs
        self.d1 = sum_d1
        self.bp1 = sum_bp1
        self.bp2 = sum_bp2
        self.bp3 = sum_bp3
        self.epe = sum_epe_val

    def get_single_update(self):
        """Retrieves the most recently calculated metric values as a dictionary.

        This function returns the metric values from the latest batch processed
        by the `update` method. The values are converted to standard Python
        numbers using `.item()` for easy serialization and use.

        Returns:
            dict: A dictionary containing the following scalar metric values
                from the last processed batch:
                - 'd1' (float): The D1-metric.
                - 'd2' (float): The Delta-2 metric.
                - 'd3' (float): The Delta-3 metric.
                - 'bp1' (float): The bad-pixel metric at threshold 1.
                - 'bp2' (float): The bad-pixel metric at threshold 2.
                - 'bp3' (float): The bad-pixel metric at threshold 3.
                - 'epe' (float): The End-Point Error.
                - 'abs_rel' (float): The absolute relative error.
                - 'sq_rel' (float): The squared relative error.
                - 'rmse' (float): The Root Mean Squared Error.
                - 'rmse_log' (float): The Root Mean Squared Logarithmic Error.
        """
        return {"d1": self.d1.item(), "d2": self.d2.item(), "d3": self.d3.item(),
                "bp1": self.bp1.item(), "bp2": self.bp2.item(), "bp3": self.bp3.item(),
                "epe": self.epe.item(), "abs_rel": self.abs_rel.item(), "sq_rel": self.sq_rel.item(),
                "rmse": self.rmse.item(), 'rmse_log': self.rmse_log.item()}

    def compute(self):
        """Computes and returns the final depth evaluation metrics.

        This function aggregates the accumulated metric sums (`self.sum_*`)
        and divides them by the total number of observations (`self.total`)
        to compute the final, average metrics over the entire dataset.

        Returns:
            dict: A dictionary containing the following aggregated scalar
                metric values:
                - 'd1' (float): The final D1-metric.
                - 'd2' (float): The final Delta-2 metric.
                - 'd3' (float): The final Delta-3 metric.
                - 'bp1' (float): The final bad-pixel metric at threshold 1.
                - 'bp2' (float): The final bad-pixel metric at threshold 2.
                - 'bp3' (float): The final bad-pixel metric at threshold 3.
                - 'epe' (float): The final End-Point Error.
                - 'abs_rel' (float): The final absolute relative error.
                - 'sq_rel' (float): The final squared relative error.
                - 'rmse' (float): The final Root Mean Squared Error.
                - 'rmse_log' (float): The final Root Mean Squared Logarithmic Error.
        """
        abs_rel = self.sum_abs_rel / self.total
        sq_rel = self.sum_sq_rel / self.total
        rmse = torch.sqrt(self.sum_rmse / self.total)
        rmse_log = torch.sqrt(self.sum_rmse_log / self.total)
        d1 = self.sum_d1 / self.total
        d2 = self.sum_d2 / self.total
        d3 = self.sum_d3 / self.total
        bp1 = self.sum_bp1 / self.total
        bp2 = self.sum_bp2 / self.total
        bp3 = self.sum_bp3 / self.total
        epe = self.sum_epe / self.total

        return {"d1": d1.item(), "d2": d2.item(), "d3": d3.item(),
                "bp1": bp1.item(), "bp2": bp2.item(), "bp3": bp3.item(),
                "epe": epe.item(), "abs_rel": abs_rel.item(), "sq_rel": sq_rel.item(),
                "rmse": rmse.item(), 'rmse_log': rmse_log.item()}
