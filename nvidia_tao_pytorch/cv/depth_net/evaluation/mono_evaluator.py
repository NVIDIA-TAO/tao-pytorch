# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Depth Net Evaluator in distributed mode. """

from typing import Tuple, List, Dict

import torch
from torch import Tensor, tensor

from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_same_shape


def align_depth_least_square(
    gt: Tensor,
    pred: Tensor,
):
    """Align depth using least square method.

    This function performs least squares alignment between predicted and ground truth depth/disparity values.
    It finds optimal scale and shift parameters to minimize the squared error between predictions and ground truth.

    Args:
        gt (torch.Tensor): Ground truth disparity/depth tensor. Shape: (N,) where N is the number of valid pixels.
                           Typically flattened from (H, W) using valid_mask, resulting in (H*W,) or subset thereof.
                           Will be squeezed internally to ensure 1D tensor.
        pred (torch.Tensor): Predicted disparity/depth tensor. Shape: (N,) where N is the number of valid pixels.
                             Must match gt shape. Typically flattened from (H, W) using valid_mask.
                             Will be squeezed internally to ensure 1D tensor.

    Returns:
        aligned_pred (torch.Tensor): Aligned disparity/depth tensor with same shape as input pred tensor.
                                    Values are transformed as: pred * scale + shift
    """
    ori_shape = pred.shape  # input shape
    gt = gt.squeeze()  # [H, W]
    pred = pred.squeeze()
    assert (
        gt.shape == pred.shape
    ), f"GT shape: {gt.shape}, Pred shape: {pred.shape} are not matched."

    gt_masked = gt.reshape((-1, 1))
    pred_masked = pred.reshape((-1, 1))

    # numpy solver
    _ones = torch.ones_like(pred_masked)
    A = torch.cat([pred_masked, _ones], dim=-1)
    X = torch.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)
    return aligned_pred


def _delta_log_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor, Tensor, int]:
    """Update and returns variables required to compute delta accuracy metrics.

    Computes delta accuracy metrics (δ1, δ2, δ3) which measure the percentage of pixels
    where the ratio between predicted and target values is within certain thresholds.

    Check for same shape of input tensors.

    Args:
        preds (torch.Tensor): Predicted depth/disparity tensor. Shape: (N*H*W,) where N is batch size, H*W is flattened spatial dimensions.
                              Created by concatenating masked predictions from multiple images: each image's valid pixels
                              are flattened using valid_mask, then concatenated along dim=0.
                              Will be converted to float if not already.
        target (torch.Tensor): Ground truth depth/disparity tensor. Shape: (N*H*W,) where N is batch size, H*W is flattened spatial dimensions.
                               Must match preds shape. Created by concatenating masked ground truth from multiple images.
                               Will be converted to float if not already.

    Returns:
        d1 (torch.Tensor): Delta 1 - count of pixels where max(pred/target, target/pred) < 1.25.
                           Shape: () - scalar tensor.
        d2 (torch.Tensor): Delta 2 - count of pixels where max(pred/target, target/pred) < 1.25².
                           Shape: () - scalar tensor.
        d3 (torch.Tensor): Delta 3 - count of pixels where max(pred/target, target/pred) < 1.25³.
                           Shape: () - scalar tensor.
        num_obs (int): Number of observations (N*H*W - total number of valid pixels across all images).
    """
    _check_same_shape(preds, target)

    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo

    thresh = torch.max((target / preds), (preds / target))

    d1 = torch.sum(thresh < 1.25, dim=0)
    d2 = torch.sum(thresh < 1.25 ** 2, dim=0)
    d3 = torch.sum(thresh < 1.25 ** 3, dim=0)

    return d1, d2, d3, target.shape[0]


def _abs_rel_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute absolute relative error.

    Computes the sum of absolute relative error between predicted and target values.
    The absolute relative error is defined as |pred - target| / target.

    Check for same shape of input tensors.

    Args:
        preds (torch.Tensor): Predicted depth/disparity tensor. Shape: (N*H*W,) where N is batch size, H*W is flattened spatial dimensions.
                              Created by concatenating masked predictions from multiple images: each image's valid pixels
                              are flattened using valid_mask, then concatenated along dim=0.
                              Will be converted to float if not already.
        target (torch.Tensor): Ground truth depth/disparity tensor. Shape: (N*H*W,) where N is batch size, H*W is flattened spatial dimensions.
                               Must match preds shape. Created by concatenating masked ground truth from multiple images.
                               Will be converted to float if not already.

    Returns:
        sum_abs_error (torch.Tensor): Sum of absolute relative error |pred - target| / target.
                                      Shape: () - scalar tensor.
        num_obs (int): Number of observations (N*H*W - total number of valid pixels across all images).
    """
    _check_same_shape(preds, target)
    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo

    sum_abs_error = torch.sum(torch.abs(preds - target) / target, dim=0)
    return sum_abs_error, target.shape[0]


class MonoDepthEvaluator(Metric):
    """Depth Evaluation Metric Class."""

    def __init__(self, align_gt: bool = True, min_depth: float = 0.001, max_depth: float = 10, **kwargs):
        """Initialize for Depth Metric Class.
        Args:
            align_gt (bool): Whether to align the ground truth disparity/depth tensor.
            min_depth (float): Minimum depth value.
            max_depth (float): Maximum depth value.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

        self.align_gt = align_gt
        self.min_depth = min_depth
        self.max_depth = max_depth
        num_outputs = 1
        self.add_state("sum_abs_rel", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_sq_rel", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_rmse", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_rmse_log", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_d1", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_d2", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_d3", default=torch.zeros(num_outputs), dist_reduce_fx="sum")

        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, post_processed_results: List[Dict]) -> None:
        """Function to accumulate the metric results to reduce in the final step.
        Args:
            post_processed_results (List[Dict]): List of post-processed results.
        """
        pred_list = []
        target_list = []
        for result in post_processed_results:
            pred_i = result['depth_pred']
            gt_i = result['disp_gt']
            valid_mask_i = result['valid_mask']
            if self.align_gt:
                pred_aligned = align_depth_least_square(
                    gt=gt_i[valid_mask_i],
                    pred=pred_i[valid_mask_i],
                )
                pred_aligned = torch.clip(pred_aligned, min=1e-8, max=None)  # avoid 0 disparity
                target = torch.clip(gt_i[valid_mask_i], min=1e-8, max=None)  # avoid 0 disparity
            else:
                pred_aligned = torch.clip(pred_i[valid_mask_i], min=self.min_depth, max=self.max_depth)  # avoid 0 disparity
                target = torch.clip(gt_i[valid_mask_i], min=self.min_depth, max=self.max_depth)
            pred_list.append(pred_aligned)
            target_list.append(target)

        pred_aligned = torch.concat(pred_list, dim=0)
        target = torch.concat(target_list, dim=0)

        sum_abs_rel, num_obs = _abs_rel_update(pred_aligned, target)
        sum_d1, sum_d2, sum_d3, _ = _delta_log_update(pred_aligned, target)

        self.sum_abs_rel += sum_abs_rel
        self.sum_d1 += sum_d1
        self.sum_d2 += sum_d2
        self.sum_d3 += sum_d3

        self.total += num_obs

    def compute(self):
        """Reduce and compute the final depth evaluation metric.
        Returns:
            dict: Dictionary containing the depth evaluation metrics.
        """
        abs_rel = self.sum_abs_rel / self.total
        d1 = self.sum_d1 / self.total
        d2 = self.sum_d2 / self.total
        d3 = self.sum_d3 / self.total

        return {"d1": d1.item(), "d2": d2.item(), "d3": d3.item(), "abs_rel": abs_rel.item()}
