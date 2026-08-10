# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed sufficient-statistic reduction helpers for OneFormer metrics."""

import numpy as np
import torch


def distributed_sum_array(values, device=None):
    """Sum a numeric array across initialized distributed workers.

    The caller must reduce additive sufficient statistics with this function
    before computing nonlinear metrics such as mIoU or PQ.
    """
    array = np.asarray(values, dtype=np.float64)
    tensor = torch.as_tensor(array, dtype=torch.float64, device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.detach().cpu().numpy()


def summarize_semantic_iou(area_intersect, area_union, area_label):
    """Compute semantic IoU and pixel accuracy from global sufficient statistics."""
    area_intersect = np.asarray(area_intersect, dtype=np.float64)
    area_union = np.asarray(area_union, dtype=np.float64)
    area_label = np.asarray(area_label, dtype=np.float64)
    if not (
        area_intersect.shape == area_union.shape == area_label.shape and
        area_intersect.ndim == 1
    ):
        raise ValueError("Semantic metric statistics must be same-length 1-D arrays.")
    for name, value in (
        ("area_intersect", area_intersect),
        ("area_union", area_union),
        ("area_label", area_label),
    ):
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"{name} must contain finite non-negative values.")

    valid_union = area_union > 0
    iou = np.divide(
        area_intersect,
        area_union,
        out=np.full_like(area_union, np.nan),
        where=valid_union,
    )
    miou = float(np.nanmean(iou)) if valid_union.any() else 0.0
    label_total = float(area_label.sum())
    accuracy = float(area_intersect.sum() / label_total) if label_total > 0 else 0.0
    return {"iou": iou, "mIoU": miou, "all_acc": accuracy}
