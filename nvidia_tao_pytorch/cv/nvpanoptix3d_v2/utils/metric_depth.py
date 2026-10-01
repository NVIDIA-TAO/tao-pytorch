# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metric-depth supervision for the shared VGGT metric scale head."""

from typing import Dict

import torch
import torch.nn as nn


class MetricDepthLoss(nn.Module):
    """SILog (+ optional AbsRel) supervision for the metric scale head.

    The scale-invariant log (SILog) loss is the standard objective for
    monocular metric depth regression (NYU / KITTI):

        L_SILog =  (1/N) Σ (log d_pred − log d_gt)^2
                 − (λ / N^2) (Σ (log d_pred − log d_gt))^2

    with ``λ = 0.85``.  An optional AbsRel term

        L_AbsRel = mean( |d_pred − d_gt| / d_gt )

    can be added for direct metric accuracy supervision.

    Args:
        silog_weight:  Coefficient on the SILog term.
        absrel_weight: Coefficient on the auxiliary AbsRel term (set to 0
            to disable).
        silog_lambda:  ``λ`` mixing constant (0..1) inside SILog.
        min_depth:     Lower bound for valid GT depth (metres).
        max_depth:     Upper bound for valid GT depth (metres).
        min_pred:      Lower clamp on the prediction before taking ``log``.
        eps:           Numerical safety added inside ``log``.
    """

    def __init__(
        self,
        silog_weight: float = 1.0,
        absrel_weight: float = 0.1,
        silog_lambda: float = 0.85,
        min_depth: float = 0.1,
        max_depth: float = 10.0,
        min_pred: float = 1e-3,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.silog_weight = silog_weight
        self.absrel_weight = absrel_weight
        self.silog_lambda = silog_lambda
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_pred = min_pred
        self.eps = eps

    def forward(
        self,
        metric_depth: torch.Tensor,
        gt_depth: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            metric_depth:       ``[B, S,H,W,1]`` scaled metric depth
                ``s · d_rel``.
            gt_depth:           ``[B, S, H, W]`` GT metric depth in metres.
        Returns:
            Dict with ``loss_silog``, ``loss_absrel`` and the aggregated
            ``loss_metric_total``.  Per-batch elements with too few valid
            pixels contribute zero (and don't contaminate the average).
        """
        if metric_depth.dim() == 5:
            pred = metric_depth.squeeze(-1)  # [B, S, H, W]
        else:
            pred = metric_depth

        gt = gt_depth

        valid = (
            torch.isfinite(gt) &
            (gt > self.min_depth) &
            (gt < self.max_depth) &
            torch.isfinite(pred) &
            (pred > self.min_pred)
        )

        device = pred.device
        zero = torch.zeros((), device=device, dtype=pred.dtype)

        B = pred.shape[0]
        silog_terms = []
        absrel_terms = []

        for b in range(B):
            m = valid[b]
            n = int(m.sum().item())
            if n < 10:
                silog_terms.append(zero)
                absrel_terms.append(zero)
                continue

            p = pred[b][m].clamp(min=self.min_pred)
            g = gt[b][m].clamp(min=self.eps)

            log_diff = torch.log(p + self.eps) - torch.log(g + self.eps)
            silog = log_diff.pow(2).mean() - self.silog_lambda * log_diff.mean().pow(2)
            silog_terms.append(silog)

            if self.absrel_weight > 0:
                absrel_terms.append(((p - g).abs() / g).mean())
            else:
                absrel_terms.append(zero)

        loss_silog = torch.stack(silog_terms).mean()
        loss_absrel = torch.stack(absrel_terms).mean()

        loss_total = (
            self.silog_weight * loss_silog +
            self.absrel_weight * loss_absrel
        )

        return {
            "loss_silog": loss_silog,
            "loss_absrel": loss_absrel,
            "loss_metric_total": loss_total,
        }
