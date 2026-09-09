# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Losses for NVPanoptix3Dv2 reasoning training."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.utils.metric_depth import MetricDepthLoss


def dice_loss_prob(prob: Tensor, target: Tensor, eps: float = 1.0) -> Tensor:
    """Per-row soft Dice loss for flattened probability/target tensors."""
    num = 2.0 * (prob * target).sum(-1) + eps
    den = prob.sum(-1) + target.sum(-1) + eps
    return 1.0 - num / den


class NVPanoptix3Dv2ReasoningLoss(nn.Module):
    """Text CE plus best-query SAM mask and score losses.

    SAM3 emits many object queries for one prompt. For the single-target
    task, the target query is selected by the lowest detached mask cost against
    the GT instance. That query receives BCE and Dice supervision, while the
    SAM query scores receive a one-hot BCE target.
    """

    def __init__(
        self,
        text_weight: float = 1.0,
        mask_weight: float = 20.0,
        dice_weight: float = 1.0,
        score_weight: float = 1.0,
        metric_weight: float = 0.0,
        metric_silog_weight: float = 1.0,
        metric_absrel_weight: float = 0.1,
        metric_silog_lambda: float = 0.85,
        metric_min_depth: float = 0.1,
        metric_max_depth: float = 30.0,
    ):
        super().__init__()
        self.text_weight = float(text_weight)
        self.mask_weight = float(mask_weight)
        self.dice_weight = float(dice_weight)
        self.score_weight = float(score_weight)
        self.metric_weight = float(metric_weight)
        self.metric_depth_criterion = MetricDepthLoss(
            silog_weight=metric_silog_weight,
            absrel_weight=metric_absrel_weight,
            silog_lambda=metric_silog_lambda,
            min_depth=metric_min_depth,
            max_depth=metric_max_depth,
        ) if self.metric_weight > 0 else None

    @staticmethod
    def text_loss(lm_logits: Tensor, labels: Tensor) -> Tensor:
        """Shifted causal-LM cross entropy."""
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    @staticmethod
    def resize_gt(gt_masks: Tensor, size: Tuple[int, int]) -> Tensor:
        """Resize ground-truth masks to the prediction resolution."""
        if gt_masks.shape[-2:] == size:
            return gt_masks.float()
        return F.interpolate(gt_masks[:, None].float(), size=size, mode="area")[:, 0]

    def sam_loss(
        self,
        pred_masks: Tensor,
        pred_logits: Tensor,
        gt_masks: Tensor,
        valid: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``(bce, dice, score, best_idx)`` for valid samples."""
        B, Q, h, w = pred_masks.shape
        device = pred_masks.device
        if not bool(valid.any()):
            zero = pred_masks.sum() * 0.0
            return zero, zero, zero, torch.full((B,), -1, dtype=torch.long, device=device)

        gt = self.resize_gt(gt_masks.to(device), (h, w))
        pred_flat = pred_masks.reshape(B, Q, h * w)
        gt_flat = gt.reshape(B, 1, h * w).expand(B, Q, h * w)

        bce_q = F.binary_cross_entropy_with_logits(pred_flat, gt_flat, reduction="none").mean(-1)
        dice_q = dice_loss_prob(pred_flat.sigmoid(), gt_flat)
        cost = bce_q + dice_q
        best_idx = cost.detach().argmin(dim=1)

        valid = valid.to(device).bool()
        b = torch.arange(B, device=device)
        bce = bce_q[b[valid], best_idx[valid]].mean()
        dice = dice_q[b[valid], best_idx[valid]].mean()

        score_tgt = torch.zeros_like(pred_logits)
        score_tgt[b[valid], best_idx[valid]] = 1.0
        score = F.binary_cross_entropy_with_logits(pred_logits[valid], score_tgt[valid])
        best_idx = torch.where(valid, best_idx, torch.full_like(best_idx, -1))
        return bce, dice, score, best_idx

    def metric_depth_loss(
        self,
        out: Dict[str, object],
        gt_depth: Optional[Tensor],
    ) -> Tensor:
        """SILog + AbsRel supervision on the metric depth, when the head is enabled."""
        metric_depth = out.get("metric_depth")
        if self.metric_depth_criterion is None or gt_depth is None or metric_depth is None:
            device = out["pred_masks"].device
            return torch.zeros((), device=device)
        if metric_depth.dim() == 5:
            metric_depth = metric_depth.squeeze(-1)
        gt = gt_depth.to(metric_depth.device).float()
        if metric_depth.shape[-2:] != gt.shape[-2:]:
            B, S = metric_depth.shape[:2]
            metric_depth = F.interpolate(
                metric_depth.reshape(B * S, 1, *metric_depth.shape[2:]),
                size=tuple(gt.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, S, *gt.shape[-2:])
        return self.metric_depth_criterion(
            metric_depth, gt,
        )["loss_metric_total"]

    def forward(
        self,
        out: Dict[str, object],
        gt_masks: Tensor,
        mask_valid: Tensor,
        gt_depth: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute total loss and scalar logs."""
        device = out["pred_masks"].device
        if out.get("labels") is not None and self.text_weight > 0:
            l_text = self.text_loss(out["lm_logits"], out["labels"])
        else:
            l_text = torch.zeros((), device=device)

        valid = mask_valid.to(device).bool() & out["seg_valid"].to(device).bool()
        l_bce, l_dice, l_score, best_idx = self.sam_loss(
            out["pred_masks"], out["pred_logits"], gt_masks, valid
        )
        total = (
            self.text_weight * l_text +
            self.mask_weight * l_bce +
            self.dice_weight * l_dice +
            self.score_weight * l_score
        )
        l_metric = self.metric_depth_loss(out, gt_depth)
        total = total + self.metric_weight * l_metric
        logs = {
            "loss_total": float(total.detach()),
            "loss_text": float(l_text.detach()),
            "loss_mask": float(l_bce.detach()),
            "loss_dice": float(l_dice.detach()),
            "loss_score": float(l_score.detach()),
            "loss_metric": float(l_metric.detach()),
            "n_mask_valid": float(valid.sum().item()),
            "best_query_mean": (
                float(best_idx[best_idx >= 0].float().mean())
                if bool((best_idx >= 0).any()) else -1.0
            ),
        }
        params = out.get("scale_shift_params")
        if isinstance(params, dict) and "scale" in params:
            logs["metric_scale_mean"] = float(params["scale"].detach().mean())
        return total, logs
