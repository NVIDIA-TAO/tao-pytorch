# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical-point-cloud metrics for reasoning segmentation.

Each validation sample is scored once on a checkpoint-independent canonical
point domain: valid metric-depth pixels in ``[view, y, x]`` order. Prediction
and ground truth therefore always address identical points. The XYZ values are
not needed for binary set IoU; their fixed validity and ordering are sufficient.

Only the requested metrics are reported:

* ``mIoU``: mean per-query point IoU.
* ``mAP50``: fraction of queries whose point IoU is at least 0.50.
* ``mAP25``: fraction of queries whose point IoU is at least 0.25.

The threshold metrics retain the original benchmark's per-query hit-rate
semantics under their requested names. Values are fractions in ``[0, 1]``.
"""

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.utils.reasoning.masks import (
    build_target_masks,
)


def point_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Return binary set IoU for prediction and GT on identical points."""
    pred = np.asarray(pred, dtype=bool).reshape(-1)
    gt = np.asarray(gt, dtype=bool).reshape(-1)
    if pred.shape != gt.shape:
        raise ValueError(
            f"prediction and GT must address identical points: {pred.shape} != {gt.shape}"
        )
    intersection = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    return 0.0 if union == 0.0 else intersection / union


def aggregate(inter_union: Iterable[Tuple[float, float]]) -> Dict[str, float]:
    """Aggregate intersection/union pairs into mIoU, mAP50, and mAP25."""
    pairs = list(inter_union)
    if not pairs:
        return {}
    ious = np.asarray([
        0.0 if union == 0.0 else intersection / union
        for intersection, union in pairs
    ], dtype=np.float64)
    return {
        "mIoU": float(ious.mean()),
        "mAP50": float((ious >= 0.50).mean()),
        "mAP25": float((ious >= 0.25).mean()),
    }


class CanonicalPointCloudMetrics:
    """Additive accumulator for canonical point-cloud reasoning metrics."""

    def __init__(self):
        self._iou_sum = 0.0
        self._map50_hits = 0
        self._map25_hits = 0
        self._num_samples = 0

    def add_counts(self, intersection: float, union: float) -> None:
        """Add one query using its canonical-point intersection and union."""
        iou = 0.0 if union == 0.0 else float(intersection) / float(union)
        self._iou_sum += iou
        self._map50_hits += int(iou >= 0.50)
        self._map25_hits += int(iou >= 0.25)
        self._num_samples += 1

    def add(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """Add one query from binary masks defined on identical points."""
        pred = np.asarray(pred, dtype=bool).reshape(-1)
        gt = np.asarray(gt, dtype=bool).reshape(-1)
        if pred.shape != gt.shape:
            raise ValueError(
                "prediction and GT must address identical points: "
                f"{pred.shape} != {gt.shape}"
            )
        self.add_counts(
            float(np.logical_and(pred, gt).sum()),
            float(np.logical_or(pred, gt).sum()),
        )

    def raw_state(self) -> List[float]:
        """Return the summable state used for DDP all-reduce."""
        return [
            self._iou_sum,
            float(self._map50_hits),
            float(self._map25_hits),
            float(self._num_samples),
        ]

    def load_raw_state(self, state: Iterable[float]) -> None:
        """Load a state previously returned by :meth:`raw_state`."""
        values = list(state)
        if len(values) != 4:
            raise ValueError(f"expected four metric accumulators, got {len(values)}")
        self._iou_sum = float(values[0])
        self._map50_hits = int(round(float(values[1])))
        self._map25_hits = int(round(float(values[2])))
        self._num_samples = int(round(float(values[3])))

    def compute(self) -> Dict[str, float]:
        """Return only mIoU, mAP50, and mAP25."""
        if self._num_samples == 0:
            return {}
        return {
            "mIoU": self._iou_sum / self._num_samples,
            "mAP50": self._map50_hits / self._num_samples,
            "mAP25": self._map25_hits / self._num_samples,
        }


def score_batch_on_canonical_points(
    meter: CanonicalPointCloudMetrics,
    out,
    pan_inst,
    target_inst,
    canonical_valid,
    mask_threshold: float,
) -> None:
    """Score a multi-view batch on its fixed metric-depth point domain.

    All views participate in the canonical domain. A predicted ``[SEG]`` mask
    is inserted at its bound view; views without a binding remain empty. Every
    positive-target sample is scored, so a missing ``[SEG]`` prediction becomes
    an empty mask rather than silently disappearing from the metric.

    Args:
        meter: Destination metric accumulator.
        out: Model outputs containing SAM ``pred_masks``, ``pred_logits``, and
            binding indices ``seg_sample_idx`` / ``seg_view_idx``.
        pan_inst: ``[B, S, H, W]`` canonical GT instance IDs.
        target_inst: ``[B]`` target instance IDs.
        canonical_valid: ``[B, S, H, W]`` mask of valid canonical depth points.
        mask_threshold: Probability threshold for the selected SAM mask.
    """
    if pan_inst.shape != canonical_valid.shape:
        raise ValueError(
            "panoptic IDs and canonical validity must have identical shape: "
            f"{tuple(pan_inst.shape)} != {tuple(canonical_valid.shape)}"
        )
    batch_size, num_views, height, width = pan_inst.shape
    device = pan_inst.device
    pred_by_sample = torch.zeros(
        (batch_size, num_views, height, width),
        dtype=torch.bool,
        device=device,
    )

    sample_indices = out["seg_sample_idx"].to(device)
    view_indices = out["seg_view_idx"].to(device)
    num_bindings = int(sample_indices.numel())
    if num_bindings:
        pred_masks = out["pred_masks"]
        pred_logits = out["pred_logits"]
        selected_queries = pred_logits.float().argmax(dim=1)
        selected_masks = pred_masks[
            torch.arange(num_bindings, device=pred_masks.device),
            selected_queries,
        ]
        probabilities = F.interpolate(
            selected_masks.unsqueeze(1).float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()[:, 0].to(device)
        predictions = probabilities > float(mask_threshold)
        for binding in range(num_bindings):
            sample = int(sample_indices[binding].item())
            view = int(view_indices[binding].item())
            pred_by_sample[sample, view] |= predictions[binding]

    ground_truth = build_target_masks(pan_inst, target_inst)
    canonical_valid = canonical_valid.to(device=device, dtype=torch.bool)
    for sample in range(batch_size):
        if int(target_inst[sample].item()) <= 0:
            continue
        valid = canonical_valid[sample]
        if not bool(valid.any()):
            raise ValueError(
                f"canonical point cloud is empty for batch sample {sample}; "
                "reasoning evaluation requires valid metric depth"
            )
        gt = ground_truth[sample][valid]
        if not bool(gt.any()):
            raise ValueError(
                f"target instance {int(target_inst[sample].item())} has no points "
                f"in the canonical cloud for batch sample {sample}"
            )
        pred = pred_by_sample[sample][valid]
        intersection = float((pred & gt).sum().item())
        union = float((pred | gt).sum().item())
        meter.add_counts(intersection, union)
