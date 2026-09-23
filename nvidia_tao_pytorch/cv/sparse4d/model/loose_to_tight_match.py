# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-camera matching and cross-camera aggregation for 2D supervision."""

from collections import defaultdict
from typing import Dict, List, Optional

import torch
from scipy.optimize import linear_sum_assignment


__all__ = [
    "pairwise_giou",
    "pairwise_l1",
    "match_camera",
    "aggregate_consistency",
]


_EPS = 1e-6


def pairwise_giou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise generalized IoU for ``(N, 4)`` and ``(M, 4)`` boxes."""
    boxes_a = boxes_a[:, None, :]
    boxes_b = boxes_b[None, :, :]

    intersection_x_min = torch.maximum(boxes_a[..., 0], boxes_b[..., 0])
    intersection_y_min = torch.maximum(boxes_a[..., 1], boxes_b[..., 1])
    intersection_x_max = torch.minimum(boxes_a[..., 2], boxes_b[..., 2])
    intersection_y_max = torch.minimum(boxes_a[..., 3], boxes_b[..., 3])
    intersection = (intersection_x_max - intersection_x_min).clamp_min(0.0) * (
        intersection_y_max - intersection_y_min
    ).clamp_min(0.0)

    area_a = (boxes_a[..., 2] - boxes_a[..., 0]).clamp_min(0.0) * (
        boxes_a[..., 3] - boxes_a[..., 1]
    ).clamp_min(0.0)
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]).clamp_min(0.0) * (
        boxes_b[..., 3] - boxes_b[..., 1]
    ).clamp_min(0.0)
    union = (area_a + area_b - intersection).clamp_min(_EPS)
    iou = intersection / union

    enclosing_x_min = torch.minimum(boxes_a[..., 0], boxes_b[..., 0])
    enclosing_y_min = torch.minimum(boxes_a[..., 1], boxes_b[..., 1])
    enclosing_x_max = torch.maximum(boxes_a[..., 2], boxes_b[..., 2])
    enclosing_y_max = torch.maximum(boxes_a[..., 3], boxes_b[..., 3])
    enclosing_area = (
        (enclosing_x_max - enclosing_x_min).clamp_min(0.0) *
        (enclosing_y_max - enclosing_y_min).clamp_min(0.0)
    ).clamp_min(_EPS)
    return iou - (enclosing_area - union) / enclosing_area


def pairwise_l1(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise coordinate L1 distance for xyxy boxes."""
    return (boxes_a[:, None, :] - boxes_b[None, :, :]).abs().sum(dim=-1)


def _empty_matches(reference: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Construct an empty match result on ``reference``'s device."""
    return {
        "q": torch.zeros(0, dtype=torch.long, device=reference.device),
        "box": reference.new_zeros((0, 4)),
        "cls": torch.zeros(0, dtype=torch.long, device=reference.device),
        "score": reference.new_zeros(0),
        "giou": reference.new_zeros(0),
    }


def match_camera(
    amodal: torch.Tensor,
    valid: torch.Tensor,
    q_cls_prob: torch.Tensor,
    det_boxes: torch.Tensor,
    det_cls: torch.Tensor,
    det_score: torch.Tensor,
    img_diag: float,
    giou_thr: float = 0.3,
    w_giou: float = 2.0,
    w_l1: float = 1.0,
    w_cls: float = 1.0,
    score_thr: float = 0.0,
    class_gate: bool = True,
) -> Dict[str, torch.Tensor]:
    """Hungarian-match projected queries to one camera's detections.

    Matching is deliberately non-differentiable and operates on detached boxes.
    The caller uses returned indices to recompute the loss from live projections.
    """
    if amodal.ndim != 2 or amodal.shape[-1] != 4:
        raise ValueError("amodal must have shape (N, 4)")
    if q_cls_prob.ndim != 2 or q_cls_prob.shape[0] != amodal.shape[0]:
        raise ValueError("q_cls_prob must have shape (N, num_classes)")
    if valid.shape != amodal.shape[:1]:
        raise ValueError("valid must have shape (N,)")

    empty = _empty_matches(amodal)
    device = amodal.device
    detections = det_boxes.to(dtype=amodal.dtype, device=device)
    detection_classes = det_cls.to(dtype=torch.long, device=device)
    detection_scores = det_score.to(dtype=amodal.dtype, device=device)
    query_probabilities = q_cls_prob.to(device=device)
    valid = valid.to(dtype=torch.bool, device=device)

    if detections.ndim != 2 or detections.shape[-1] != 4:
        raise ValueError("det_boxes must have shape (M, 4)")
    if detection_classes.shape != detections.shape[:1]:
        raise ValueError("det_cls must have shape (M,)")
    if detection_scores.shape != detections.shape[:1]:
        raise ValueError("det_score must have shape (M,)")

    keep_detection = detection_scores >= score_thr
    if not bool(keep_detection.any()) or not bool(valid.any()):
        return empty

    detections = detections[keep_detection]
    detection_classes = detection_classes[keep_detection]
    detection_scores = detection_scores[keep_detection]
    if bool(
        (
            (detection_classes < 0) |
            (detection_classes >= query_probabilities.shape[1])
        ).any()
    ):
        raise ValueError("det_cls contains an out-of-range class ID")

    query_indices = valid.nonzero(as_tuple=False).squeeze(dim=-1)
    projected = amodal[query_indices].detach()
    query_classes = query_probabilities[query_indices].argmax(dim=-1)
    generalized_iou = pairwise_giou(projected, detections)
    coordinate_l1 = pairwise_l1(projected, detections) / max(float(img_diag), _EPS)
    class_cost = -query_probabilities[query_indices][:, detection_classes]
    cost = w_giou * (1.0 - generalized_iou) + w_l1 * coordinate_l1 + w_cls * class_cost
    if class_gate:
        class_mismatch = query_classes[:, None] != detection_classes[None, :]
        cost = cost + class_mismatch.to(cost.dtype) * 1e6

    row_indices, column_indices = linear_sum_assignment(cost.detach().cpu().numpy())
    rows = torch.as_tensor(row_indices, dtype=torch.long, device=device)
    columns = torch.as_tensor(column_indices, dtype=torch.long, device=device)
    selected_giou = generalized_iou[rows, columns]
    accepted = selected_giou >= giou_thr
    if class_gate:
        accepted = accepted & (query_classes[rows] == detection_classes[columns])

    rows = rows[accepted]
    columns = columns[accepted]
    selected_giou = selected_giou[accepted]
    return {
        "q": query_indices[rows],
        "box": detections[columns],
        "cls": detection_classes[columns],
        "score": detection_scores[columns],
        "giou": selected_giou,
    }


def _infer_device(
    per_cam_matches: List[Dict[str, torch.Tensor]],
    boxes7: Optional[torch.Tensor],
    device,
) -> torch.device:
    """Infer an output device when the caller does not provide one."""
    if device is not None:
        return torch.device(device)
    if boxes7 is not None:
        return boxes7.device
    for matches in per_cam_matches:
        query_indices = matches.get("q")
        if isinstance(query_indices, torch.Tensor):
            return query_indices.device
    return torch.device("cpu")


def aggregate_consistency(
    per_cam_matches: List[Dict[str, torch.Tensor]],
    num_queries: int,
    boxes7: Optional[torch.Tensor] = None,
    min_cams: int = 1,
    dedup_dist: float = 0.0,
    device=None,
):
    """Apply a cross-camera support gate and optional 3D query de-duplication.

    Returns:
        ``(keep_pairs, cls_target, cls_weight)``. ``keep_pairs`` contains tuples
        ``(query_index, camera_index, detection_box, score)``. Unmatched or
        suppressed queries have class target ``-1`` and weight ``0``.
    """
    if num_queries < 0:
        raise ValueError("num_queries must be non-negative")
    if min_cams < 1:
        raise ValueError("min_cams must be at least one")
    if boxes7 is not None and boxes7.shape[0] < num_queries:
        raise ValueError("boxes7 must contain every query")

    output_device = _infer_device(per_cam_matches, boxes7, device)
    matches_by_query = defaultdict(list)
    for camera_index, matches in enumerate(per_cam_matches):
        for match_index in range(int(matches["q"].shape[0])):
            query_index = int(matches["q"][match_index].item())
            if query_index < 0 or query_index >= num_queries:
                raise ValueError("match contains an out-of-range query index")
            matches_by_query[query_index].append(
                (
                    camera_index,
                    matches["box"][match_index],
                    int(matches["cls"][match_index].item()),
                    float(matches["score"][match_index].item()),
                )
            )

    class_target = torch.full(
        (num_queries,),
        -1,
        dtype=torch.long,
        device=output_device,
    )
    class_weight = torch.zeros(
        num_queries,
        dtype=torch.float32,
        device=output_device,
    )
    keep_pairs = []
    for query_index, query_matches in matches_by_query.items():
        if len(query_matches) < min_cams:
            continue
        best_match = max(query_matches, key=lambda match: match[3])
        class_target[query_index] = best_match[2]
        class_weight[query_index] = best_match[3]
        for camera_index, box, _, score in query_matches:
            keep_pairs.append((query_index, camera_index, box, score))

    if dedup_dist > 0.0 and boxes7 is not None:
        positive_queries = [
            query_index
            for query_index in range(num_queries)
            if int(class_target[query_index].item()) >= 0
        ]
        positive_queries.sort(
            key=lambda query_index: float(class_weight[query_index].item()),
            reverse=True,
        )
        centers = boxes7[:, :2]
        suppressed = set()
        for index, query_index in enumerate(positive_queries):
            if query_index in suppressed:
                continue
            for other_query in positive_queries[index + 1:]:
                if other_query in suppressed:
                    continue
                same_class = int(class_target[other_query].item()) == int(
                    class_target[query_index].item()
                )
                if not same_class:
                    continue
                distance = torch.linalg.norm(
                    centers[query_index] - centers[other_query]
                ).item()
                if distance < dedup_dist:
                    suppressed.add(other_query)

        for query_index in suppressed:
            class_target[query_index] = -1
            class_weight[query_index] = 0.0
        keep_pairs = [pair for pair in keep_pairs if pair[0] not in suppressed]

    return keep_pairs, class_target, class_weight
