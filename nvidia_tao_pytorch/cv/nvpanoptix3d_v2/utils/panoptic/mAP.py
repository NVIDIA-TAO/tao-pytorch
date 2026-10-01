# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mAP, mAP50, and mAP25 for 3D instance segmentation.

The implementation follows the SegVGGT/mmdet3d ScanNet instance-segmentation
protocol: class-aware matching, ScanNet AP integration, and ignored false
positives when most of a prediction lies in void.  mAP averages AP at IoU
thresholds 0.50:0.05:0.90; mAP50 and mAP25 use their named thresholds.

Only the two operations needed by NVPanoptix3Dv2 evaluation are exposed:

* :func:`prepare_instance_map_sample` converts one multi-view prediction and
  its ground truth into a compact intersection table.
* :func:`evaluate_instance_map_segvggt` aggregates those tables into mAP,
  mAP50, and mAP25.

Keeping intersections rather than full per-instance masks makes epoch-level
evaluation practical for dense multi-view inputs and DDP object gathering.
"""

from typing import Dict, Iterable, List, Sequence

import numpy as np


_MAP_THRESHOLDS = np.round(np.arange(0.50, 0.95, 0.05), 2)


def map_ids(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Map arbitrary positive segment IDs to dense IDs in ``[1, len(ids)]``."""
    values = np.asarray(values).reshape(-1)
    dense = np.zeros(values.shape, dtype=np.int32)
    if ids.size == 0:
        return dense

    order = np.argsort(ids)
    sorted_ids = ids[order]
    positions = np.searchsorted(sorted_ids, values)
    valid = positions < sorted_ids.size
    valid_indices = np.flatnonzero(valid)
    valid[valid_indices] &= (
        sorted_ids[positions[valid_indices]] == values[valid_indices]
    )
    # Dense indices follow the metadata order, not the sorted-ID order.
    dense[valid] = order[positions[valid]] + 1
    return dense


def ground_truth_segments(
    instance_ids: np.ndarray,
    class_ids: np.ndarray,
    num_categories: int,
    evaluated_category_ids: Sequence[int],
) -> List[Dict[str, int]]:
    """Infer labels for non-background GT instances in evaluated classes."""
    evaluated_ids = set(evaluated_category_ids)
    valid = (
        (instance_ids > 0) &
        (class_ids >= 0) &
        (class_ids < num_categories)
    )
    valid_instance_ids = instance_ids[valid]
    valid_class_ids = class_ids[valid]
    if valid_instance_ids.size == 0:
        return []

    # Class IDs are invariant for a scene-level instance UID. Selecting the
    # first valid occurrence avoids constructing one full-resolution bool mask
    # per instance while retaining deterministic behavior.
    unique_ids, first_indices = np.unique(
        valid_instance_ids, return_index=True,
    )
    segments = []
    for instance_id, first_index in zip(unique_ids, first_indices):
        category_id = int(valid_class_ids[first_index])
        if category_id in evaluated_ids:
            segments.append({
                "id": int(instance_id),
                "category_id": category_id,
            })
    return segments


def prepare_instance_map_sample(
    pred_pan: np.ndarray,
    pred_segments: Sequence[Dict],
    gt_instance_ids: np.ndarray,
    gt_class_ids: np.ndarray,
    *,
    num_categories: int,
    evaluated_category_ids: Iterable[int],
    min_points: int = 100,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Build the compact matching data for one multi-view sample.

    Args:
        pred_pan: Predicted segment-ID map with shape ``[S, H, W]``.
        pred_segments: Metadata containing ``id``, ``category_id``, and
            ``score`` for every predicted segment.
        gt_instance_ids: Ground-truth instance-ID map, shaped like
            ``pred_pan``. Zero denotes background/void.
        gt_class_ids: Dense ground-truth category IDs, shaped like
            ``gt_instance_ids``.
        num_categories: Size of the dense category vocabulary.
        evaluated_category_ids: Category IDs included in instance mAP. For a
            panoptic taxonomy this should contain only ``isthing`` classes.
        min_points: Predictions and GT instances smaller than this are ignored.

    Returns:
        A per-category dictionary of scores, areas, intersections, and void
        intersections. Its size depends on the number of instances rather
        than the number of input pixels.
    """
    pred_flat = np.asarray(pred_pan, dtype=np.int64).reshape(-1)
    gt_inst_flat = np.asarray(gt_instance_ids, dtype=np.int64).reshape(-1)
    gt_cls_flat = np.asarray(gt_class_ids, dtype=np.int64).reshape(-1)
    if pred_flat.shape != gt_inst_flat.shape or gt_inst_flat.shape != gt_cls_flat.shape:
        raise ValueError(
            "pred_pan, gt_instance_ids, and gt_class_ids must contain the "
            f"same number of points, got {pred_flat.size}, "
            f"{gt_inst_flat.size}, and {gt_cls_flat.size}"
        )
    if num_categories <= 0:
        raise ValueError(f"num_categories must be positive, got {num_categories}")
    if min_points <= 0:
        raise ValueError(f"min_points must be positive, got {min_points}")

    evaluated_ids = tuple(dict.fromkeys(int(cid) for cid in evaluated_category_ids))
    if any(cid < 0 or cid >= num_categories for cid in evaluated_ids):
        raise ValueError(
            f"evaluated category IDs must be in [0, {num_categories}), "
            f"got {evaluated_ids}"
        )

    pred_metadata = []
    seen_pred_ids = set()
    for segment in pred_segments:
        segment_id = int(segment["id"])
        category_id = int(segment["category_id"])
        if segment_id <= 0 or segment_id in seen_pred_ids:
            raise ValueError(f"predicted segment IDs must be unique and positive: {segment_id}")
        seen_pred_ids.add(segment_id)
        if not 0 <= category_id < num_categories:
            continue
        pred_metadata.append({
            "id": segment_id,
            "category_id": category_id,
            "score": float(segment["score"]),
        })

    gt_metadata = ground_truth_segments(
        gt_inst_flat, gt_cls_flat, num_categories, evaluated_ids,
    )
    pred_ids = np.asarray([s["id"] for s in pred_metadata], dtype=np.int64)
    gt_ids = np.asarray([s["id"] for s in gt_metadata], dtype=np.int64)
    pred_dense = map_ids(pred_flat, pred_ids)
    gt_dense = map_ids(gt_inst_flat, gt_ids)

    pred_areas = np.bincount(pred_dense, minlength=pred_ids.size + 1)
    gt_areas = np.bincount(gt_dense, minlength=gt_ids.size + 1)
    pair_ids = pred_dense.astype(np.int64) * (gt_ids.size + 1) + gt_dense
    intersections = np.bincount(
        pair_ids,
        minlength=(pred_ids.size + 1) * (gt_ids.size + 1),
    ).reshape(pred_ids.size + 1, gt_ids.size + 1)

    cache: Dict[int, Dict[str, np.ndarray]] = {}
    for category_id in evaluated_ids:
        pred_indices = np.asarray([
            index + 1
            for index, segment in enumerate(pred_metadata)
            if segment["category_id"] == category_id and
            pred_areas[index + 1] >= min_points
        ], dtype=np.int64)
        all_category_gt_indices = np.asarray([
            index + 1
            for index, segment in enumerate(gt_metadata)
            if segment["category_id"] == category_id
        ], dtype=np.int64)
        gt_indices = all_category_gt_indices[
            gt_areas[all_category_gt_indices] >= min_points
        ]
        small_gt_indices = all_category_gt_indices[
            gt_areas[all_category_gt_indices] < min_points
        ]
        if pred_indices.size == 0 and gt_indices.size == 0:
            continue

        scores_by_dense_id = {
            index + 1: segment["score"]
            for index, segment in enumerate(pred_metadata)
        }
        cache[category_id] = {
            "pred_scores": np.asarray(
                [scores_by_dense_id[int(i)] for i in pred_indices],
                dtype=np.float64,
            ),
            "pred_areas": pred_areas[pred_indices].astype(np.int64, copy=False),
            "gt_areas": gt_areas[gt_indices].astype(np.int64, copy=False),
            "inter": intersections[np.ix_(pred_indices, gt_indices)].astype(
                np.int64, copy=False,
            ),
            # Background, invalid GT, and non-evaluated categories are void.
            # Same-class GT smaller than the protocol's region threshold is
            # ignored as well.
            "void_inter": (
                intersections[pred_indices, 0] +
                intersections[np.ix_(pred_indices, small_gt_indices)].sum(axis=1)
            ).astype(np.int64, copy=False),
        }
    return cache


def ap_for_category(
    sample_caches: Sequence[Dict[int, Dict[str, np.ndarray]]],
    category_id: int,
    iou_threshold: float,
) -> float:
    """Compute AP for one category and one IoU threshold."""
    y_true = []
    y_score = []
    has_gt = False
    has_pred = False
    hard_false_negatives = 0
    pred_visited = {}

    for sample_index, cache in enumerate(sample_caches):
        category = cache.get(category_id)
        if category is None:
            continue
        for pred_index in range(category["pred_scores"].shape[0]):
            pred_visited[(sample_index, pred_index)] = False

    for sample_index, cache in enumerate(sample_caches):
        category = cache.get(category_id)
        if category is None:
            continue

        pred_scores = category["pred_scores"]
        pred_areas = category["pred_areas"]
        gt_areas = category["gt_areas"]
        intersections = category["inter"]
        void_intersections = category["void_inter"]
        num_predictions = pred_scores.shape[0]
        num_ground_truth = gt_areas.shape[0]
        has_gt |= num_ground_truth > 0
        has_pred |= num_predictions > 0

        current_true = np.ones(num_ground_truth, dtype=np.float64)
        current_score = np.full(num_ground_truth, -np.inf, dtype=np.float64)
        current_match = np.zeros(num_ground_truth, dtype=bool)
        duplicate_true = []
        duplicate_score = []

        for gt_index in range(num_ground_truth):
            found_match = False
            for pred_index in range(num_predictions):
                if pred_visited[(sample_index, pred_index)]:
                    continue
                intersection = int(intersections[pred_index, gt_index])
                union = int(
                    pred_areas[pred_index] + gt_areas[gt_index] - intersection
                )
                if union <= 0 or intersection / union <= iou_threshold:
                    continue

                confidence = float(pred_scores[pred_index])
                if current_match[gt_index]:
                    previous = current_score[gt_index]
                    current_score[gt_index] = max(previous, confidence)
                    duplicate_true.append(0.0)
                    duplicate_score.append(min(previous, confidence))
                else:
                    found_match = True
                    current_match[gt_index] = True
                    current_score[gt_index] = confidence
                    pred_visited[(sample_index, pred_index)] = True
            if not found_match:
                hard_false_negatives += 1

        current_true = current_true[current_match]
        current_score = current_score[current_match]
        if duplicate_true:
            current_true = np.concatenate([
                current_true, np.asarray(duplicate_true, dtype=np.float64),
            ])
            current_score = np.concatenate([
                current_score, np.asarray(duplicate_score, dtype=np.float64),
            ])

        # Unmatched predictions are false positives unless most of their area
        # is void, matching the ScanNet ignore rule.
        for pred_index in range(num_predictions):
            if pred_visited[(sample_index, pred_index)]:
                continue
            overlaps_gt = False
            for gt_index in range(num_ground_truth):
                intersection = int(intersections[pred_index, gt_index])
                union = int(
                    pred_areas[pred_index] + gt_areas[gt_index] - intersection
                )
                if union > 0 and intersection / union > iou_threshold:
                    overlaps_gt = True
                    break
            if overlaps_gt:
                continue

            ignored_fraction = (
                int(void_intersections[pred_index]) /
                max(int(pred_areas[pred_index]), 1)
            )
            if ignored_fraction <= iou_threshold:
                current_true = np.append(current_true, 0.0)
                current_score = np.append(
                    current_score, float(pred_scores[pred_index]),
                )

        y_true.append(current_true)
        y_score.append(current_score)

    if not has_gt:
        return float("nan")
    if not has_pred:
        return 0.0

    y_true_array = np.concatenate(y_true) if y_true else np.zeros(0)
    y_score_array = np.concatenate(y_score) if y_score else np.zeros(0)
    if y_score_array.size == 0:
        return 0.0

    order = np.argsort(y_score_array)
    sorted_true = y_true_array[order]
    sorted_scores = y_score_array[order]
    true_cumsum = np.cumsum(sorted_true)
    _, unique_indices = np.unique(sorted_scores, return_index=True)

    precision = np.zeros(len(unique_indices) + 1, dtype=np.float64)
    recall = np.zeros(len(unique_indices) + 1, dtype=np.float64)
    num_examples = len(sorted_scores)
    num_true_examples = float(true_cumsum[-1]) if true_cumsum.size else 0.0
    cumsum_padded = np.append(true_cumsum, 0.0)
    for result_index, score_index in enumerate(unique_indices):
        cumsum = cumsum_padded[score_index - 1]
        true_positives = num_true_examples - cumsum
        false_positives = num_examples - score_index - true_positives
        false_negatives = cumsum + hard_false_negatives
        precision_denominator = true_positives + false_positives
        recall_denominator = true_positives + false_negatives
        precision[result_index] = (
            true_positives / precision_denominator
            if precision_denominator > 0 else 0.0
        )
        recall[result_index] = (
            true_positives / recall_denominator
            if recall_denominator > 0 else 0.0
        )

    precision[-1] = 1.0
    recall[-1] = 0.0
    recall_for_convolution = np.concatenate(([recall[0]], recall, [0.0]))
    step_widths = np.convolve(
        recall_for_convolution, [-0.5, 0.0, 0.5], "valid",
    )
    return float(np.dot(precision, step_widths))


def finite_mean(values: Sequence[float]) -> float:
    """Mean finite values, returning zero when no evaluated GT is present."""
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else 0.0


def evaluate_instance_map_segvggt(
    sample_caches: Sequence[Dict[int, Dict[str, np.ndarray]]],
    evaluated_category_ids: Iterable[int],
) -> Dict[str, float]:
    """Aggregate prepared samples into mAP, mAP50, and mAP25 percentages."""
    category_ids = tuple(dict.fromkeys(int(cid) for cid in evaluated_category_ids))

    ap50 = [
        ap_for_category(sample_caches, category_id, 0.50)
        for category_id in category_ids
    ]
    ap25 = [
        ap_for_category(sample_caches, category_id, 0.25)
        for category_id in category_ids
    ]
    ap = []
    for category_id in category_ids:
        category_ap = [
            ap_for_category(sample_caches, category_id, float(threshold))
            for threshold in _MAP_THRESHOLDS
        ]
        ap.append(
            finite_mean(category_ap)
            if any(np.isfinite(value) for value in category_ap)
            else float("nan")
        )

    return {
        "mAP": 100.0 * finite_mean(ap),
        "mAP50": 100.0 * finite_mean(ap50),
        "mAP25": 100.0 * finite_mean(ap25),
    }
