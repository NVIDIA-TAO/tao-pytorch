# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory COCO-style Panoptic Quality statistics for OneFormer."""

from collections.abc import Mapping

import numpy as np


IOU = 0
TRUE_POSITIVE = 1
FALSE_POSITIVE = 2
FALSE_NEGATIVE = 3
NUM_PQ_STATS = 4
VOID_SEGMENT_ID = 0


def _segment_index(segments_info, num_classes, *, prediction):
    """Validate and index COCO panoptic segment metadata by segment ID."""
    indexed = {}
    owner = "prediction" if prediction else "ground truth"
    for segment in segments_info:
        if not isinstance(segment, Mapping):
            raise TypeError(f"Each {owner} segment record must be a mapping.")
        segment_id = int(segment["id"])
        category_id = int(segment["category_id"])
        if segment_id <= VOID_SEGMENT_ID:
            raise ValueError(f"{owner} segment IDs must be positive; got {segment_id}.")
        if segment_id in indexed:
            raise ValueError(f"Duplicate {owner} segment ID: {segment_id}.")
        if not 0 <= category_id < num_classes:
            raise ValueError(
                f"{owner} category ID {category_id} is outside [0, {num_classes})."
            )
        indexed[segment_id] = {
            "id": segment_id,
            "category_id": category_id,
            "iscrowd": bool(segment.get("iscrowd", False)),
        }
    return indexed


def _areas(id_map):
    """Return pixel areas keyed by integer segment ID."""
    ids, counts = np.unique(id_map, return_counts=True)
    return {int(segment_id): int(area) for segment_id, area in zip(ids, counts)}


def panoptic_quality_stats(
    prediction,
    prediction_segments_info,
    ground_truth,
    ground_truth_segments_info,
    num_classes,
):
    """Compute additive COCO-style PQ sufficient statistics for one image.

    Statistics are returned as ``[IoU sum, TP, FP, FN]`` per contiguous class.
    Void pixels use segment ID zero. Crowd ground-truth segments are ignored
    according to the COCO panoptic protocol.
    """
    prediction = np.asarray(prediction)
    ground_truth = np.asarray(ground_truth)
    if prediction.ndim != 2 or ground_truth.ndim != 2:
        raise ValueError("Panoptic prediction and ground truth must be 2-D ID maps.")
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            "Panoptic prediction and ground truth must have identical shapes; "
            f"got {prediction.shape} and {ground_truth.shape}."
        )
    if not np.issubdtype(prediction.dtype, np.integer):
        raise TypeError("Panoptic prediction IDs must use an integer dtype.")
    if not np.issubdtype(ground_truth.dtype, np.integer):
        raise TypeError("Panoptic ground-truth IDs must use an integer dtype.")
    if np.any(prediction < 0) or np.any(ground_truth < 0):
        raise ValueError("Panoptic segment IDs cannot be negative.")
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer.")

    pred_segments = _segment_index(
        prediction_segments_info, num_classes, prediction=True
    )
    gt_segments = _segment_index(
        ground_truth_segments_info, num_classes, prediction=False
    )
    pred_areas = _areas(prediction)
    gt_areas = _areas(ground_truth)

    unknown_pred = set(pred_areas) - {VOID_SEGMENT_ID} - set(pred_segments)
    unknown_gt = set(gt_areas) - {VOID_SEGMENT_ID} - set(gt_segments)
    if unknown_pred:
        raise ValueError(
            "Prediction ID map contains segments missing from segments_info: "
            f"{sorted(unknown_pred)}."
        )
    if unknown_gt:
        raise ValueError(
            "Ground-truth ID map contains segments missing from segments_info: "
            f"{sorted(unknown_gt)}."
        )

    # Resizing can remove a tiny GT segment completely. Such a segment is not
    # part of the evaluated raster and therefore must not become a false negative.
    gt_segments = {
        segment_id: segment
        for segment_id, segment in gt_segments.items()
        if segment_id in gt_areas
    }
    pred_segments = {
        segment_id: segment
        for segment_id, segment in pred_segments.items()
        if segment_id in pred_areas
    }

    pair_offset = int(prediction.max(initial=0)) + 1
    combined = (
        ground_truth.astype(np.uint64, copy=False) * np.uint64(pair_offset) +
        prediction.astype(np.uint64, copy=False)
    )
    pair_ids, pair_counts = np.unique(combined, return_counts=True)
    intersections = {
        (int(pair_id // pair_offset), int(pair_id % pair_offset)): int(area)
        for pair_id, area in zip(pair_ids, pair_counts)
    }

    stats = np.zeros((num_classes, NUM_PQ_STATS), dtype=np.float64)
    gt_matched = set()
    pred_matched = set()

    void_overlap = {
        pred_id: intersections.get((VOID_SEGMENT_ID, pred_id), 0)
        for pred_id in pred_segments
    }

    for (gt_id, pred_id), intersection in intersections.items():
        gt_segment = gt_segments.get(gt_id)
        pred_segment = pred_segments.get(pred_id)
        if gt_segment is None or pred_segment is None:
            continue
        if gt_segment["iscrowd"]:
            continue
        if gt_segment["category_id"] != pred_segment["category_id"]:
            continue

        union = (
            gt_areas[gt_id] +
            pred_areas[pred_id] -
            intersection -
            void_overlap[pred_id]
        )
        if union <= 0:
            continue
        iou = intersection / union
        if iou > 0.5:
            category_id = gt_segment["category_id"]
            stats[category_id, IOU] += iou
            stats[category_id, TRUE_POSITIVE] += 1
            gt_matched.add(gt_id)
            pred_matched.add(pred_id)

    crowd_by_category = {}
    for gt_id, segment in gt_segments.items():
        if segment["iscrowd"]:
            crowd_by_category.setdefault(segment["category_id"], []).append(gt_id)
        elif gt_id not in gt_matched:
            stats[segment["category_id"], FALSE_NEGATIVE] += 1

    for pred_id, segment in pred_segments.items():
        if pred_id in pred_matched:
            continue
        ignored_overlap = void_overlap[pred_id]
        for crowd_id in crowd_by_category.get(segment["category_id"], ()):
            ignored_overlap += intersections.get((crowd_id, pred_id), 0)
        if ignored_overlap / pred_areas[pred_id] > 0.5:
            continue
        stats[segment["category_id"], FALSE_POSITIVE] += 1

    return stats


def _mean_quality(stats, category_mask):
    """Reduce per-class sufficient statistics over a category subset."""
    denominator = (
        stats[:, TRUE_POSITIVE] +
        0.5 * stats[:, FALSE_POSITIVE] +
        0.5 * stats[:, FALSE_NEGATIVE]
    )
    valid = category_mask & (denominator > 0)
    category_count = int(valid.sum())
    if category_count == 0:
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "n": 0}

    pq = np.divide(
        stats[:, IOU],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    sq = np.divide(
        stats[:, IOU],
        stats[:, TRUE_POSITIVE],
        out=np.zeros_like(denominator),
        where=stats[:, TRUE_POSITIVE] > 0,
    )
    rq = np.divide(
        stats[:, TRUE_POSITIVE],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return {
        "pq": float(pq[valid].mean()),
        "sq": float(sq[valid].mean()),
        "rq": float(rq[valid].mean()),
        "n": category_count,
    }


def summarize_panoptic_quality(stats, is_thing, class_names=None):
    """Summarize globally aggregated PQ statistics.

    Returned PQ/SQ/RQ values use the unit interval, matching TAO's existing
    segmentation KPI scale.
    """
    stats = np.asarray(stats, dtype=np.float64)
    if stats.ndim != 2 or stats.shape[1] != NUM_PQ_STATS:
        raise ValueError(
            f"PQ stats must have shape (num_classes, {NUM_PQ_STATS}); got {stats.shape}."
        )
    if not np.isfinite(stats).all() or np.any(stats < 0):
        raise ValueError("PQ stats must be finite and non-negative.")

    is_thing = np.asarray(is_thing, dtype=bool)
    if is_thing.shape != (stats.shape[0],):
        raise ValueError("is_thing must contain exactly one flag per class.")
    if class_names is not None and len(class_names) != stats.shape[0]:
        raise ValueError("class_names must contain exactly one name per class.")

    all_result = _mean_quality(stats, np.ones(stats.shape[0], dtype=bool))
    thing_result = _mean_quality(stats, is_thing)
    stuff_result = _mean_quality(stats, ~is_thing)
    summary = {
        "PQ": all_result["pq"],
        "SQ": all_result["sq"],
        "RQ": all_result["rq"],
        "PQ_th": thing_result["pq"],
        "SQ_th": thing_result["sq"],
        "RQ_th": thing_result["rq"],
        "PQ_st": stuff_result["pq"],
        "SQ_st": stuff_result["sq"],
        "RQ_st": stuff_result["rq"],
        "PQ_categories": all_result["n"],
        "PQ_th_categories": thing_result["n"],
        "PQ_st_categories": stuff_result["n"],
    }

    if class_names is not None:
        denominator = (
            stats[:, TRUE_POSITIVE] +
            0.5 * stats[:, FALSE_POSITIVE] +
            0.5 * stats[:, FALSE_NEGATIVE]
        )
        per_class = {}
        for index, name in enumerate(class_names):
            if denominator[index] <= 0:
                continue
            per_class[str(name)] = {
                "PQ": float(stats[index, IOU] / denominator[index]),
                "SQ": (
                    float(stats[index, IOU] / stats[index, TRUE_POSITIVE])
                    if stats[index, TRUE_POSITIVE] > 0
                    else 0.0
                ),
                "RQ": float(stats[index, TRUE_POSITIVE] / denominator[index]),
            }
        summary["per_class"] = per_class
    return summary
