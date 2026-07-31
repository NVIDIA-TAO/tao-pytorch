# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-aware metric routing for Mask2Former.

This module intentionally has no torch dependency.  Keeping the routing and
metric-name contract independent from the model makes it possible to validate
the task semantics without constructing a model or reserving an accelerator.
"""

from math import isfinite


SUPPORTED_TASK_MODES = frozenset({"semantic", "instance", "panoptic"})
SUPPORTED_EVALUATION_SPLITS = frozenset({"val", "test"})


def normalize_task_mode(mode):
    """Return a validated, normalized Mask2Former task mode."""
    normalized = str(mode).strip().lower()
    if normalized not in SUPPORTED_TASK_MODES:
        raise ValueError(
            f"Unsupported Mask2Former task mode {mode!r}; expected one of "
            f"{sorted(SUPPORTED_TASK_MODES)}."
        )
    return normalized


def normalize_evaluation_split(split):
    """Return a validated validation/test split name."""
    normalized = str(split).strip().lower()
    if normalized not in SUPPORTED_EVALUATION_SPLITS:
        raise ValueError(
            f"Unsupported evaluation split {split!r}; expected one of "
            f"{sorted(SUPPORTED_EVALUATION_SPLITS)}."
        )
    return normalized


def instance_metric_names(split):
    """Return the stable COCO mask-AP metric names for an evaluation split."""
    split = normalize_evaluation_split(split)
    return f"segm_{split}_mAP", f"segm_{split}_mAP50"


def semantic_metric_names(mode, split):
    """Return semantic metric names without mislabelling panoptic diagnostics."""
    mode = normalize_task_mode(mode)
    split = normalize_evaluation_split(split)
    if mode == "instance":
        raise ValueError("Instance mode must use COCO mask AP, not semantic metrics.")
    if mode == "semantic":
        return "mIoU", "all_acc"
    return (
        f"panoptic_{split}_semantic_mIoU_diagnostic",
        f"panoptic_{split}_semantic_all_acc_diagnostic",
    )


def ordered_coco_category_ids(coco):
    """Return COCO category IDs in the dataset's declared contiguous order."""
    dataset = getattr(coco, "dataset", None)
    categories = dataset.get("categories") if isinstance(dataset, dict) else None
    if not isinstance(categories, list) or not categories:
        raise ValueError("COCO annotations must contain a non-empty categories list.")

    category_ids = []
    for category in categories:
        if not isinstance(category, dict) or isinstance(category.get("id"), bool):
            raise ValueError("Every COCO category must contain an integer id.")
        category_id = category.get("id")
        if not isinstance(category_id, int):
            raise ValueError("Every COCO category must contain an integer id.")
        category_ids.append(category_id)

    if len(set(category_ids)) != len(category_ids):
        raise ValueError("COCO category IDs must be unique.")
    return tuple(category_ids)


def validate_instance_category_contract(coco, num_classes):
    """Validate and return the model-index to dataset-category mapping."""
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer.")
    category_ids = ordered_coco_category_ids(coco)
    if len(category_ids) != num_classes:
        raise ValueError(
            "Mask2Former class count does not match the COCO category inventory: "
            f"model={num_classes}, dataset={len(category_ids)}."
        )
    return category_ids


def create_instance_evaluator(coco, num_classes, evaluator_cls=None):
    """Create the repository COCO evaluator for globally aggregated mask AP."""
    category_ids = validate_instance_category_contract(coco, num_classes)
    if evaluator_cls is None:
        # Lazy import keeps pure task-routing tests independent from torch.
        from nvidia_tao_pytorch.cv.deformable_detr.utils.coco_eval import (  # pylint: disable=import-outside-toplevel
            CocoEvaluator,
        )
        evaluator_cls = CocoEvaluator
    return evaluator_cls(
        coco,
        iou_types=["segm"],
        eval_class_ids=list(category_ids),
    )


def finalize_instance_evaluator(evaluator, split, is_print=False):
    """Synchronize a COCO evaluator and return finite mask AP metrics."""
    metric_name, metric50_name = instance_metric_names(split)
    evaluator.synchronize_between_processes()
    evaluator.overall_accumulate()
    evaluator.overall_summarize(is_print=is_print)

    coco_eval = getattr(evaluator, "coco_eval", {})
    if "segm" not in coco_eval:
        raise RuntimeError("Mask2Former instance evaluator did not produce segm results.")
    stats = getattr(coco_eval["segm"], "stats", None)
    if stats is None or len(stats) < 2:
        raise RuntimeError("COCO segmentation evaluator did not produce AP statistics.")

    map_value = float(stats[0])
    map50_value = float(stats[1])
    if not isfinite(map_value) or not isfinite(map50_value):
        raise RuntimeError(
            "COCO segmentation evaluator produced a non-finite AP statistic."
        )
    return {
        metric_name: map_value,
        metric50_name: map50_value,
    }
