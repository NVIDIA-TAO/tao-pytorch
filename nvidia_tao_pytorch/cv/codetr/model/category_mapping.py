# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-inference category-grouping utilities.

Maps the model's original class IDs to user-defined output categories
(e.g. ``{"bicycle": ["bicycle", "motorcycle"], "car": ["car", "bus", "truck"]}``)
and re-applies per-group soft-NMS so duplicates within a merged group are
suppressed (the model's per-class soft-NMS does not catch them because the
queries have different original labels).

Design:
    Two-step pipeline applied AFTER ``PostProcess.forward``:
    1. ``build_output_label_map_and_remap`` — derives the new label map and a
       ``original_class_id -> output_category_id`` lookup. Done once per run.
    2. ``apply_category_mapping_groupnms`` — per-image: drop unmapped detections,
       relabel surviving ones with the output category id, optionally re-run
       per-output-category soft-NMS.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.deformable_detr.model.post_process import _soft_nms


LabelMap = List[Dict[str, Any]]


def build_output_label_map_and_remap(
    input_label_map: LabelMap,
    category_mapping: Mapping[str, Sequence[str]],
) -> Tuple[LabelMap, Dict[int, int]]:
    """Build the output label map + ``original_id -> output_id`` lookup.

    Args:
        input_label_map: list of ``{"id": int, "name": str}`` (the model's
            original classmap).
        category_mapping: ordered mapping from output category name to the list
            of original class names that should be merged into it.

    Returns:
        output_label_map: list of ``{"id": int, "name": str}`` with one entry
            per output category. IDs are sequential 0..K-1 in the iteration
            order of ``category_mapping``.
        remap: dict mapping original class id (int) to output category id
            (int). Original classes not present in any group are absent from
            this dict and will be dropped by ``apply_category_mapping_groupnms``.
    """
    name_to_id = {entry["name"]: int(entry["id"]) for entry in input_label_map}
    output_label_map: LabelMap = []
    remap: Dict[int, int] = {}

    missing: List[str] = []
    for new_id, (output_name, original_names) in enumerate(category_mapping.items()):
        output_label_map.append({"id": new_id, "name": output_name})
        for orig_name in original_names:
            if orig_name not in name_to_id:
                missing.append(orig_name)
                continue
            orig_id = name_to_id[orig_name]
            if orig_id in remap and remap[orig_id] != new_id:
                first_group = next(
                    e["name"] for e in output_label_map if e["id"] == remap[orig_id]
                )
                logging.warning(
                    "[category_mapping] original class %r is mapped to both %r and %r; "
                    "keeping the first (%r).",
                    orig_name, first_group, output_name, first_group,
                )
                continue
            remap[orig_id] = new_id

    if missing:
        logging.warning(
            "[category_mapping] %d original class names not found in classmap "
            "and were ignored: %s",
            len(missing), missing,
        )
    if not remap:
        raise ValueError(
            "category_mapping produced an empty remap — no original classes "
            "matched the classmap. Check the names in category_mapping vs "
            "dataset.infer_data_sources.classmap."
        )
    return output_label_map, remap


def _remap_one(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    remap: Dict[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop detections not in remap and relabel survivors with output ids."""
    if labels.numel() == 0:
        return boxes, labels, scores
    cpu_labels = labels.detach().cpu().tolist()
    keep_idx: List[int] = []
    new_labels: List[int] = []
    for i, lid in enumerate(cpu_labels):
        if lid in remap:
            keep_idx.append(i)
            new_labels.append(remap[lid])
    if not keep_idx:
        empty = torch.zeros(0, device=boxes.device)
        return (
            torch.zeros(0, 4, device=boxes.device),
            torch.zeros(0, dtype=labels.dtype, device=labels.device),
            empty,
        )
    keep_t = torch.tensor(keep_idx, device=boxes.device, dtype=torch.long)
    new_labels_t = torch.tensor(new_labels, device=labels.device, dtype=labels.dtype)
    return boxes.index_select(0, keep_t), new_labels_t, scores.index_select(0, keep_t)


def apply_category_mapping_groupnms(
    results: List[Dict[str, Any]],
    remap: Dict[int, int],
    soft_nms_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Per-image: remap labels, drop unmapped detections, optionally re-NMS.

    Args:
        results: output of ``PostProcess.forward`` (list of dicts with keys
            ``boxes``, ``labels``, ``scores``, ``image_names``).
        remap: ``original_id -> output_id`` lookup from
            :func:`build_output_label_map_and_remap`.
        soft_nms_kwargs: if provided, an additional per-output-category soft-NMS
            pass runs after remapping. Expected keys: ``method``,
            ``iou_threshold``, ``sigma``. Pass ``None`` to skip the second NMS.

    Returns:
        New list of dicts with the same shape as ``results`` but with
        ``labels`` replaced by output category ids and detections filtered
        accordingly.
    """
    out: List[Dict[str, Any]] = []
    for res in results:
        boxes, labels, scores = res["boxes"], res["labels"], res["scores"]
        boxes, labels, scores = _remap_one(boxes, labels, scores, remap)

        if soft_nms_kwargs is not None and labels.numel() > 0:
            keep_ids: List[torch.Tensor] = []
            decayed_scores: List[torch.Tensor] = []
            for cls_id in labels.unique():
                cls_mask = labels == cls_id
                cls_keep, cls_scores = _soft_nms(
                    boxes[cls_mask],
                    scores[cls_mask],
                    method=soft_nms_kwargs.get("method", "linear"),
                    iou_threshold=soft_nms_kwargs.get("iou_threshold", 0.8),
                    sigma=soft_nms_kwargs.get("sigma", 0.5),
                )
                global_indices = cls_mask.nonzero(as_tuple=False).squeeze(1)
                keep_ids.append(global_indices[cls_keep])
                decayed_scores.append(cls_scores)
            if keep_ids:
                keep_ids_t = torch.cat(keep_ids)
                scores = torch.cat(decayed_scores)
                boxes = boxes.index_select(0, keep_ids_t)
                labels = labels.index_select(0, keep_ids_t)
            else:
                boxes = torch.zeros(0, 4, device=boxes.device)
                labels = torch.zeros(0, dtype=labels.dtype, device=labels.device)
                scores = torch.zeros(0, device=scores.device)

        new_res = dict(res)
        new_res["boxes"] = boxes
        new_res["labels"] = labels
        new_res["scores"] = scores
        out.append(new_res)
    return out


# Distinct PIL-compatible color names; cycled when there are more output
# categories than entries in the palette.
_DEFAULT_PALETTE = (
    "red", "lime", "blue", "yellow", "magenta", "cyan",
    "orange", "deeppink", "lightseagreen", "gold", "blueviolet", "tomato",
)


def build_default_color_map(output_label_map: LabelMap) -> Dict[str, str]:
    """Pick a stable color per output category, cycling a small distinct palette."""
    return {
        entry["name"]: _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)]
        for i, entry in enumerate(output_label_map)
    }


def soft_nms_kwargs_from_model_config(model_config) -> Optional[Dict[str, Any]]:
    """Extract soft-NMS params from a model config when ``soft_nms_enabled`` is True.

    Returns ``None`` when soft-NMS is disabled (callers should skip the second
    NMS pass in that case).
    """
    if not bool(model_config.get("soft_nms_enabled", False)):
        return None
    return {
        "method": model_config.get("soft_nms_method", "linear"),
        "iou_threshold": float(model_config.get("soft_nms_iou_threshold", 0.8)),
        "sigma": float(model_config.get("soft_nms_sigma", 0.5)),
    }
