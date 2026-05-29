# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation 3D utils."""

import torch
from typing import Dict, Tuple
from collections import OrderedDict


def intersection(ground_truth: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Compute the elementwise intersection of two binary masks.

    Args:
        ground_truth: Ground-truth mask tensor. Will be moved to ``prediction.device``.
        prediction: Predicted mask tensor that defines the output device.

    Returns:
        A float tensor with 1.0 where both masks are True, else 0.0.
    """
    ground_truth = ground_truth.to(prediction.device)
    return (ground_truth.bool() & prediction.bool()).float()


def union(ground_truth: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Compute the elementwise union of two binary masks.

    Args:
        ground_truth: Ground-truth mask tensor. Will be moved to ``prediction.device``.
        prediction: Predicted mask tensor that defines the output device.

    Returns:
        A float tensor with 1.0 where either mask is True, else 0.0.
    """
    ground_truth = ground_truth.to(prediction.device)
    return (ground_truth.bool() | prediction.bool()).float()


def compute_iou(ground_truth: torch.Tensor, prediction: torch.Tensor) -> float:
    """Compute Intersection-over-Union (IoU) between two binary masks.

    Args:
        ground_truth: Ground-truth mask tensor. Interpreted as boolean.
        prediction: Predicted mask tensor. Interpreted as boolean and defines the device.

    Returns:
        The scalar IoU value as a Python float.
    """
    ground_truth = ground_truth.to(prediction.device)
    num_intersection = float(torch.sum(intersection(ground_truth, prediction)))
    num_union = float(torch.sum(union(ground_truth, prediction)))
    iou = 0.0 if num_union == 0 else num_intersection / num_union
    return iou


class PQStatCategory:
    """Accumulates per-category statistics used to compute PQ/SQ/RQ.

    Attributes:
        is_thing: Whether this category is a "thing" class (vs. "stuff").
    """

    def __init__(self, is_thing=True):
        """Initialize an empty accumulator for one category.

        Args:
            is_thing: Flag indicating whether the category is a "thing" class.
        """
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.n = 0
        self.is_thing = is_thing

    def __iadd__(self, pq_stat_cat):
        """In-place add another category accumulator into this one.

        Args:
            pq_stat_cat: Another :class:`PQStatCategory` with the same semantics.

        Returns:
            ``self`` (mutated).
        """
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        self.n += pq_stat_cat.n
        return self

    @property
    def as_metric(self):
        """Return a plain-Python dict view of the accumulated statistics."""
        return {"iou": self.iou, "tp": self.tp, "fp": self.fp, "fn": self.fn, "n": self.n}

    def __repr__(self):
        """Return a string representation of the accumulated statistics."""
        return str(self.as_metric)


class PanopticReconstructionQuality:
    """Compute Panoptic Reconstruction Quality (PRQ) style statistics in 3D.

    This implementation matches instances between prediction and ground truth
    within each semantic category based on 3D mask IoU. It accumulates per-category
    counts and IoU sums so that PQ/SQ/RQ-style scores can be reduced across samples.
    """

    def __init__(self, metadata, matching_threshold=0.25, ignore_labels=None, reduction="mean"):
        """Initialize PRQ accumulator.

        Args:
            metadata: Dataset metadata object with ``class_info`` entries containing
                ``trainId``, ``name`` and ``isthing``.
            matching_threshold: IoU threshold above which a predicted instance is
                considered matched to a GT instance (within the same semantic label).
            ignore_labels: Optional list of semantic labels to ignore (e.g., freespace).
                If None, defaults to ``[0, 12]``.
            reduction: Reduction mode used by :meth:`reduce_mean`. Currently only
                ``"mean"`` is implemented/used.
        """
        super().__init__()

        # ignore freespace label and ceiling
        if ignore_labels is None:
            ignore_labels = [0, 12]
        self.ignore_labels = ignore_labels

        self.matching_threshold = matching_threshold

        self.categories = {}

        for item in metadata.class_info:
            self.categories[item["trainId"]] = PQStatCategory(item["isthing"] == 1)

        self.categories[0] = PQStatCategory(True)

        self.reduction = reduction
        self.metadata = metadata
        self.class_id_to_name = OrderedDict()
        self.class_id_to_name[0] = "void"
        for item in self.metadata.class_info:
            self.class_id_to_name[item["trainId"]] = item["name"]

    def add(
        self, prediction: Dict[int, Tuple[torch.Tensor, int]],
        ground_truth: Dict[int, Tuple[torch.Tensor, int]]
    ) -> Dict:
        """Match predicted and GT instances for one sample and return per-class stats.

        Args:
            prediction: Mapping from prediction instance id to ``(mask, semantic_label)``.
                ``mask`` is interpreted as a binary tensor.
            ground_truth: Mapping from GT instance id to ``(mask, semantic_label)``.

        Returns:
            A dict mapping semantic label (``trainId``) to a :class:`PQStatCategory`
            containing this sample's statistics for that label.
        """
        matched_ids_ground_truth = set()
        matched_ids_prediction = set()

        per_sample_result = {}
        for item in self.metadata.class_info:
            per_sample_result[item["trainId"]] = PQStatCategory(item["isthing"] == 1)

        # True Positives
        for ground_truth_instance_id, (
            ground_truth_instance_mask, ground_truth_semantic_label
        ) in ground_truth.items():
            self.categories[ground_truth_semantic_label].n += 1
            per_sample_result[ground_truth_semantic_label].n += 1

            for prediction_instance_id, (
                prediction_instance_mask, prediction_semantic_label
            ) in prediction.items():

                # 0: Check if prediction was already matched
                if prediction_instance_id in matched_ids_prediction:
                    continue

                # 1: Check if they have the same label
                are_same_category = ground_truth_semantic_label == prediction_semantic_label

                if not are_same_category:
                    continue

                # 2: Compute overlap and check if they are overlapping enough
                overlap = compute_iou(
                    ground_truth_instance_mask, prediction_instance_mask
                )
                is_match = overlap > self.matching_threshold

                if is_match:
                    self.categories[ground_truth_semantic_label].iou += overlap
                    self.categories[ground_truth_semantic_label].tp += 1

                    per_sample_result[ground_truth_semantic_label].iou += overlap
                    per_sample_result[ground_truth_semantic_label].tp += 1

                    matched_ids_ground_truth.add(ground_truth_instance_id)
                    matched_ids_prediction.add(prediction_instance_id)
                    break

        # False Negatives
        for ground_truth_instance_id, (
            _, ground_truth_semantic_label
        ) in ground_truth.items():
            # 0: Check if ground truth has not yet been matched
            if ground_truth_instance_id not in matched_ids_ground_truth:
                self.categories[ground_truth_semantic_label].fn += 1
                per_sample_result[ground_truth_semantic_label].fn += 1

        # False Positives
        for prediction_instance_id, (_, prediction_semantic_label) in prediction.items():
            # 0: Check if prediction has not yet been matched
            if prediction_instance_id not in matched_ids_prediction:
                self.categories[prediction_semantic_label].fp += 1
                per_sample_result[prediction_semantic_label].fp += 1

        return per_sample_result

    def add_sample(self, sample):
        """Accumulate a per-sample result (from :meth:`add`) into global totals.

        Args:
            sample: Dict mapping semantic label to :class:`PQStatCategory`.
        """
        for k in sample.keys():
            if k in self.categories:
                self.categories[k] += sample[k]

    def convert_metric_to_tensor(self, metric_per_sample: Dict[int, PQStatCategory]) -> torch.Tensor:
        """Convert per-sample per-class stats into a tensor for downstream aggregation.

        Args:
            metric_per_sample: Dict keyed by semantic label (trainId) containing
                :class:`PQStatCategory` values.

        Returns:
            A tensor of shape ``(num_classes, 5)`` with dtype inferred by PyTorch.
        """
        metric_tensor = []
        for item in self.metadata.class_info:
            metric_per_class = metric_per_sample[item["trainId"]]  # PQStatCategory
            metric_per_class = [
                metric_per_class.as_metric[key] for key in ["iou", "tp", "fp", "fn", "n"]
            ]
            metric_tensor.append(metric_per_class)
        return torch.tensor(metric_tensor)

    def revert_metric_to_dict(self, metric_tensor: torch.Tensor) -> Dict[int, Dict[str, float]]:
        """Convert a tensor representation back to a dict representation.

        Args:
            metric_tensor: Tensor of shape ``(num_classes, 5)`` where columns are
                ``[iou, tp, fp, fn, n]``.

        Returns:
            Dict mapping class label to a plain dict with keys ``iou/tp/fp/fn/n``.
        """
        metric_per_sample = {}
        metric_per_sample[0] = {"iou": 0.0, "tp": 0, "fp": 0, "fn": 0, "n": 0}
        for class_idx in range(metric_tensor.shape[0]):
            metric_per_sample[class_idx + 1] = {
                "iou": float(metric_tensor[class_idx][0]),
                "tp": float(metric_tensor[class_idx][1]),
                "fp": float(metric_tensor[class_idx][2]),
                "fn": float(metric_tensor[class_idx][3]),
                "n": int(metric_tensor[class_idx][4])
            }
        return metric_per_sample

    def reduce_mean(self, categories: Dict[int, PQStatCategory]) -> Dict[str, float]:
        """Reduce per-class stats into mean PQ/SQ/RQ and a per-class breakdown.

        Args:
            categories: Dict mapping class label to a dict-like object with keys
                ``iou``, ``tp``, ``fp``, ``fn``, ``n``. (This typically comes from
                ``PQStatCategory.as_metric`` for each class.)

        Returns:
            An OrderedDict containing:
            - ``pq/sq/rq``: mean over classes with at least one GT instance (n>0)
            - ``n``: number of contributing classes
            - One entry per contributing class name with its ``pq/sq/rq/n``
        """
        pq, sq, rq, n = 0, 0, 0, 0

        per_class_results = {}

        for class_label, class_stats in categories.items():
            iou = class_stats["iou"]
            tp = class_stats["tp"]
            fp = class_stats["fp"]
            fn = class_stats["fn"]
            num_samples = class_stats["n"]

            if tp + fp + fn == 0:
                per_class_results[class_label] = {"pq": 0.0, "sq": 0.0, "rq": 0.0, "n": 0}
                continue

            if num_samples > 0:
                n += 1
                pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
                sq_class = iou / tp if tp != 0 else 0
                rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
                per_class_results[class_label] = {
                    "pq": pq_class,
                    "sq": sq_class,
                    "rq": rq_class,
                    "n": num_samples
                }
                pq += pq_class
                sq += sq_class
                rq += rq_class

        results = OrderedDict()
        results.update({"pq": pq / n, "sq": sq / n, "rq": rq / n, "n": n})

        for label, per_class_result in per_class_results.items():
            if per_class_result["n"] > 0:
                label = self.class_id_to_name[label]
                results[label] = {
                    "pq": per_class_result["pq"],
                    "sq": per_class_result["sq"],
                    "rq": per_class_result["rq"],
                    "n": per_class_result["n"],
                }

        return results
