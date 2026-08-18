# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classification metrics for video_clip evaluation (VAD-R1-ps style).

Mirrors the cosmos-embed1 ``validation_eval_callback`` definition: each
**category is a query** and all **videos are documents**.  For a category, the
videos whose label matches it are relevant; all others are non-relevant.

- MAP: per-category Average Precision (sum of precision@k at each relevant
  video's rank / |relevant|); MAP = mean AP across categories.
- MRR: per-category 1 / (rank of first relevant video); mean across categories.
- Top-K hit: per-video, whether its ground-truth category is in the top-K
  categories ranked by similarity (video-as-query classification view).
- Macro precision/recall/F1: argmax-category prediction vs ground truth.
- Categories in ``exclude`` (e.g. Normal/Abnormal) are dropped from MAP, MRR
  and the macro averages, matching cosmos-embed1.

This module is pure NumPy and re-uses :func:`compute_ap` so the AP kernel is
identical to the retrieval path.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from nvidia_tao_pytorch.multimodal.video_clip.model.evaluation.metrics import (
    compute_ap,
)


@dataclass
class ClassificationMetrics:
    """Container for classification metrics.

    Attributes:
        map_score: Mean Average Precision across (non-excluded) categories.
        mrr: Mean Reciprocal Rank across (non-excluded) categories.
        macro_precision: Macro-averaged precision over non-excluded categories.
        macro_recall: Macro-averaged recall over non-excluded categories.
        macro_f1: Macro-averaged F1 over non-excluded categories.
        top_k_hit: Mapping k -> fraction of videos whose GT category is top-K.
        per_category: Per-category precision/recall/f1/support/ap.
        num_categories: Number of categories scored for MAP/MRR.
        num_samples: Number of videos evaluated.
    """

    map_score: float = 0.0
    mrr: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    top_k_hit: Dict[int, float] = field(default_factory=dict)
    per_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    num_categories: int = 0
    num_samples: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Flatten to a scalar dict for logging."""
        result = {
            'mAP': self.map_score,
            'MRR': self.mrr,
            'macro_precision': self.macro_precision,
            'macro_recall': self.macro_recall,
            'macro_f1': self.macro_f1,
            'num_categories': self.num_categories,
            'num_samples': self.num_samples,
        }
        for k, v in self.top_k_hit.items():
            result[f'top{k}_hit'] = v
        return result

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Classification Metrics (samples={self.num_samples}, "
            f"categories={self.num_categories}):",
            f"  mAP: {self.map_score:.4f}",
            f"  MRR: {self.mrr:.4f}",
            f"  macro-P: {self.macro_precision:.4f}",
            f"  macro-R: {self.macro_recall:.4f}",
            f"  macro-F1: {self.macro_f1:.4f}",
        ]
        for k in sorted(self.top_k_hit.keys()):
            lines.append(f"  top{k}_hit: {self.top_k_hit[k]:.4f}")
        return '\n'.join(lines)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization with a numerical floor."""
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10)


def evaluate_classification(
    video_embs: np.ndarray,
    category_embs: np.ndarray,
    category_names: Sequence[str],
    sample_cat_idx: np.ndarray,
    top_k: Tuple[int, ...] = (1, 5, 10),
    exclude: Sequence[str] = (),
) -> ClassificationMetrics:
    """Compute classification metrics (MAP/MRR/top-K/macro P-R-F1).

    Args:
        video_embs: Video embeddings of shape (N, D).
        category_embs: Category prototype embeddings of shape (C, D), aligned
            row-for-row with ``category_names``.
        category_names: Length-C list of category label strings.
        sample_cat_idx: Length-N int array; the category index (into
            ``category_names``) that each video belongs to.
        top_k: K values for the top-K hit rate.
        exclude: Category names excluded from MAP, MRR and macro averages
            (case-insensitively), e.g. ("Normal", "Abnormal").

    Returns:
        Populated :class:`ClassificationMetrics`.
    """
    video_embs = np.asarray(video_embs, dtype=np.float32)
    category_embs = np.asarray(category_embs, dtype=np.float32)
    sample_cat_idx = np.asarray(sample_cat_idx).astype(int)
    category_names = list(category_names)

    n_samples = video_embs.shape[0]
    n_categories = len(category_names)
    exclude_lower = {e.lower() for e in exclude}

    if n_samples == 0 or n_categories == 0:
        return ClassificationMetrics(num_samples=n_samples)

    video_norm = _l2_normalize(video_embs)
    cat_norm = _l2_normalize(category_embs)

    # vid_to_cat: (N, C) used for top-K and classification (video as query).
    vid_to_cat = video_norm @ cat_norm.T
    # cat_to_vid: (C, N) used for MAP/MRR (category as query).
    cat_to_vid = cat_norm @ video_norm.T

    # --- Top-K hit rate (video-as-query classification view) ---
    sorted_cat_per_video = np.argsort(-vid_to_cat, axis=1)
    top_k_hits = {k: 0 for k in top_k}
    for i in range(n_samples):
        gt_idx = sample_cat_idx[i]
        rank_positions = np.where(sorted_cat_per_video[i] == gt_idx)[0]
        if len(rank_positions) > 0:
            rank = rank_positions[0] + 1
            for k in top_k:
                if rank <= k:
                    top_k_hits[k] += 1
    top_k_hit = {k: top_k_hits[k] / n_samples for k in top_k}

    # --- MAP & MRR: each category is a query, videos are documents ---
    sorted_vid_per_cat = np.argsort(-cat_to_vid, axis=1)  # (C, N)
    average_precisions: List[float] = []
    reciprocal_ranks: List[float] = []
    per_category_ap: Dict[str, float] = {}
    for j, cat in enumerate(category_names):
        if cat.lower() in exclude_lower:
            continue
        relevant_mask = sample_cat_idx == j
        num_relevant = int(relevant_mask.sum())
        if num_relevant == 0:
            continue
        ranking = sorted_vid_per_cat[j]
        # sorted_labels: relevance of each document in ranked order.
        sorted_labels = relevant_mask[ranking].astype(np.float32)
        ap = compute_ap(sorted_labels)
        average_precisions.append(ap)
        per_category_ap[cat] = ap
        first_rel = np.where(sorted_labels == 1)[0]
        if len(first_rel) > 0:
            reciprocal_ranks.append(1.0 / (first_rel[0] + 1))

    map_score = float(np.mean(average_precisions)) if average_precisions else 0.0
    mrr_score = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    # --- Per-category precision/recall/F1 (argmax prediction) ---
    pred_idx = np.argmax(vid_to_cat, axis=1)
    per_category: Dict[str, Dict[str, float]] = {}
    macro_p = macro_r = macro_f1 = 0.0
    num_scored = 0
    for j, cat in enumerate(category_names):
        tp = int(np.sum((pred_idx == j) & (sample_cat_idx == j)))
        fp = int(np.sum((pred_idx == j) & (sample_cat_idx != j)))
        support = int(np.sum(sample_cat_idx == j))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        per_category[cat] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'ap': per_category_ap.get(cat, 0.0),
        }
        if cat.lower() not in exclude_lower and support > 0:
            macro_p += precision
            macro_r += recall
            macro_f1 += f1
            num_scored += 1

    if num_scored > 0:
        macro_p /= num_scored
        macro_r /= num_scored
        macro_f1 /= num_scored

    return ClassificationMetrics(
        map_score=map_score,
        mrr=mrr_score,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        top_k_hit=top_k_hit,
        per_category=per_category,
        num_categories=len(average_precisions),
        num_samples=n_samples,
    )


def log_classification_metrics(
    metrics: ClassificationMetrics, prefix: str = ""
) -> None:
    """Log classification metrics."""
    head = f"{prefix}Classification Metrics" if prefix else "Classification Metrics"
    logging.info("%s:\n%s", head, str(metrics))
