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

"""Core metric computations for CLIP retrieval evaluation.

This module provides fundamental metric functions for retrieval tasks.

Functions:
    compute_ap: Average Precision for retrieval.
    compute_ndcg: NDCG@K for ranking quality.
    compute_auc: Area Under ROC Curve.
    batched: Utility for batching iterables.
"""

from itertools import islice

import numpy as np


def compute_ap(sorted_labels: np.ndarray) -> float:
    """Compute Average Precision for a single query.

    Args:
        sorted_labels: Binary relevance labels sorted by descending score.

    Returns:
        Average Precision score.
    """
    n_pos = np.sum(sorted_labels)
    if n_pos == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, label in enumerate(sorted_labels):
        if label:
            hits += 1
            precision_sum += hits / (rank + 1)

    return precision_sum / n_pos


def compute_ndcg(sorted_labels: np.ndarray, k: int) -> float:
    """Compute NDCG@k for a single query.

    Args:
        sorted_labels: Binary relevance labels sorted by descending score.
        k: Number of top results to consider.

    Returns:
        NDCG@k score.
    """
    n_pos = np.sum(sorted_labels)
    if n_pos == 0:
        return 0.0

    actual_k = min(k, len(sorted_labels))
    top_k = sorted_labels[:actual_k]
    gains = top_k / np.log2(np.arange(2, actual_k + 2))
    dcg = np.sum(gains)

    ideal_k = min(int(n_pos), actual_k)
    ideal = np.ones(ideal_k)
    ideal_gains = ideal / np.log2(np.arange(2, ideal_k + 2))
    idcg = np.sum(ideal_gains)

    return dcg / idcg if idcg > 0 else 0.0


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUC (Area Under ROC Curve).

    Args:
        scores: Similarity/confidence scores.
        labels: Binary labels (1 for positive, 0 for negative).

    Returns:
        AUC score.
    """
    n_pos = np.sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 1.0 if n_neg == 0 else 0.0

    sorted_idx = np.argsort(scores)
    sorted_labels = labels[sorted_idx]

    ranks = np.arange(1, len(labels) + 1, dtype=np.float64)
    pos_rank_sum = np.sum(ranks[sorted_labels == 1])

    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def batched(iterable, n: int):
    """Batch data into lists of length n. The last batch may be shorter.

    Based on more-itertools impl, to be replaced by python 3.12 itertools.batched.

    Args:
        iterable: Input iterable to batch.
        n: Batch size.

    Yields:
        Lists of up to n items.
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch
