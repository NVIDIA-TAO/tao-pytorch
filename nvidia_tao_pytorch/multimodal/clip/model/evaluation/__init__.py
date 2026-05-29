# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation utilities for CLIP model.

This module provides tools for evaluating CLIP models on:
    - Text-to-image and image-to-text retrieval

Module Structure:
    metrics: Core metric computations (mAP, recall, NDCG, AUC)
    retrieval: Retrieval evaluation (RetrievalEvaluator, RetrievalMetrics)

Example:
    >>> from nvidia_tao_pytorch.multimodal.clip.model.evaluation import (
    ...     RetrievalEvaluator,
    ...     RetrievalMetrics,
    ... )
"""

from nvidia_tao_pytorch.multimodal.clip.model.evaluation.metrics import (
    batched,
    compute_ap,
    compute_auc,
    compute_ndcg,
)
from nvidia_tao_pytorch.multimodal.clip.model.evaluation.retrieval import (
    log_retrieval_metrics,
    RetrievalEvaluator,
    RetrievalMetrics,
)


__all__ = [
    # Core metrics
    "batched",
    "compute_ap",
    "compute_auc",
    "compute_ndcg",
    # Retrieval
    "log_retrieval_metrics",
    "RetrievalEvaluator",
    "RetrievalMetrics",
]
