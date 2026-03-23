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
