# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KNN classification for distillation validation — re-export shim.

Dependency inversion (Epic C): the KNN vote math now lives in the shared
``nvidia_tao_pytorch.core.evaluation.knn`` module (source of truth, consumed by
the SSL ``evaluate`` action and the vfm-eval harness too). This module is kept as
a thin back-compatibility shim so existing imports
(``from ...distillation.knn_classification import knn_top1_accuracy``) keep working
unchanged. The functions are the exact same implementations that previously lived
here — relocated, not modified — so distillation KNN-top1 numbers are unchanged.
"""

from nvidia_tao_pytorch.core.evaluation.knn import (  # noqa: F401
    _get_vote_cls,
    distributed_topk,
    knn_predict,
    knn_top1_accuracy,
)

__all__ = ["_get_vote_cls", "distributed_topk", "knn_predict", "knn_top1_accuracy"]
