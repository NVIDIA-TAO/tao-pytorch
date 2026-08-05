# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared embedding evaluation — source of truth for SSL + RADIO + vfm-eval.

Backbones are unified behind the ``(summary, features)``
model-adapter contract (`model_adapter.py`). Each metric is an ``Evaluator``
registered in ``base.EVALUATOR_REGISTRY``, runnable offline (`run(ctx)`) or
online (validation hooks). The metric implementations (`knn.py`,
`segmentation.py`, `transforms.py`, `datasets/`, `caching.py`) are ported from
`vfm-eval/c-radiov4`; `retrieval.py` is ours.
"""

from nvidia_tao_pytorch.core.evaluation.base import (  # noqa: F401
    EVALUATOR_REGISTRY,
    EvalContext,
    Evaluator,
    build_enabled_evaluators,
    register_evaluator,
)
from nvidia_tao_pytorch.core.evaluation.model_adapter import (  # noqa: F401
    ADAPTER_REGISTRY,
    BackboneV2Adapter,
    DinoV2Adapter,
    MAEViTAdapter,
    ModelAdapter,
    RadioStudentAdapter,
    build_adapter,
    features_to_map,
    load_tao_state_dict,
)

# Importing an evaluator module registers it in EVALUATOR_REGISTRY (via the
# @register_evaluator decorator).
from nvidia_tao_pytorch.core.evaluation import knn  # noqa: F401,E402
from nvidia_tao_pytorch.core.evaluation import segmentation  # noqa: F401,E402
from nvidia_tao_pytorch.core.evaluation import retrieval  # noqa: F401,E402
