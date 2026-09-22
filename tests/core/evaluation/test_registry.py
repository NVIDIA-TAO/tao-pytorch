# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Evaluator registry + enabled-selection."""

from types import SimpleNamespace

import pytest

from nvidia_tao_pytorch.core.evaluation import EVALUATOR_REGISTRY, build_enabled_evaluators
from nvidia_tao_pytorch.core.evaluation.knn import KNNEvaluator
from nvidia_tao_pytorch.core.evaluation.segmentation import SegmentationEvaluator


@pytest.mark.unit
def test_registry_has_all_evaluators():
    """KNN, segmentation, and retrieval evaluators self-register on import."""
    assert {"knn", "segmentation", "retrieval"} <= set(EVALUATOR_REGISTRY)


@pytest.mark.unit
def test_online_support_flags():
    """KNN supports online hooks; the seg probe is offline-only."""
    assert KNNEvaluator.supports_online is True
    assert SegmentationEvaluator.supports_online is False
    assert SegmentationEvaluator.requires_fit is True


@pytest.mark.unit
def test_build_enabled_evaluators_filters_by_enabled():
    """Only evaluators whose config block is enabled are instantiated."""
    cfg = SimpleNamespace(
        knn=SimpleNamespace(enabled=True),
        segmentation=SimpleNamespace(enabled=False),
        retrieval=SimpleNamespace(enabled=False),
    )
    names = {ev.name for ev in build_enabled_evaluators(cfg)}
    assert names == {"knn"}


@pytest.mark.unit
def test_build_enabled_evaluators_multiple():
    """Multiple enabled blocks yield multiple evaluators."""
    cfg = SimpleNamespace(
        knn=SimpleNamespace(enabled=True),
        segmentation=SimpleNamespace(enabled=True),
        retrieval=SimpleNamespace(enabled=False),
    )
    names = {ev.name for ev in build_enabled_evaluators(cfg)}
    assert names == {"knn", "segmentation"}
