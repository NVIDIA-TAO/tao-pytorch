# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for Mask Grounding DINO COCO metric routing."""

import ast
from pathlib import Path

import pytest

from nvidia_tao_pytorch.cv.mask_grounding_dino.model.pl_gdino_model import (
    coco_metric_values,
)


MODEL_PATH = (
    Path(__file__).parents[3]
    / "nvidia_tao_pytorch"
    / "cv"
    / "mask_grounding_dino"
    / "model"
    / "pl_gdino_model.py"
)

pytestmark = pytest.mark.cv_unit


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    model_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MaskGDINOPlModel"
    )
    return next(
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _called_names(function: ast.FunctionDef) -> list[str]:
    names = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    return names


def test_object_detection_uses_distributed_coco_evaluator():
    source = MODEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(MODEL_PATH))

    imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "CocoEvaluator" in imports
    assert "OD_Evaluator" not in imports

    for name in ("on_validation_epoch_start", "on_test_epoch_start"):
        assert "CocoEvaluator" in _called_names(_function(tree, name))

    for name in ("on_validation_epoch_end", "on_test_epoch_end"):
        calls = _called_names(_function(tree, name))
        assert "synchronize_between_processes" in calls
        assert "overall_accumulate" in calls
        assert "overall_summarize" in calls


def test_object_detection_emits_task_correct_bbox_and_mask_metrics():
    source = MODEL_PATH.read_text(encoding="utf-8")

    assert 'metric_name = f"{prefix}val_{key}"' in source
    assert 'metric_name = f"{prefix}test_{key}"' in source
    assert '"mAP"' in source
    assert '"mAP50"' in source


def test_coco_metric_values_maps_ordered_stats_to_runtime_kpis():
    """The production conversion preserves COCOeval's metric ordering."""
    metrics = coco_metric_values(range(12))

    assert metrics["mAP"] == 0.0
    assert metrics["mAP50"] == 1.0
    assert metrics["AR100"] == 8.0
    assert metrics["AR_large"] == 11.0
