# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static and data-only tests for OneFormer runtime product fixes."""

import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
ONEFORMER_ROOT = REPO_ROOT / "nvidia_tao_pytorch" / "cv" / "oneformer"


def _load_file_module(name, relative_path):
    """Load a pure utility module without importing the model package."""
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


checkpoint_utils = _load_file_module(
    "oneformer_checkpoint_test_target",
    "nvidia_tao_pytorch/cv/oneformer/utils/checkpoint.py",
)
pq_utils = _load_file_module(
    "oneformer_pq_test_target",
    "nvidia_tao_pytorch/cv/oneformer/utils/panoptic_quality.py",
)


class _FakeTensor:
    """Small array wrapper implementing the reducer's tensor surface."""

    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float64)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeDistributed:
    class ReduceOp:
        SUM = "sum"

    calls = []

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def is_initialized():
        return True

    @classmethod
    def all_reduce(cls, tensor, op):
        cls.calls.append(op)
        tensor.value *= 2


class _FakeTorch:
    float64 = np.float64
    distributed = _FakeDistributed

    @staticmethod
    def as_tensor(value, dtype=None, device=None):
        del dtype, device
        return _FakeTensor(value)


_real_torch = sys.modules.get("torch")
sys.modules["torch"] = _FakeTorch
metric_utils = _load_file_module(
    "oneformer_metric_reduction_test_target",
    "nvidia_tao_pytorch/cv/oneformer/utils/metric_reduction.py",
)
if _real_torch is None:
    del sys.modules["torch"]
else:
    sys.modules["torch"] = _real_torch


class _ShapeValue:
    """Tensor-free object exposing only the shape contract used by matching."""

    def __init__(self, shape, value):
        self.shape = shape
        self.value = value


def test_full_checkpoint_prefixes_and_shapes_are_matched():
    """Lightning/DDP wrappers are removed and incompatible heads are excluded."""
    source = {
        "module.model.backbone.weight": _ShapeValue((2, 2), "backbone"),
        "_orig_mod.model.decoder.weight": _ShapeValue((3,), "decoder"),
        "model.class_head.weight": _ShapeValue((9,), "wrong-shape"),
        "model.unused.weight": _ShapeValue((1,), "unused"),
    }
    target = {
        "backbone.weight": _ShapeValue((2, 2), None),
        "decoder.weight": _ShapeValue((3,), None),
        "class_head.weight": _ShapeValue((4,), None),
    }

    matched, report = checkpoint_utils.match_pretrained_state_dict(source, target)

    assert list(matched) == ["decoder.weight", "backbone.weight"]
    assert matched["backbone.weight"].value == "backbone"
    assert report.loaded_key_count == 2
    assert report.missing_keys == ("class_head.weight",)
    assert report.incompatible_shape_keys == ("model.class_head.weight",)
    assert report.unexpected_keys == ("model.unused.weight",)


def test_checkpoint_prefix_collision_is_rejected_deterministically():
    """Two source names may not silently overwrite the same target key."""
    value = _ShapeValue((1,), None)
    with pytest.raises(ValueError, match="same target key"):
        checkpoint_utils.match_pretrained_state_dict(
            {"model.weight": value, "module.model.weight": value},
            {"weight": value},
        )


def test_distributed_reduction_precedes_nonlinear_semantic_summary():
    """A data-only fake collective verifies global sufficient-statistic sums."""
    _FakeDistributed.calls.clear()
    reduced = metric_utils.distributed_sum_array(
        np.asarray([[2.0, 1.0], [4.0, 2.0], [4.0, 4.0]])
    )
    summary = metric_utils.summarize_semantic_iou(*reduced)

    assert _FakeDistributed.calls == [_FakeDistributed.ReduceOp.SUM]
    np.testing.assert_array_equal(
        reduced,
        np.asarray([[4.0, 2.0], [8.0, 4.0], [8.0, 8.0]]),
    )
    assert summary["mIoU"] == 0.5
    assert summary["all_acc"] == 0.375


def test_perfect_panoptic_prediction_has_unit_pq():
    """Exact thing/stuff matches produce unit PQ, SQ, and RQ."""
    id_map = np.asarray([[1, 1, 2], [1, 2, 2]], dtype=np.int64)
    info = [
        {"id": 1, "category_id": 0, "iscrowd": 0},
        {"id": 2, "category_id": 1, "iscrowd": 0},
    ]
    stats = pq_utils.panoptic_quality_stats(id_map, info, id_map, info, 2)
    summary = pq_utils.summarize_panoptic_quality(
        stats, [True, False], ["thing", "stuff"]
    )

    np.testing.assert_array_equal(
        stats,
        np.asarray([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.float64),
    )
    assert summary["PQ"] == 1.0
    assert summary["PQ_th"] == 1.0
    assert summary["PQ_st"] == 1.0


def test_wrong_category_is_false_positive_and_false_negative():
    """Geometric overlap cannot match segments with different categories."""
    id_map = np.ones((2, 2), dtype=np.int64)
    stats = pq_utils.panoptic_quality_stats(
        id_map,
        [{"id": 1, "category_id": 1}],
        id_map,
        [{"id": 1, "category_id": 0}],
        2,
    )

    assert stats[0, pq_utils.FALSE_NEGATIVE] == 1
    assert stats[1, pq_utils.FALSE_POSITIVE] == 1
    assert pq_utils.summarize_panoptic_quality(stats, [True, True])["PQ"] == 0.0


def test_void_and_same_category_crowd_overlap_are_ignored():
    """Predictions mostly covering void/crowd do not become false positives."""
    prediction = np.asarray([[1, 1], [1, 1]], dtype=np.int64)
    ground_truth = np.asarray([[0, 2], [2, 2]], dtype=np.int64)
    stats = pq_utils.panoptic_quality_stats(
        prediction,
        [{"id": 1, "category_id": 0}],
        ground_truth,
        [{"id": 2, "category_id": 0, "iscrowd": 1}],
        1,
    )

    np.testing.assert_array_equal(stats, np.zeros((1, 4), dtype=np.float64))


@pytest.mark.parametrize(
    "prediction,prediction_info,ground_truth,ground_truth_info,error",
    [
        (
            np.asarray([[3]], dtype=np.int64),
            [],
            np.asarray([[0]], dtype=np.int64),
            [],
            "missing from segments_info",
        ),
        (
            np.asarray([[1]], dtype=np.float64),
            [{"id": 1, "category_id": 0}],
            np.asarray([[1]], dtype=np.int64),
            [{"id": 1, "category_id": 0}],
            "integer dtype",
        ),
    ],
)
def test_invalid_panoptic_inputs_fail_closed(
    prediction, prediction_info, ground_truth, ground_truth_info, error
):
    """Malformed panoptic evidence never enters PQ reporting."""
    with pytest.raises((TypeError, ValueError), match=error):
        pq_utils.panoptic_quality_stats(
            prediction,
            prediction_info,
            ground_truth,
            ground_truth_info,
            1,
        )


def test_task_routing_and_distributed_reduction_are_explicit_in_runtime():
    """Production AST exposes task branches and reduces raw statistics first."""
    model_path = ONEFORMER_ROOT / "model" / "pl_oneformer.py"
    source = model_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "load_pretrained_weights" in functions
    assert "panoptic_quality_stats" in ast.unparse(functions["_eval_step"])
    assert "summarize_panoptic_quality" in ast.unparse(
        functions["_record_panoptic_summary"]
    )
    assert "distributed_sum_array" in ast.unparse(
        functions["_global_semantic_stats"]
    )
    assert "distributed_sum_array" in ast.unparse(
        functions["_global_panoptic_stats"]
    )
    assert "sync_dist=False" in ast.unparse(functions["_log_value"])
    assert "is_global_zero" in ast.unparse(functions["on_validation_epoch_end"])
    assert "is_global_zero" in ast.unparse(functions["on_test_epoch_end"])


def test_train_entrypoint_and_evaluation_schema_use_production_contract():
    """The caller, schema, and dataloader agree on the new production paths."""
    train_source = (ONEFORMER_ROOT / "scripts" / "train.py").read_text(
        encoding="utf-8"
    )
    evaluate_source = (
        REPO_ROOT / "nvidia_tao_pytorch/config/oneformer/evaluate.py"
    ).read_text(encoding="utf-8")
    dataset_source = (ONEFORMER_ROOT / "dataloader" / "datasets.py").read_text(
        encoding="utf-8"
    )

    assert "model.load_pretrained_weights(cfg.train.pretrained_model)" in train_source
    assert 'valid_options="semantic,panoptic"' in evaluate_source
    assert 'data["panoptic_seg"]' in dataset_source
    assert 'data["panoptic_segments_info"]' in dataset_source
    assert "'panopticapi'" in dataset_source
    assert "requires" in dataset_source
