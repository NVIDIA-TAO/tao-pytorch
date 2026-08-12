# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static and evaluator-contract tests for Mask2Former task metrics."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TASK_METRICS_SOURCE = (
    REPOSITORY_ROOT
    / "nvidia_tao_pytorch/cv/mask2former/utils/task_metrics.py"
)
TASK_METRICS_SPEC = importlib.util.spec_from_file_location(
    "mask2former_task_metrics_under_test",
    TASK_METRICS_SOURCE,
)
TASK_METRICS = importlib.util.module_from_spec(TASK_METRICS_SPEC)
TASK_METRICS_SPEC.loader.exec_module(TASK_METRICS)
create_instance_evaluator = TASK_METRICS.create_instance_evaluator
finalize_instance_evaluator = TASK_METRICS.finalize_instance_evaluator
instance_metric_names = TASK_METRICS.instance_metric_names
normalize_task_mode = TASK_METRICS.normalize_task_mode
ordered_coco_category_ids = TASK_METRICS.ordered_coco_category_ids
semantic_metric_names = TASK_METRICS.semantic_metric_names
validate_instance_category_contract = (
    TASK_METRICS.validate_instance_category_contract
)

class _Coco:
    """Minimal COCO-like object for pure routing tests."""

    def __init__(self, category_ids=(3, 17)):
        self.dataset = {
            "categories": [
                {"id": category_id, "name": f"class-{category_id}"}
                for category_id in category_ids
            ]
        }


class _FakeCocoEvaluator:
    """Evaluator double recording the global finalization protocol."""

    def __init__(self, coco=None, iou_types=None, eval_class_ids=None):
        self.constructor = {
            "coco": coco,
            "iou_types": iou_types,
            "eval_class_ids": eval_class_ids,
        }
        self.calls = []
        self.coco_eval = {
            "segm": type("_Stats", (), {"stats": np.array([0.42, 0.64])})()
        }

    def synchronize_between_processes(self):
        self.calls.append("synchronize")

    def overall_accumulate(self):
        self.calls.append("accumulate")

    def overall_summarize(self, is_print=False):
        self.calls.append(("summarize", is_print))


@pytest.mark.cv_unit
def test_task_modes_and_metric_names_are_explicit():
    """Instance, semantic, and panoptic metrics cannot be confused."""
    assert normalize_task_mode(" INSTANCE ") == "instance"
    assert instance_metric_names("val") == ("val_mAP", "val_mAP50")
    assert instance_metric_names("test") == ("test_mAP", "test_mAP50")
    assert semantic_metric_names("semantic", "val") == ("mIoU", "all_acc")

    assert semantic_metric_names("panoptic", "test") == ("mIoU", "all_acc")
    with pytest.raises(ValueError, match="must use COCO mask AP"):
        semantic_metric_names("instance", "val")
    with pytest.raises(ValueError, match="Unsupported Mask2Former task mode"):
        normalize_task_mode("classification")


@pytest.mark.cv_unit
def test_coco_category_mapping_is_ordered_validated_and_not_assumed_contiguous():
    """Model indices map back to the exact dataset category IDs."""
    coco = _Coco(category_ids=(1, 5, 90))
    assert ordered_coco_category_ids(coco) == (1, 5, 90)
    assert validate_instance_category_contract(coco, 3) == (1, 5, 90)

    with pytest.raises(ValueError, match="class count"):
        validate_instance_category_contract(coco, 2)
    with pytest.raises(ValueError, match="unique"):
        ordered_coco_category_ids(_Coco(category_ids=(1, 1)))
    with pytest.raises(ValueError, match="integer id"):
        ordered_coco_category_ids(_Coco(category_ids=(True,)))


@pytest.mark.cv_unit
def test_instance_evaluator_uses_segm_and_global_finalize_order():
    """The instance path is COCO-segm-only and globally synchronized."""
    coco = _Coco()
    evaluator = create_instance_evaluator(
        coco,
        num_classes=2,
        evaluator_cls=_FakeCocoEvaluator,
    )
    assert evaluator.constructor == {
        "coco": coco,
        "iou_types": ["segm"],
        "eval_class_ids": [3, 17],
    }

    metrics = finalize_instance_evaluator(
        evaluator,
        split="val",
        is_print=False,
    )
    assert evaluator.calls == [
        "synchronize",
        "accumulate",
        ("summarize", False),
    ]
    assert metrics == {
        "val_mAP": pytest.approx(0.42),
        "val_mAP50": pytest.approx(0.64),
    }


@pytest.mark.cv_unit
def test_instance_evaluator_rejects_missing_or_nonfinite_statistics():
    """Invalid evaluator output cannot be emitted as a valid objective."""
    evaluator = _FakeCocoEvaluator()
    evaluator.coco_eval["segm"].stats = np.array([np.nan, 0.5])
    with pytest.raises(RuntimeError, match="non-finite"):
        finalize_instance_evaluator(evaluator, split="test")

    evaluator = _FakeCocoEvaluator()
    evaluator.coco_eval = {}
    with pytest.raises(RuntimeError, match="did not produce segm"):
        finalize_instance_evaluator(evaluator, split="test")


@pytest.mark.cv_unit
def test_pycocotools_dependency_computes_task_correct_mask_ap():
    """Fail explicitly if the production COCO mask evaluator dependency is absent."""
    height = width = 10
    ground_truth_mask = np.zeros((height, width), dtype=np.uint8)
    ground_truth_mask[2:8, 3:7] = 1
    ground_truth_rle = mask_utils.encode(
        np.asfortranarray(ground_truth_mask[:, :, None])
    )[0]
    ground_truth_rle["counts"] = ground_truth_rle["counts"].decode("ascii")

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": [{"id": 11, "height": height, "width": width}],
        "categories": [{"id": 7, "name": "object"}],
        "annotations": [{
            "id": 1,
            "image_id": 11,
            "category_id": 7,
            "segmentation": ground_truth_rle,
            "area": int(ground_truth_mask.sum()),
            "bbox": [3, 2, 4, 6],
            "iscrowd": 0,
        }],
    }
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes([{
        "image_id": 11,
        "category_id": 7,
        "segmentation": ground_truth_rle,
        "score": 0.99,
    }])

    evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
    evaluator.params.imgIds = [11]
    evaluator.params.catIds = [7]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    assert evaluator.stats[0] == pytest.approx(1.0)
    assert evaluator.stats[1] == pytest.approx(1.0)
