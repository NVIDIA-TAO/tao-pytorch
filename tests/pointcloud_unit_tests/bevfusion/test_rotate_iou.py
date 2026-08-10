# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the BEVFusion CPU rotated-IoU fallback."""

import numpy as np
import pytest

from nvidia_tao_pytorch.cv.bevfusion.evaluation.functional import rotate_iou


pytestmark = [pytest.mark.cv_unit, pytest.mark.bevfusion]


def test_default_backend_uses_cpu_without_initializing_cuda(monkeypatch):
    monkeypatch.delenv("BEVFUSION_ROTATE_IOU_BACKEND", raising=False)

    assert rotate_iou._use_cpu_iou() is True


def test_cpu_fallback_computes_expected_iou(monkeypatch):
    monkeypatch.setenv("BEVFUSION_ROTATE_IOU_BACKEND", "cpu")
    boxes = np.array([[0.0, 0.0, 2.0, 2.0, 0.0]], dtype=np.float32)
    queries = np.array(
        [
            [0.0, 0.0, 2.0, 2.0, 0.0],
            [1.0, 0.0, 2.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = rotate_iou.rotate_iou_gpu_eval(boxes, queries)

    np.testing.assert_allclose(result, [[1.0, 1.0 / 3.0]], rtol=1e-6)


def test_empty_input_does_not_initialize_backend(monkeypatch):
    monkeypatch.setattr(
        rotate_iou,
        "_use_cpu_iou",
        lambda: (_ for _ in ()).throw(AssertionError("backend should not be queried")),
    )

    result = rotate_iou.rotate_iou_gpu_eval(
        np.empty((0, 5), dtype=np.float32),
        np.ones((2, 5), dtype=np.float32),
    )

    assert result.shape == (0, 2)
