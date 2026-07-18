# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for rotate_iou_gpu_eval empty-prediction guard."""

import sys
import types
import importlib
from unittest.mock import MagicMock

import numpy as np
import pytest


def _load_rotate_iou_with_mocked_pycuda():
    """Import rotate_iou_pycuda with pycuda stubbed out.

    The module compiles a CUDA kernel at import time via pycuda.compiler.SourceModule,
    so we patch sys.modules before the import to avoid requiring a real GPU/CUDA
    installation for tests that exercise the Python-side early-return path.
    """
    mock_pycuda = types.ModuleType("pycuda")
    mock_autoinit = types.ModuleType("pycuda.autoinit")
    mock_compiler = types.ModuleType("pycuda.compiler")
    mock_driver = types.ModuleType("pycuda.driver")

    # pycuda.autoinit.device.retain_primary_context() -> mock context
    mock_ctx = MagicMock()
    mock_device = MagicMock()
    mock_device.retain_primary_context.return_value = mock_ctx
    mock_autoinit.device = mock_device
    mock_pycuda.autoinit = mock_autoinit

    # pycuda.compiler.SourceModule(...) -> mock module
    mock_compiler.SourceModule = MagicMock(return_value=MagicMock())

    # pycuda.driver.In / Out / mem_alloc_like etc. are not called on the
    # empty-box path, but the module-level `import pycuda.driver as cuda`
    # still needs to resolve.
    mock_driver.In = MagicMock()
    mock_driver.Out = MagicMock()

    patches = {
        "pycuda": mock_pycuda,
        "pycuda.autoinit": mock_autoinit,
        "pycuda.compiler": mock_compiler,
        "pycuda.driver": mock_driver,
    }

    # Remove any previously cached real or mock versions so we get a clean load.
    saved = {k: sys.modules.pop(k) for k in list(patches) if k in sys.modules}
    module_key = (
        "nvidia_tao_pytorch.pointcloud.pointpillars.pcdet.datasets"
        ".kitti.kitti_object_eval_python.rotate_iou_pycuda"
    )
    saved[module_key] = sys.modules.pop(module_key, None)

    try:
        sys.modules.update(patches)
        mod = importlib.import_module(module_key)
        return mod
    finally:
        # Restore original sys.modules state.
        for k in patches:
            sys.modules.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


_rotate_iou_module = _load_rotate_iou_with_mocked_pycuda()
rotate_iou_gpu_eval = _rotate_iou_module.rotate_iou_gpu_eval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_boxes(n, cols=5):
    """Return (n, cols) float32 array of random box parameters."""
    return np.random.rand(n, cols).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.pointcloud_unit
class TestRotateIouGpuEvalEmptyGuard:
    """Regression tests for the N==0 / K==0 early-return added to
    rotate_iou_gpu_eval to prevent a cuMemAlloc failure when the
    detector produces no predictions.
    """

    def test_empty_detections_returns_zeros(self):
        """N=0 (no detections) should return a (0, K) float32 zero array."""
        boxes = _random_boxes(0)
        queries = _random_boxes(4)
        result = rotate_iou_gpu_eval(boxes, queries)
        assert result.shape == (0, 4)
        assert result.dtype == np.float32

    def test_empty_ground_truth_returns_zeros(self):
        """K=0 (no ground-truth) should return a (N, 0) float32 zero array."""
        boxes = _random_boxes(3)
        queries = _random_boxes(0)
        result = rotate_iou_gpu_eval(boxes, queries)
        assert result.shape == (3, 0)
        assert result.dtype == np.float32

    def test_both_empty_returns_zeros(self):
        """N=0 and K=0 should return a (0, 0) float32 zero array."""
        boxes = _random_boxes(0)
        queries = _random_boxes(0)
        result = rotate_iou_gpu_eval(boxes, queries)
        assert result.shape == (0, 0)
        assert result.dtype == np.float32

    def test_empty_result_values_are_zero(self):
        """All values in the returned array must be zero."""
        result = rotate_iou_gpu_eval(_random_boxes(0), _random_boxes(5))
        assert np.all(result == 0.0)
