# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-smoke tests for Sparse4D's spatialai_data_utils dependency.

These catch dependency API drift (e.g. module relocations in spatialai_data_utils, like
the 1.0.0 -> 2.0.0 restructure in TAO-2183 / FF-5) in the fast cv_unit stage instead of
the slow functional suite. The repo-wide tests/test_imports.py only validates *internal*
nvidia_tao_pytorch imports, so external-dependency relocations slipped through to the
Jenkins functional run; this test closes that gap for the spatialai symbols Sparse4D uses
(including the ones imported lazily inside methods, which a plain module import would not
exercise).
"""

import importlib
from platform import machine

import pytest

pytestmark = pytest.mark.skipif(
    ("aarch64" in machine().lower()) or ("arm" in machine().lower()),
    reason="Sparse4D test suite is skipped on ARM (see sibling tests).",
)

# (module, attribute) for every spatialai_data_utils symbol the Sparse4D dataloader imports
# — both the module-level imports and the ones imported lazily inside eval/tracking methods.
SPATIALAI_SYMBOLS = [
    ("spatialai_data_utils.constants", "FPS"),
    ("spatialai_data_utils.eval.detection.data_classes", "DetectionConfig"),
    ("spatialai_data_utils.eval.tracking.data_classes", "TrackingConfig"),
    ("spatialai_data_utils.visualization", "COLOR_MAP"),
    ("spatialai_data_utils.core.boxes.aicity_box", "AICityBox"),
    ("spatialai_data_utils.converters.nusc_results_to_nvschema", "convert_sparse4d_to_nvschema"),
    ("spatialai_data_utils.eval.detection.evaluate", "AIC24DetEval"),
    ("spatialai_data_utils.eval.tracking.aic24_eval", "AIC24TrackEval"),
    ("spatialai_data_utils.eval.tracking.hota.hota_eval", "evaluate_hota"),
    ("spatialai_data_utils.eval.tracking.hota.hota_eval", "HOTA_FIELDS"),
]


@pytest.mark.cv_unit
@pytest.mark.parametrize("module_path, attr", SPATIALAI_SYMBOLS)
def test_spatialai_symbol_importable(module_path, attr):
    """Each spatialai_data_utils symbol Sparse4D depends on must import and exist."""
    module = importlib.import_module(module_path)
    assert hasattr(module, attr), (
        f"{module_path}.{attr} is missing — spatialai_data_utils API drift. "
        f"Update nvidia_tao_pytorch/cv/sparse4d/dataloader/dataset.py imports to match."
    )


@pytest.mark.cv_unit
def test_sparse4d_dataset_module_imports():
    """The Sparse4D dataloader module imports cleanly (numpy/cv2/spatialai all resolve)."""
    importlib.import_module("nvidia_tao_pytorch.cv.sparse4d.dataloader.dataset")
