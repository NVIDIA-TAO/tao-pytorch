# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
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

"""Shared fixtures for DINO AutoML integration tests."""

import os
import sys
from pathlib import Path

import pytest

from .dataset import create_tiny_coco_dataset


REPO_ROOT = Path(__file__).resolve().parents[4]


def _add_tao_automl_source_path():
    """Append an explicit or sibling tao-automl checkout for test imports."""
    candidates = []
    env_src = os.getenv("TAO_AUTOML_SRC")
    if env_src:
        candidates.append(Path(env_src).expanduser())
    candidates.append(REPO_ROOT.parent / "tao-automl" / "src")
    for candidate in candidates:
        if (candidate / "tao_automl").exists():
            sys.path.append(str(candidate))
            return candidate
    return None


_add_tao_automl_source_path()


@pytest.fixture(scope="package", autouse=True)
def _patch_deformable_attention_loader():
    """Patch release-image ops only for this package and restore on teardown."""
    from . import harness

    original_functions_loader = harness.deformable_ops_functions.load_ops
    original_modules_loader = harness.deformable_ops_modules.load_ops
    harness._patch_deformable_attention_loader_for_release_image()
    try:
        yield
    finally:
        harness.deformable_ops_functions.load_ops = original_functions_loader
        harness.deformable_ops_modules.load_ops = original_modules_loader


@pytest.fixture(autouse=True)
def _require_training_device():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available() or os.getenv("TAO_AUTOML_ALLOW_CPU") == "1":
        return
    pytest.skip("DINO AutoML integration tests run real training; set TAO_AUTOML_ALLOW_CPU=1 to run on CPU.")


@pytest.fixture()
def tiny_coco_dataset(tmp_path):
    return create_tiny_coco_dataset(tmp_path)


@pytest.fixture()
def dino_case(tmp_path, tiny_coco_dataset):
    from .harness import DINOAutoMLHarness

    dataset_root, annotation_file = tiny_coco_dataset
    return DINOAutoMLHarness(tmp_path / "automl_workspace", dataset_root, annotation_file)
