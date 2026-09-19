# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require matching DINOv3 manifest schemas in the multi-repository handoff."""

from dataclasses import asdict, fields
import importlib
import os

import pytest

from nvidia_tao_pytorch.config.dinov3 import default_config as runtime


def test_manifest_schema_matches_core():
    """Standalone tests tolerate an old Core release; integrated QA must not."""
    try:
        core = importlib.import_module("nvidia_tao_core.config.dinov3.default_config")
        if "train_manifest" not in core.DINOv3DatasetConfig.__dataclass_fields__:
            raise ImportError("Installed Core predates the DINOv3 manifest schema")
    except ImportError:
        if os.environ.get("TAO_DEFT_REQUIRE_SCHEMA_PARITY") == "1":
            raise
        pytest.skip("Core companion feature not installed; source handoff makes this mandatory")
    for name in ("DINOv3DatasetConfig", "DINOv3TrainExpConfig"):
        left, right = getattr(runtime, name), getattr(core, name)
        assert asdict(left()) == asdict(right())
        assert {field.name: dict(field.metadata) for field in fields(left)} == {
            field.name: dict(field.metadata) for field in fields(right)
        }
