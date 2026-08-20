# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 convert entrypoint unit tests."""

from types import SimpleNamespace
from unittest import mock

import pytest

from nvidia_tao_pytorch.ssl.dinov3.scripts import convert


@pytest.mark.ssl_unit
def test_convert_logs_loadable_backbone_registry_name(monkeypatch, tmp_path):
    """The downstream load hint must use a real ``BACKBONE_REGISTRY`` key."""
    checkpoint = tmp_path / "teacher.pth"
    checkpoint.touch()
    output = tmp_path / "backbone.safetensors"
    cfg = SimpleNamespace(
        convert=SimpleNamespace(
            checkpoint=str(checkpoint),
            source="teacher",
            output_path=str(output),
            results_dir=None,
            validate=True,
        ),
        model=SimpleNamespace(
            backbone=SimpleNamespace(student_type="vit_b", teacher_type="vit_b")
        ),
        results_dir=str(tmp_path),
    )
    log = mock.Mock()
    monkeypatch.setattr(convert, "convert_ssl_to_timm", mock.Mock())
    monkeypatch.setattr(convert.logging, "info", log)

    convert.run_convert(cfg)

    assert "BACKBONE_REGISTRY.get('dinov3_vitb16')" in log.call_args_list[-1].args[0]
