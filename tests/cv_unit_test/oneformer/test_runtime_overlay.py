# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data-only tests for the OneFormer pinned-SQSH source overlay."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
OVERLAY_SOURCE = REPO_ROOT / "release" / "oneformer_runtime_overlay"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_module(
    "oneformer_overlay_builder_test_target",
    OVERLAY_SOURCE / "build_overlay.py",
)
installer = _load_module(
    "oneformer_overlay_installer_test_target",
    OVERLAY_SOURCE / "install_overlay.py",
)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _fake_overlay(tmp_path, *, base=b"base", patched=b"patched"):
    overlay = tmp_path / "overlay"
    payload = overlay / "payload" / "nvidia_tao_pytorch" / "target.py"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(patched)
    manifest = {
        "source": {"commit": "f" * 40},
        "container": {"sha256": "e" * 64},
        "files": [
            {
                "path": "nvidia_tao_pytorch/target.py",
                "base_sha256": _sha256(base),
                "sha256": _sha256(patched),
            }
        ],
    }
    (overlay / "MANIFEST.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    site_packages = tmp_path / "site-packages"
    target = site_packages / "nvidia_tao_pytorch" / "target.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(base)
    return overlay, site_packages, target


def test_overlay_payload_contains_every_runtime_fix():
    """The source overlay cannot omit a production dependency of the patch."""
    payload = set(builder.PAYLOAD_PATHS)
    assert {
        "nvidia_tao_pytorch/config/oneformer/evaluate.py",
        "nvidia_tao_pytorch/cv/oneformer/dataloader/datasets.py",
        "nvidia_tao_pytorch/cv/oneformer/model/pl_oneformer.py",
        "nvidia_tao_pytorch/cv/oneformer/scripts/train.py",
        "nvidia_tao_pytorch/cv/oneformer/utils/checkpoint.py",
        "nvidia_tao_pytorch/cv/oneformer/utils/metric_reduction.py",
        "nvidia_tao_pytorch/cv/oneformer/utils/panoptic_quality.py",
    }.issubset(payload)
    assert builder.CONTAINER["sha256"] == (
        "e36640f9ae7a03bc80828cf7de93bd6bdbbb0fecf509a71a243be0ab5b497fc2"
    )


def test_overlay_installer_verifies_base_and_writes_receipt(tmp_path, monkeypatch):
    """A recognized base is atomically replaced and recorded."""
    overlay, site_packages, target = _fake_overlay(tmp_path)
    monkeypatch.setattr(installer.importlib.util, "find_spec", lambda _: object())
    receipt_path = tmp_path / "receipt.json"

    receipt = installer.install(
        overlay, site_packages, receipt_path, dry_run=False
    )

    assert target.read_bytes() == b"patched"
    assert receipt["actions"][0]["action"] == "replace_base"
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt


def test_overlay_installer_rejects_unknown_container_base(tmp_path, monkeypatch):
    """An unexpected installed source hash fails before mutation."""
    overlay, site_packages, target = _fake_overlay(tmp_path)
    target.write_bytes(b"unknown")
    monkeypatch.setattr(installer.importlib.util, "find_spec", lambda _: object())

    with pytest.raises(RuntimeError, match="base hash mismatch"):
        installer.install(
            overlay, site_packages, tmp_path / "receipt.json", dry_run=False
        )
    assert target.read_bytes() == b"unknown"


def test_overlay_installer_enforces_panoptic_dependency(tmp_path, monkeypatch):
    """The dependency gate fails before writing any patched file."""
    overlay, site_packages, target = _fake_overlay(tmp_path)
    monkeypatch.setattr(installer.importlib.util, "find_spec", lambda _: None)

    with pytest.raises(RuntimeError, match="does not provide panopticapi"):
        installer.install(
            overlay, site_packages, tmp_path / "receipt.json", dry_run=False
        )
    assert target.read_bytes() == b"base"
