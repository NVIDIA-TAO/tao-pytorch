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
    base_site_packages = tmp_path / "base-site-packages"
    base_target = base_site_packages / "nvidia_tao_pytorch" / "target.py"
    base_target.parent.mkdir(parents=True)
    base_target.write_bytes(base)
    site_packages = tmp_path / "overlay-site-packages"
    target = site_packages / "nvidia_tao_pytorch" / "target.py"
    return overlay, base_site_packages, site_packages, base_target, target


def test_overlay_payload_contains_every_runtime_fix():
    """The source overlay cannot omit a production dependency of the patch."""
    payload = set(builder.PAYLOAD_PATHS)
    assert {
        "nvidia_tao_pytorch/config/oneformer/evaluate.py",
        "nvidia_tao_pytorch/cv/oneformer/dataloader/datasets.py",
        "nvidia_tao_pytorch/cv/oneformer/model/pl_oneformer.py",
        "nvidia_tao_pytorch/cv/oneformer/utils/checkpoint.py",
        "nvidia_tao_pytorch/cv/oneformer/utils/metric_reduction.py",
        "nvidia_tao_pytorch/cv/oneformer/utils/panoptic_quality.py",
    }.issubset(payload)
    assert builder.CONTAINER["sha256"] == (
        "e36640f9ae7a03bc80828cf7de93bd6bdbbb0fecf509a71a243be0ab5b497fc2"
    )


def test_overlay_installer_verifies_base_and_writes_receipt(tmp_path, monkeypatch):
    """A recognized base is atomically replaced and recorded."""
    overlay, base_site, site_packages, base_target, target = _fake_overlay(
        tmp_path
    )
    monkeypatch.setattr(installer.importlib.util, "find_spec", lambda _: object())
    receipt_path = tmp_path / "receipt.json"

    receipt = installer.install(
        overlay,
        base_site,
        site_packages,
        receipt_path,
        dry_run=False,
    )

    assert target.read_bytes() == b"patched"
    assert base_target.read_bytes() == b"base"
    assert receipt["actions"][0]["action"] == "replace_base"
    assert receipt["base_site_packages"] == str(base_site)
    assert receipt["site_packages"] == str(site_packages)
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt


def test_overlay_installer_rejects_unknown_container_base(tmp_path, monkeypatch):
    """An unexpected installed source hash fails before mutation."""
    overlay, base_site, site_packages, base_target, target = _fake_overlay(
        tmp_path
    )
    base_target.write_bytes(b"unknown")
    monkeypatch.setattr(installer.importlib.util, "find_spec", lambda _: object())

    with pytest.raises(RuntimeError, match="base hash mismatch"):
        installer.install(
            overlay,
            base_site,
            site_packages,
            tmp_path / "receipt.json",
            dry_run=False,
        )
    assert base_target.read_bytes() == b"unknown"
    assert not target.exists()


def test_overlay_installer_enforces_panoptic_dependency(tmp_path, monkeypatch):
    """The dependency gate fails before writing any patched file."""
    overlay, base_site, site_packages, base_target, target = _fake_overlay(
        tmp_path
    )
    monkeypatch.setattr(installer.importlib.util, "find_spec", lambda _: None)

    with pytest.raises(RuntimeError, match="does not provide panopticapi"):
        installer.install(
            overlay,
            base_site,
            site_packages,
            tmp_path / "receipt.json",
            dry_run=False,
        )
    assert base_target.read_bytes() == b"base"
    assert not target.exists()


def test_overlay_installer_never_audits_the_empty_output_tree(
    tmp_path, monkeypatch
):
    """The writable overlay cannot impersonate the pinned package root."""
    overlay, _, site_packages, _, target = _fake_overlay(tmp_path)
    monkeypatch.setattr(installer.importlib.util, "find_spec", lambda _: object())

    with pytest.raises(RuntimeError, match="Pinned TAO PyTorch package root"):
        installer.install(
            overlay,
            site_packages,
            site_packages,
            tmp_path / "receipt.json",
            dry_run=False,
        )
    assert not target.exists()


def test_overlay_installer_adds_a_new_file_only_when_base_is_absent(
    tmp_path, monkeypatch
):
    """New runtime helpers are written to the overlay, not the SQSH tree."""
    overlay, base_site, site_packages, base_target, target = _fake_overlay(
        tmp_path
    )
    base_target.unlink()
    manifest_path = overlay / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["base_sha256"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(installer.importlib.util, "find_spec", lambda _: object())

    receipt = installer.install(
        overlay,
        base_site,
        site_packages,
        tmp_path / "receipt.json",
        dry_run=False,
    )

    assert receipt["actions"][0]["action"] == "install_new"
    assert not base_target.exists()
    assert target.read_bytes() == b"patched"
