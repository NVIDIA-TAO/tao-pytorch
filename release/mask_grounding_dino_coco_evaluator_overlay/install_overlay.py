#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed installer for the Mask Grounding DINO evaluator overlay."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        with source.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, temporary)
    os.chmod(temporary_path, 0o644)
    os.replace(temporary_path, destination)


def install(overlay_root, base_site_packages, site_packages, receipt_path):
    """Verify the pinned base and install only the sealed replacement file."""
    manifest = json.loads(
        (overlay_root / "MANIFEST.json").read_text(encoding="utf-8")
    )
    base_package_root = base_site_packages / "nvidia_tao_pytorch"
    if not base_package_root.is_dir():
        raise RuntimeError(
            f"Pinned TAO PyTorch package root does not exist: {base_package_root}"
        )

    actions = []
    for record in manifest["files"]:
        relative_path = Path(record["path"])
        source = overlay_root / "payload" / relative_path
        base_destination = base_site_packages / relative_path
        destination = site_packages / relative_path
        if _sha256(source) != record["sha256"]:
            raise RuntimeError(f"Overlay payload hash mismatch: {relative_path}")
        if not base_destination.is_file():
            raise RuntimeError(f"Pinned overlay target is missing: {base_destination}")
        base_sha256 = _sha256(base_destination)
        if base_sha256 != record["base_sha256"]:
            raise RuntimeError(
                "Pinned-container base hash mismatch for "
                f"{base_destination}: expected {record['base_sha256']}, "
                f"got {base_sha256}"
            )
        _install_file(source, destination)
        if _sha256(destination) != record["sha256"]:
            raise RuntimeError(f"Installed overlay hash mismatch: {destination}")
        actions.append(
            {
                "path": record["path"],
                "action": "replace_base",
                "base_sha256": base_sha256,
                "sha256": record["sha256"],
            }
        )

    receipt = {
        "schema_version": 1,
        "overlay_source_commit": manifest["source"]["commit"],
        "container_expected_sha256": manifest["container"]["sha256"],
        "base_site_packages": str(base_site_packages),
        "site_packages": str(site_packages),
        "installed_package_mutated": False,
        "actions": actions,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-site-packages", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = install(
        Path(__file__).resolve().parent,
        args.base_site_packages,
        args.site_packages,
        args.receipt,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
