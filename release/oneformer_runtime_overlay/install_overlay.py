#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed installer for the OneFormer TAO PyTorch source overlay."""

import argparse
import hashlib
import importlib.util
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
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        with source.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, temporary)
    os.chmod(temporary_path, 0o644)
    os.replace(temporary_path, destination)


def install(
    overlay_root,
    base_site_packages,
    site_packages,
    receipt_path,
    dry_run=False,
):
    """Audit the pinned package root and write to an ephemeral overlay root.

    ``base_site_packages`` is the immutable package tree supplied by the
    pinned SQSH.  ``site_packages`` is a separate writable tree placed first
    on ``PYTHONPATH``.  Keeping these paths distinct prevents an empty output
    tree from being mistaken for the pinned base during hash validation.
    """
    manifest = json.loads(
        (overlay_root / "MANIFEST.json").read_text(encoding="utf-8")
    )
    if importlib.util.find_spec("panopticapi") is None:
        raise RuntimeError(
            "The pinned TAO runtime does not provide panopticapi. Install the "
            "declared dependency before applying this OneFormer overlay."
        )
    base_package_root = base_site_packages / "nvidia_tao_pytorch"
    if not base_package_root.is_dir():
        raise RuntimeError(
            "Pinned TAO PyTorch package root does not exist: "
            f"{base_package_root}"
        )
    package_root = site_packages / "nvidia_tao_pytorch"
    package_root.mkdir(parents=True, exist_ok=True)

    actions = []
    for record in manifest["files"]:
        relative_path = Path(record["path"])
        source = overlay_root / "payload" / relative_path
        base_destination = base_site_packages / relative_path
        destination = site_packages / relative_path
        if _sha256(source) != record["sha256"]:
            raise RuntimeError(f"Overlay payload hash mismatch: {relative_path}")

        base_sha = (
            _sha256(base_destination) if base_destination.is_file() else None
        )
        if record["base_sha256"] is None:
            if base_destination.exists():
                raise RuntimeError(
                    "A new overlay target already exists in the pinned base: "
                    f"{base_destination}"
                )
            base_action = "install_new"
        elif base_sha != record["base_sha256"]:
            raise RuntimeError(
                "Pinned-container base hash mismatch for "
                f"{base_destination}: expected {record['base_sha256']}, "
                f"got {base_sha}."
            )
        else:
            base_action = "replace_base"

        output_sha = _sha256(destination) if destination.is_file() else None
        action = (
            "already_installed"
            if output_sha == record["sha256"]
            else base_action
        )

        if not dry_run and action != "already_installed":
            _install_file(source, destination)
            if _sha256(destination) != record["sha256"]:
                raise RuntimeError(
                    f"Installed overlay hash mismatch: {destination}"
                )
        actions.append(
            {
                "path": record["path"],
                "action": action,
                "base_sha256": base_sha,
                "sha256": record["sha256"],
            }
        )

    receipt = {
        "schema_version": 2,
        "overlay_source_commit": manifest["source"]["commit"],
        "container_expected_sha256": manifest["container"]["sha256"],
        "base_site_packages": str(base_site_packages),
        "site_packages": str(site_packages),
        "dry_run": dry_run,
        "actions": actions,
    }
    if not dry_run:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return receipt


def main():
    """Command-line entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-site-packages",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--site-packages",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("/tmp/oneformer-runtime-overlay-receipt.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    overlay_root = Path(__file__).resolve().parent
    receipt = install(
        overlay_root,
        args.base_site_packages,
        args.site_packages,
        args.receipt,
        dry_run=args.dry_run,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
