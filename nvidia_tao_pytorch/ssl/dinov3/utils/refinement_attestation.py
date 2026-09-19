# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared implementation identity for native DINOv3 DEFT leaves."""

import hashlib
import json
import os
from pathlib import Path
from nvidia_tao_pytorch.ssl.dinov3.utils.atomic import atomic_json

CLOSURE_VERSION = "1.0"


def file_sha256(path: str | Path) -> str:
    """Hash one implementation file with the workflow's canonical prefix."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_digest(value: dict) -> str:
    """Hash a JSON object using the controller's canonical encoding."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def implementation_files(subtask: str) -> dict[str, Path]:
    """Return the exact dependency closure sealed by Data Services."""
    if subtask not in {"grit_score", "train"}:
        raise ValueError(f"Unsupported native DINOv3 subtask: {subtask}")
    action_root = Path(__file__).resolve().parents[1]
    package_root = action_root.parents[1]
    inherited_root = action_root.parent / "nvdinov2"
    directories = [
        action_root / "model",
        action_root / "utils",
        inherited_root / "model",
        action_root / "entrypoint",
        action_root / "scripts",
        package_root / "core",
        package_root / "config/dinov3",
        package_root / "config/nvdinov2",
        package_root / "config/common",
        package_root / "config/utils",
    ]
    required = {
        package_root / "__init__.py",
        package_root / "config/__init__.py",
        action_root.parent / "__init__.py",
        action_root / "__init__.py",
        inherited_root / "__init__.py",
        package_root / "config/dinov3/default_config.py",
    }
    if subtask == "grit_score":
        directories.append(action_root / "data_refinement")
        required.update(
            {
                action_root / "dataloader/__init__.py",
                action_root / "dataloader/local_image.py",
                package_root / "config/dinov3/grit_score.py",
            }
        )
    else:
        directories.extend(
            [
                action_root / "dataloader",
                inherited_root / "dataloader",
            ]
        )
        required.update(
            {
                package_root / "core/entrypoint.py",
                package_root / "core/initialize_experiments.py",
            }
        )
    implementations = set(required)
    for directory in directories:
        if not directory.is_dir():
            raise ValueError(
                f"Native DINOv3 implementation closure is incomplete: {directory}"
            )
        implementations.update(directory.rglob("*.py"))
    result = {}
    for path in sorted((item.resolve() for item in implementations), key=str):
        if not path.is_file():
            raise ValueError(
                f"Native DINOv3 implementation closure is incomplete: {path}"
            )
        relative = path.relative_to(package_root).as_posix()
        result[f"native.{relative}"] = path
    return result


def native_attestation(subtask: str) -> dict[str, str]:
    """Compute the entrypoint and full implementation closure identities."""
    action_root = Path(__file__).resolve().parents[1]
    values = {
        name: file_sha256(path) for name, path in implementation_files(subtask).items()
    }
    return {
        "closure_version": CLOSURE_VERSION,
        "entrypoint_sha256": file_sha256(action_root / "scripts" / f"{subtask}.py"),
        "implementation_sha256": canonical_digest(values),
    }


def verify_native_attestation(subtask: str) -> dict[str, str]:
    """Fail closed when the running leaf differs from the controller lock."""
    observed = native_attestation(subtask)
    version = os.environ.get("TAO_REFINEMENT_CLOSURE_VERSION")
    if version and version != CLOSURE_VERSION:
        raise ValueError(f"Native DINOv3 attestation closure version skew: controller={version}, leaf={CLOSURE_VERSION}")
    expected = {
        "entrypoint_sha256": os.environ.get("TAO_REFINEMENT_ENTRYPOINT_SHA256"),
        "implementation_sha256": os.environ.get("TAO_REFINEMENT_IMPLEMENTATION_SHA256"),
    }
    mismatches = {
        name: {"expected": value, "observed": observed[name]}
        for name, value in expected.items()
        if value and value != observed[name]
    }
    if mismatches:
        details = json.dumps(mismatches, sort_keys=True)
        raise ValueError(
            f"Native DINOv3 implementation does not match the controller lock: {details}"
        )
    return observed


def publish_native_attestation(subtask: str, output_dir: str | Path) -> Path | None:
    """Verify every worker and publish one durable rank-zero attestation."""
    observed = verify_native_attestation(subtask)
    if any(
        int(os.environ.get(name) or "0") != 0
        for name in ("RANK", "NODE_RANK", "LOCAL_RANK")
    ):
        return None
    destination = Path(output_dir) / f"{subtask}_implementation_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "1.0", "subtask": subtask}
    payload.update(observed)
    atomic_json(destination, payload)
    return destination
