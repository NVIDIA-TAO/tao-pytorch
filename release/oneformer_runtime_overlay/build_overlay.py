#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a deterministic source overlay for the pinned TAO 7.1 SQSH image."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


BASE_REF = "origin/release/7.1.0"
BASE_COMMIT = "99741bc8229617d0d3dd52e30540111d55efd1af"
CONTAINER = {
    "image_reference": (
        "nvcr.io/nvstaging/tao/tao-toolkit-pyt:"
        "7.1.0-rc-245-multiarch"
    ),
    "sha256": "e36640f9ae7a03bc80828cf7de93bd6bdbbb0fecf509a71a243be0ab5b497fc2",
}
PAYLOAD_PATHS = (
    "nvidia_tao_pytorch/config/oneformer/evaluate.py",
    "nvidia_tao_pytorch/cv/oneformer/dataloader/datasets.py",
    "nvidia_tao_pytorch/cv/oneformer/model/pl_oneformer.py",
    "nvidia_tao_pytorch/cv/oneformer/utils/checkpoint.py",
    "nvidia_tao_pytorch/cv/oneformer/utils/metric_reduction.py",
    "nvidia_tao_pytorch/cv/oneformer/utils/panoptic_quality.py",
)


def _run(repo_root, *args):
    return subprocess.run(
        args,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _base_blob(repo_root, path):
    result = subprocess.run(
        ("git", "show", f"{BASE_COMMIT}:{path}"),
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else None


def _ensure_base_commit(repo_root):
    """Fetch the pinned provenance commit when a checkout is shallow."""
    object_name = f"{BASE_COMMIT}^{{commit}}"
    present = subprocess.run(
        ("git", "cat-file", "-e", object_name),
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if present.returncode == 0:
        return

    ref_name = BASE_REF.split("/", 1)[-1]
    fetch = subprocess.run(
        ("git", "fetch", "--no-tags", "origin", ref_name),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if fetch.returncode != 0:
        fetch = subprocess.run(
            ("git", "fetch", "--no-tags", "origin", BASE_COMMIT),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )

    present = subprocess.run(
        ("git", "cat-file", "-e", object_name),
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if present.returncode != 0:
        detail = (fetch.stderr or fetch.stdout).strip()
        raise RuntimeError(
            "Pinned OneFormer overlay base commit "
            f"{BASE_COMMIT} is unavailable. Fetch it from origin before building."
            + (f" Git reported: {detail}" if detail else "")
        )


def _tar_info(name, data, mode=0o644):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    return info


def build_overlay(repo_root, output):
    """Build the deterministic archive and return its manifest."""
    if _run(repo_root, "git", "status", "--porcelain"):
        raise RuntimeError(
            "Refusing to build provenance from a dirty tree. Commit the reviewed "
            "runtime changes first."
        )

    _ensure_base_commit(repo_root)
    source_commit = _run(repo_root, "git", "rev-parse", "HEAD")
    base_commit = _run(repo_root, "git", "rev-parse", BASE_COMMIT)
    files = []
    payload = {}
    for path in sorted(PAYLOAD_PATHS):
        data = (repo_root / path).read_bytes()
        base_data = _base_blob(repo_root, path)
        files.append(
            {
                "path": path,
                "size_bytes": len(data),
                "sha256": _sha256(data),
                "base_sha256": _sha256(base_data) if base_data is not None else None,
            }
        )
        payload[f"payload/{path}"] = data

    manifest = {
        "schema_version": 2,
        "artifact_type": "tao_pytorch_source_overlay",
        "scope": "oneformer_runtime_product_fixes",
        "source": {
            "repository": "tao-pytorch",
            "commit": source_commit,
            "base_ref": BASE_REF,
            "base_commit": base_commit,
            "tree_clean": True,
        },
        "container": CONTAINER,
        "dependencies": [
            {
                "python_module": "panopticapi",
                "required": True,
                "reason": "COCO panoptic ID decoding and task-correct PQ evaluation",
            }
        ],
        "runtime_contract": {
            "evaluation_tasks": ["semantic", "panoptic"],
            "panoptic_primary_metric": "PQ",
            "ddp_reduction": "global_sufficient_statistics_before_metric",
            "status_writer": "global_rank_zero",
            "model_runs_during_build": 0,
            "slurm_jobs_submitted_during_build": 0,
            "base_audit_root": "provided-at-install-time",
            "overlay_output_root": "ephemeral_pythonpath_site_packages",
        },
        "files": files,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    installer = (
        repo_root / "release/oneformer_runtime_overlay/install_overlay.py"
    ).read_bytes()

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w", format=tarfile.GNU_FORMAT) as archive:
        entries = {
            "oneformer-runtime-overlay/MANIFEST.json": manifest_bytes,
            "oneformer-runtime-overlay/install_overlay.py": installer,
            **{
                f"oneformer-runtime-overlay/{name}": data
                for name, data in payload.items()
            },
        }
        for name in sorted(entries):
            data = entries[name]
            mode = 0o755 if name.endswith("/install_overlay.py") else 0o644
            archive.addfile(_tar_info(name, data, mode), io.BytesIO(data))

    digest = _sha256(output.read_bytes())
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n",
        encoding="utf-8",
    )
    return manifest


def main():
    """Command-line entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .tar path.",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    manifest = build_overlay(repo_root, args.output.resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
