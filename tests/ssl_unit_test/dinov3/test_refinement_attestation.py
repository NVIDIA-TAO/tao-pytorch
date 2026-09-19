# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native DINOv3 DEFT implementation-attestation tests."""

import json

import pytest

from nvidia_tao_pytorch.ssl.dinov3.utils import refinement_attestation


def test_native_attestation_rejects_controller_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "TAO_REFINEMENT_IMPLEMENTATION_SHA256", "sha256:" + "0" * 64
    )
    with pytest.raises(ValueError, match="controller lock"):
        refinement_attestation.verify_native_attestation("train")


def test_rank_zero_publishes_durable_training_attestation(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("RANK", "NODE_RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    expected = refinement_attestation.native_attestation("train")
    monkeypatch.setenv(
        "TAO_REFINEMENT_ENTRYPOINT_SHA256", expected["entrypoint_sha256"]
    )
    monkeypatch.setenv(
        "TAO_REFINEMENT_IMPLEMENTATION_SHA256",
        expected["implementation_sha256"],
    )
    destination = refinement_attestation.publish_native_attestation(
        "train", tmp_path
    )
    assert destination == tmp_path / "train_implementation_audit.json"
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": "1.0",
        "subtask": "train",
        **expected,
    }


def test_nonzero_worker_does_not_publish_attestation(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert refinement_attestation.publish_native_attestation("train", tmp_path) is None
    assert not list(tmp_path.iterdir())


def test_native_attestation_hashes_tao_runtime_dependencies(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = refinement_attestation.implementation_files("train")
    critical = "native.core/decorators/workflow.py"
    assert critical in files
    copied = tmp_path / "workflow.py"
    copied.write_bytes(files[critical].read_bytes())
    monkeypatch.setattr(
        refinement_attestation,
        "implementation_files",
        lambda _subtask: {critical: copied},
    )
    before = refinement_attestation.native_attestation("train")[
        "implementation_sha256"
    ]
    copied.write_text(copied.read_text(encoding="utf-8") + "\n# changed\n")
    after = refinement_attestation.native_attestation("train")[
        "implementation_sha256"
    ]
    assert after != before


def test_nonzero_worker_verifies_before_skipping_publication(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv(
        "TAO_REFINEMENT_IMPLEMENTATION_SHA256", "sha256:" + "0" * 64
    )
    with pytest.raises(ValueError, match="controller lock"):
        refinement_attestation.publish_native_attestation("train", tmp_path)
    assert not list(tmp_path.iterdir())
