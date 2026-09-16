# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightning launch behavior stays compatible while strict validation is opt-in."""

import io
import os
from types import SimpleNamespace

import pytest

from nvidia_tao_pytorch.core import entrypoint


@pytest.mark.parametrize(("network", "binary", "visible"), [
    ("nvdinov2", "python", "GPU-a,GPU-b"),
    ("dinov3", "python", "GPU-a,GPU-b"),
    ("rtdetr", "torchrun", "0, 1"),
])
def test_lightning_launch_preserves_legacy_defaults(
    tmp_path, monkeypatch, network, binary, visible
):
    """Lightning tasks use Python and retain the scheduler-owned device mask."""
    spec = tmp_path / "train.yaml"
    spec.write_text("train:\n  num_gpus: 2\n  gpu_ids: [0, 1]\n")
    calls = []

    def popen(argv, **_kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout=io.StringIO(), wait=lambda: None, returncode=0)

    for key in ("WORLD_SIZE", "NODE_RANK", "RANK", "JOB_ID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    # An inherited DEFT flag changes validation, not a model's launch policy.
    monkeypatch.setenv("TAO_STRICT_MULTINODE", "1")
    monkeypatch.setattr(entrypoint.subprocess, "Popen", popen)
    monkeypatch.setattr(entrypoint, "TELEMETRY_AVAILABLE", False)
    with pytest.raises(SystemExit) as result:
        entrypoint.launch({"subtask": "train", "experiment_spec_file": str(spec)}, [],
                          {"train": {"runner_path": "train.py"}}, network)
    assert result.value.code == 0
    assert calls[0][0] == binary
    assert os.environ["CUDA_VISIBLE_DEVICES"] == visible


@pytest.mark.parametrize("strict", [False, True])
def test_strict_validation_is_explicit_opt_in(tmp_path, monkeypatch, strict):
    """Legacy fallback remains unchanged; DINOv3 can fail closed."""
    spec = tmp_path / "train.yaml"
    spec.write_text("train:\n  num_gpus: 1\n  gpu_ids: [0]\n")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("TAO_STRICT_MULTINODE", "1")
    monkeypatch.delenv("JOB_ID", raising=False)
    monkeypatch.setattr(entrypoint, "TELEMETRY_AVAILABLE", False)
    monkeypatch.setattr(entrypoint.torch.cuda, "device_count", lambda: 1)

    def invalid(_logger):
        raise ValueError("bad rendezvous")

    monkeypatch.setattr(entrypoint, "validate_configs", invalid)
    monkeypatch.setattr(entrypoint.subprocess, "Popen", lambda *_a, **_k: SimpleNamespace(
        stdout=io.StringIO(), wait=lambda: None, returncode=0))
    with pytest.raises(RuntimeError if strict else SystemExit) as result:
        entrypoint.launch({"subtask": "train", "experiment_spec_file": str(spec)}, [],
                          {"train": {"runner_path": "train.py"}}, "dinov3" if strict else "nvdinov2",
                          strict_multinode=strict)
    if strict:
        assert "refusing to" in str(result.value)
    else:
        assert result.value.code == 0


def test_native_dinov3_lightning_launch_preserves_sealed_spec(
    tmp_path, monkeypatch
):
    spec = tmp_path / "train.yaml"
    original = (
        "train:\n  num_nodes: 2\n  num_gpus: 1\n  gpu_ids: [0]\n"
    )
    spec.write_text(original, encoding="utf-8")
    for name, value in {
        "TAO_REFINEMENT_LIGHTNING_LAUNCH": "1",
        "TAO_STRICT_MULTINODE": "1",
        "WORLD_SIZE": "2",
        "NUM_GPU_PER_NODE": "1",
        "NODE_RANK": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(entrypoint.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(entrypoint, "TELEMETRY_AVAILABLE", False)
    calls = []
    monkeypatch.setattr(
        entrypoint.subprocess,
        "Popen",
        lambda argv, **_kwargs: (
            calls.append(argv) or
            SimpleNamespace(stdout=io.StringIO(), wait=lambda: None, returncode=0)
        ),
    )
    with pytest.raises(SystemExit) as result:
        entrypoint.launch(
            {"subtask": "train", "experiment_spec_file": str(spec)},
            [],
            {"train": {"runner_path": "train.py"}},
            "dinov3",
            strict_multinode=True,
        )
    assert result.value.code == 0
    assert calls[0][0] == "python"
    assert spec.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("LOCAL_RANK", "0", "node boundary"),
        ("WORLD_SIZE", "3", "WORLD_SIZE differs"),
        ("NUM_GPU_PER_NODE", "2", "NUM_GPU_PER_NODE differs"),
    ],
)
def test_native_dinov3_lightning_launch_rejects_invalid_allocation(
    tmp_path, monkeypatch, name, value, message
):
    spec = tmp_path / "train.yaml"
    spec.write_text(
        "train:\n  num_nodes: 2\n  num_gpus: 1\n  gpu_ids: [0]\n",
        encoding="utf-8",
    )
    for env_name, env_value in {
        "TAO_REFINEMENT_LIGHTNING_LAUNCH": "1",
        "TAO_STRICT_MULTINODE": "1",
        "WORLD_SIZE": "2",
        "NUM_GPU_PER_NODE": "1",
        "NODE_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }.items():
        monkeypatch.setenv(env_name, env_value)
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(entrypoint.torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError, match=message):
        entrypoint.launch(
            {"subtask": "train", "experiment_spec_file": str(spec)},
            [],
            {"train": {"runner_path": "train.py"}},
            "dinov3",
            strict_multinode=True,
        )
