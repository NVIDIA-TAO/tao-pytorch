# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 DEFT launches stay inside Lightning and fail closed at node scope."""

import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from nvidia_tao_pytorch.core import entrypoint


def _successful_process(argv, calls):
    calls.append(argv)
    return SimpleNamespace(stdout=io.StringIO(), wait=lambda: None, returncode=0)


def _launch(
    tmp_path, monkeypatch, *, spec_text, network="dinov3",
    unknown_args=None, disable_telemetry=True,
):
    spec = tmp_path / "train.yaml"
    spec.write_text(spec_text, encoding="utf-8")
    calls = []
    monkeypatch.setenv(
        "TAO_VISIBLE_DEVICES", os.environ.get("TAO_VISIBLE_DEVICES", "")
    )
    monkeypatch.delenv("JOB_ID", raising=False)
    if disable_telemetry:
        monkeypatch.setattr(entrypoint, "TELEMETRY_AVAILABLE", False)
    monkeypatch.setattr(
        entrypoint.subprocess,
        "Popen",
        lambda argv, **_kwargs: _successful_process(argv, calls),
    )
    with pytest.raises(SystemExit) as result:
        entrypoint.launch(
            {"subtask": "train", "experiment_spec_file": str(spec)},
            unknown_args or [],
            {"train": {"runner_path": "train.py"}},
            network,
        )
    return result.value.code, calls, spec


@pytest.mark.parametrize(("network", "binary", "visible"), [
    ("nvdinov2", "python", "GPU-a,GPU-b"),
    ("dinov3", "python", "GPU-a,GPU-b"),
    ("rtdetr", "torchrun", "0, 1"),
])
def test_existing_launch_modes_are_unchanged(
    tmp_path, monkeypatch, network, binary, visible
):
    """Without the DINOv3 DEFT flag, every model retains its launch policy."""
    for key in (
        "TAO_REFINEMENT_LIGHTNING_LAUNCH", "WORLD_SIZE", "NODE_RANK",
        "RANK", "LOCAL_RANK", "JOB_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")

    code, calls, _ = _launch(
        tmp_path,
        monkeypatch,
        spec_text="train:\n  num_gpus: 2\n  gpu_ids: [0, 1]\n",
        network=network,
    )

    assert code == 0
    assert calls[0][0] == binary
    assert os.environ["CUDA_VISIBLE_DEVICES"] == visible


def _set_valid_deft_environment(monkeypatch):
    for name, value in {
        "TAO_REFINEMENT_LIGHTNING_LAUNCH": "1",
        "WORLD_SIZE": "2",
        "NUM_GPU_PER_NODE": "1",
        "NODE_RANK": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }.items():
        monkeypatch.setenv(name, value)
    for name in (
        "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
        "ROLE_RANK", "ROLE_WORLD_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_native_dinov3_deft_uses_python_and_preserves_sealed_spec(
    tmp_path, monkeypatch
):
    """A valid node allocation bypasses torchrun and spec rewriting."""
    original = "train:\n  num_nodes: 2\n  num_gpus: 1\n  gpu_ids: [0]\n"
    _set_valid_deft_environment(monkeypatch)
    monkeypatch.setattr(entrypoint.torch.cuda, "device_count", lambda: 1)

    code, calls, spec = _launch(
        tmp_path, monkeypatch, spec_text=original
    )

    assert code == 0
    assert calls[0][0] == "python"
    assert spec.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(("spec_text", "unknown_args"), [
    (
        "train:\n  num_nodes: 2\n  num_gpus: 1\n  gpu_ids: [0]\n",
        ["train.cuda_blocking=True"],
    ),
    (
        "train:\n  num_nodes: 1\n  num_gpus: 1\n  gpu_ids: [0]\n",
        ["train.num_nodes=2"],
    ),
    (
        "train:\n  num_nodes: 2\n  num_gpus: 1\n  gpu_ids: [0]\n",
        ["train.gpu_ids=[0]"],
    ),
])
def test_native_dinov3_deft_resolves_spec_before_partial_hydra_overrides(
    tmp_path, monkeypatch, spec_text, unknown_args
):
    """Unrelated or partial overrides retain sealed allocation defaults."""
    _set_valid_deft_environment(monkeypatch)
    monkeypatch.setattr(entrypoint.torch.cuda, "device_count", lambda: 1)

    code, calls, spec = _launch(
        tmp_path,
        monkeypatch,
        spec_text=spec_text,
        unknown_args=unknown_args,
    )

    assert code == 0
    assert calls[0][0] == "python"
    assert all(argument in calls[0] for argument in unknown_args)
    assert spec.read_text(encoding="utf-8") == spec_text


@pytest.mark.parametrize("worker_name", [
    "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
    "ROLE_RANK", "ROLE_WORLD_SIZE", "TORCHELASTIC_RUN_ID",
])
def test_native_dinov3_deft_rejects_worker_keys_by_presence(
    monkeypatch, worker_name
):
    """Even an empty worker key makes Lightning select external-launch mode."""
    _set_valid_deft_environment(monkeypatch)
    monkeypatch.setenv(worker_name, "")
    monkeypatch.setattr(entrypoint.torch.cuda, "device_count", lambda: 1)

    with pytest.raises(ValueError, match="node boundary"):
        entrypoint._validate_native_dinov3_deft_launch(
            num_nodes=2,
            num_gpus=1,
            gpu_ids=[0],
        )


def test_invalid_hydra_allocation_reports_user_error_telemetry(
    tmp_path, monkeypatch
):
    _set_valid_deft_environment(monkeypatch)
    monkeypatch.setattr(entrypoint.torch.cuda, "device_count", lambda: 1)
    telemetry = []
    monkeypatch.setattr(entrypoint, "TELEMETRY_AVAILABLE", True)
    monkeypatch.setattr(entrypoint, "get_device_details", lambda: [])
    monkeypatch.setattr(
        entrypoint,
        "send_telemetry_data",
        lambda *_args, **kwargs: telemetry.append(kwargs),
    )

    code, calls, _ = _launch(
        tmp_path,
        monkeypatch,
        spec_text="train:\n  num_nodes: 2\n  num_gpus: 1\n  gpu_ids: [0]\n",
        unknown_args=["train.num_nodes=bad"],
        disable_telemetry=False,
    )

    assert code == 1
    assert calls == []
    assert telemetry == [{
        "num_gpus": 1,
        "time_lapsed": 0,
        "pass_status": False,
        "user_error": True,
    }]


@pytest.mark.parametrize(("spec_text", "message"), [
    ("train: [\n", "invalid YAML"),
    ("- train\n", "root must be a mapping"),
    ("train:\n  - invalid\n", "train must be a mapping"),
])
def test_malformed_deft_spec_uses_normal_failure_path(
    tmp_path, monkeypatch, spec_text, message
):
    """Malformed YAML and schema shapes fail as parameter errors."""
    _set_valid_deft_environment(monkeypatch)
    monkeypatch.setattr(entrypoint.torch.cuda, "device_count", lambda: 1)
    telemetry = []
    monkeypatch.setattr(entrypoint, "TELEMETRY_AVAILABLE", True)
    monkeypatch.setattr(entrypoint, "get_device_details", lambda: [])
    monkeypatch.setattr(
        entrypoint,
        "send_telemetry_data",
        lambda *_args, **kwargs: telemetry.append(kwargs),
    )

    code, calls, _ = _launch(
        tmp_path,
        monkeypatch,
        spec_text=spec_text,
        disable_telemetry=False,
    )

    assert code == 1
    assert calls == []
    assert telemetry[0]["user_error"] is True
    with pytest.raises(ValueError, match=message):
        entrypoint._resolve_native_dinov3_deft_allocation(
            {"experiment_spec_file": str(tmp_path / "train.yaml")}, []
        )


def test_deft_lightning_creates_two_ranked_local_workers(tmp_path):
    """The real DEFT node contract lets Lightning create both local workers."""
    helper = Path(__file__).with_name("lightning_two_worker.py")
    environment = os.environ.copy()
    for name in (
        "WORLD_SIZE", "NUM_GPU_PER_NODE", "NODE_RANK", "MASTER_ADDR",
        "MASTER_PORT", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
        "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
    ):
        environment.pop(name, None)
    source_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH")) if value
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        master_port = listener.getsockname()[1]
    environment.update({
        "TAO_REFINEMENT_LIGHTNING_LAUNCH": "1",
        "WORLD_SIZE": "1",
        "NUM_GPU_PER_NODE": "2",
        "NODE_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(master_port),
    })
    result = subprocess.run(
        [sys.executable, str(helper), str(tmp_path), "1", "2"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    records = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    assert [record["global_rank"] for record in records] == [0, 1]
    assert [record["local_rank"] for record in records] == [0, 1]
    assert [record["environment_local_rank"] for record in records] == [0, 1]
    assert {record["node_rank"] for record in records} == {0}
    assert {record["world_size"] for record in records} == {2}
    assert {record["strategy"] for record in records} == {"DDPStrategy"}
    assert len({record["pid"] for record in records}) == 2


def test_deft_lightning_joins_two_real_nodes(tmp_path):
    """Two node entrypoints form one Lightning-owned two-rank process group."""
    helper = Path(__file__).with_name("lightning_two_worker.py")
    source_root = str(Path(__file__).resolve().parents[2])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        master_port = listener.getsockname()[1]
    processes = []
    for node_rank in range(2):
        environment = os.environ.copy()
        for name in (
            "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
            "ROLE_RANK", "ROLE_WORLD_SIZE",
        ):
            environment.pop(name, None)
        environment.update({
            "PYTHONPATH": os.pathsep.join(
                value for value in (source_root, environment.get("PYTHONPATH"))
                if value
            ),
            "TAO_REFINEMENT_LIGHTNING_LAUNCH": "1",
            "WORLD_SIZE": "2",
            "NUM_GPU_PER_NODE": "1",
            "NODE_RANK": str(node_rank),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
        })
        processes.append(
            subprocess.Popen(
                [sys.executable, str(helper), str(tmp_path), "2", "1"],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    outputs = []
    for process in processes:
        output, _ = process.communicate(timeout=120)
        outputs.append(output)
    assert [process.returncode for process in processes] == [0, 0], "\n".join(outputs)
    records = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    assert [record["global_rank"] for record in records] == [0, 1]
    assert [record["node_rank"] for record in records] == [0, 1]
    assert {record["world_size"] for record in records} == {2}
    assert len({record["pid"] for record in records}) == 2


@pytest.mark.parametrize(("name", "value", "message"), [
    ("WORLD_SIZE", "", "missing allocation fields"),
    ("WORLD_SIZE", "3", "WORLD_SIZE differs"),
    ("NUM_GPU_PER_NODE", "2", "NUM_GPU_PER_NODE differs"),
    ("NODE_RANK", "2", "NODE_RANK is out of range"),
    ("MASTER_PORT", "70000", "MASTER_PORT is invalid"),
    ("LOCAL_RANK", "", "node boundary"),
])
def test_native_dinov3_deft_invalid_allocation_uses_normal_failure_path(
    tmp_path, monkeypatch, name, value, message
):
    """Allocation errors do not launch a child or escape as raw tracebacks."""
    _set_valid_deft_environment(monkeypatch)
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(entrypoint.torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match=message):
        entrypoint._validate_native_dinov3_deft_launch(
            num_nodes=2,
            num_gpus=1,
            gpu_ids=[0],
        )

    warnings = []
    monkeypatch.setattr(
        entrypoint.logging,
        "warning",
        lambda text: warnings.append(str(text)),
    )

    code, calls, _ = _launch(
        tmp_path,
        monkeypatch,
        spec_text="train:\n  num_nodes: 2\n  num_gpus: 1\n  gpu_ids: [0]\n",
    )

    assert code == 1
    assert calls == []
    assert any("Execution status: FAIL" in warning for warning in warnings)


@pytest.mark.parametrize(("spec_text", "visible_gpus", "message"), [
    (
        "train:\n  num_nodes: 2\n  num_gpus: 2\n  gpu_ids: [0]\n",
        2,
        "length",
    ),
    (
        "train:\n  num_nodes: 2\n  num_gpus: 1\n  gpu_ids: [1]\n",
        2,
        "contiguous",
    ),
    (
        "train:\n  num_nodes: 2\n  num_gpus: 1\n  gpu_ids: [0]\n",
        0,
        "visible",
    ),
])
def test_native_dinov3_deft_rejects_invalid_local_gpu_contract(
    tmp_path, monkeypatch, spec_text, visible_gpus, message
):
    _set_valid_deft_environment(monkeypatch)
    monkeypatch.setattr(
        entrypoint.torch.cuda, "device_count", lambda: visible_gpus
    )
    if "num_gpus: 2" in spec_text:
        monkeypatch.setenv("NUM_GPU_PER_NODE", "2")
    with pytest.raises(ValueError, match=message):
        entrypoint._validate_native_dinov3_deft_launch(
            num_nodes=2,
            num_gpus=2 if "num_gpus: 2" in spec_text else 1,
            gpu_ids=[0] if "gpu_ids: [0]" in spec_text else [1],
        )


    code, calls, _ = _launch(tmp_path, monkeypatch, spec_text=spec_text)

    assert code == 1
    assert calls == []
