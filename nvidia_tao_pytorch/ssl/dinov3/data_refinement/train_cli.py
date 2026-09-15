# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train DINOv3 for a fixed number of passes over a refinement manifest."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import socket
import subprocess
import sys
import tempfile
import time
from typing import Iterator

import pyarrow.parquet as pq
import yaml


def _atomic_write_text(path: Path, value: str) -> None:
    """Publish text without exposing a partially written shared artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary_path.unlink(missing_ok=True)


@contextmanager
def _launch_spec(spec_path: Path) -> Iterator[Path]:
    """Give every launcher a private spec that TAO may rewrite."""
    node_rank = os.environ.get("NODE_RANK", os.environ.get("RANK", "unknown"))
    temporary_root_value = os.environ.get("TAO_REFINEMENT_TMPDIR")
    if temporary_root_value is None:
        temporary_root_value = os.environ.get("TMPDIR")
    temporary_root = Path(temporary_root_value or tempfile.gettempdir())
    temporary_root.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(
        (str(spec_path.resolve()) + _sha256(spec_path)).encode("utf-8")
    ).hexdigest()[:16]
    local_path = temporary_root / (
        f"tao-dinov3-refinement-node-{node_rank}-{identity}.yaml"
    )
    try:
        _atomic_write_text(local_path, spec_path.read_text(encoding="utf-8"))
        yield local_path
    finally:
        local_path.unlink(missing_ok=True)


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _configured_resources(
    base_spec: str | Path,
    *,
    num_nodes: int | None,
    gpus_per_node: int | None,
) -> tuple[int, int]:
    spec = yaml.safe_load(Path(base_spec).read_text(encoding="utf-8")) or {}
    training = spec.get("train", {})
    nodes = int(num_nodes if num_nodes is not None else training.get("num_nodes", 1))
    gpus = int(
        gpus_per_node
        if gpus_per_node is not None
        else training.get("num_gpus", 1)
    )
    if nodes <= 0 or gpus <= 0:
        raise ValueError("Resolved node and GPU counts must be positive")
    return nodes, gpus


def _distributed_launch_environment(
    *, num_nodes: int, gpus_per_node: int
) -> tuple[int, str | None, dict[str, str]]:
    """Resolve an exact TAO node-level rendezvous or fail before launch."""
    environment = os.environ.copy()
    process_launcher_variables = (
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
    )
    conflicting = [
        name for name in process_launcher_variables if os.environ.get(name)
    ]
    if conflicting:
        raise ValueError(
            "DINOv3 refinement requires exactly one adapter process per node; "
            f"remove inherited process-launcher variables: {conflicting}"
        )
    for name in (*process_launcher_variables, "RANK"):
        environment.pop(name, None)
    environment["TAO_REFINEMENT_LIGHTNING_LAUNCH"] = "1"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        visible_devices = [item for item in visible.split(",") if item.strip()]
        if len(visible_devices) != gpus_per_node:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES must exactly match the committed "
                f"allocation: expected {gpus_per_node}, found {len(visible_devices)}"
            )
    if num_nodes == 1:
        for name in (
            "WORLD_SIZE",
            "NODE_RANK",
            "RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TAO_STRICT_MULTINODE",
        ):
            environment.pop(name, None)
        environment["NUM_GPU_PER_NODE"] = str(gpus_per_node)
        return 0, None, environment

    required = {
        name: os.environ.get(name)
        for name in (
            "WORLD_SIZE",
            "NODE_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
            "NUM_GPU_PER_NODE",
        )
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "Multi-node DINOv3 refinement requires the TAO node-level "
            f"rendezvous variables: missing {missing}"
        )
    environment_nodes = int(required["WORLD_SIZE"])
    node_rank = int(required["NODE_RANK"])
    environment_gpus = int(required["NUM_GPU_PER_NODE"])
    master_port = int(required["MASTER_PORT"])
    if environment_nodes != num_nodes:
        raise ValueError(
            "TAO WORLD_SIZE is the node count and must match the committed "
            f"allocation: expected {num_nodes}, found {environment_nodes}"
        )
    if environment_gpus != gpus_per_node:
        raise ValueError(
            "NUM_GPU_PER_NODE must match the committed allocation: "
            f"expected {gpus_per_node}, found {environment_gpus}"
        )
    if not 0 <= node_rank < num_nodes:
        raise ValueError(
            f"NODE_RANK must be in [0, {num_nodes}), found {node_rank}"
        )
    if not 1024 <= master_port <= 65535:
        raise ValueError(f"MASTER_PORT must be in [1024, 65535], found {master_port}")
    master_addr = str(required["MASTER_ADDR"])
    try:
        socket.gethostbyname(master_addr)
    except socket.gaierror as error:
        raise ValueError(f"MASTER_ADDR does not resolve: {master_addr}") from error

    launch_id = os.environ.get("TAO_REFINEMENT_LAUNCH_ID")
    if not launch_id:
        raise ValueError(
            "Multi-node runners must set an attempt-unique "
            "TAO_REFINEMENT_LAUNCH_ID; a scheduler job ID alone is not safe "
            "across requeues"
        )
    environment.update(
        {
            "WORLD_SIZE": str(num_nodes),
            "NODE_RANK": str(node_rank),
            "NUM_GPU_PER_NODE": str(gpus_per_node),
            "TAO_REFINEMENT_LAUNCH_ID": launch_id,
            "TAO_STRICT_MULTINODE": "1",
        }
    )
    return node_rank, launch_id, environment


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build_training_spec(
    *,
    base_spec: str | Path,
    manifest: str | Path,
    parent_checkpoint: str | Path,
    passes: int,
    output_dir: str | Path,
    num_nodes: int | None = None,
    gpus_per_node: int | None = None,
    checkpoint_policy: str = "caller_selected",
    lr_scaling_rule: str = "sqrt_global_batch",
    lr_reference_world_size: int | None = None,
) -> tuple[dict, dict]:
    """Derive a normal TAO DINOv3 spec without mutating the user's base spec."""
    if passes <= 0:
        raise ValueError("passes must be positive")
    base_spec_path = Path(base_spec).expanduser().resolve()
    manifest_path = Path(manifest).resolve()
    parent_path = Path(parent_checkpoint).expanduser().absolute()
    resolved_parent_path = parent_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest does not exist: {manifest_path}")
    if not parent_path.exists():
        raise FileNotFoundError(f"Parent checkpoint does not exist: {parent_path}")
    spec = yaml.safe_load(base_spec_path.read_text(encoding="utf-8")) or {}
    if spec.get("model", {}).get("distill", {}).get("enable", False):
        raise ValueError(
            "DEFT refinement requires model.distill.enable=false: its output "
            "contract is the trained EMA teacher, not the frozen distillation "
            "teacher or student EMA"
        )
    spec["results_dir"] = str(Path(output_dir).resolve())
    dataset = spec.setdefault("dataset", {})
    dataset["train_manifest"] = str(manifest_path)
    dataset.setdefault("train_dataset", {}).setdefault("images_dir", "/")
    training = spec.setdefault("train", {})
    if num_nodes is not None:
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        training["num_nodes"] = num_nodes
    if gpus_per_node is not None:
        if gpus_per_node <= 0:
            raise ValueError("gpus_per_node must be positive")
        training["num_gpus"] = gpus_per_node
    training["results_dir"] = str(Path(output_dir).resolve())
    training["num_epochs"] = passes
    training["pretrained_model_path"] = str(parent_path)
    requested_checkpoint_interval = int(training.get("checkpoint_interval", 1))
    if requested_checkpoint_interval <= 0:
        raise ValueError("train.checkpoint_interval must be positive")
    requested_checkpoint_unit = str(
        training.get("checkpoint_interval_unit", "epoch")
    ).lower()
    if requested_checkpoint_unit not in {"epoch", "step"}:
        raise ValueError("train.checkpoint_interval_unit must be 'epoch' or 'step'")
    parquet = pq.ParquetFile(manifest_path)
    missing = {"sample_id", "storage_type", "path"}.difference(
        parquet.schema_arrow.names
    )
    if missing:
        raise ValueError(
            f"Training manifest is missing canonical columns: {sorted(missing)}"
        )
    resume_checkpoint = _latest_resume_checkpoint(Path(output_dir).resolve())
    training["resume_training_checkpoint_path"] = (
        None if resume_checkpoint is None else str(resume_checkpoint)
    )
    rows = int(parquet.metadata.num_rows)
    batch_size = int(dataset["batch_size"])
    world_size = int(training.get("num_nodes", 1)) * int(
        training.get("num_gpus", 1)
    )
    if lr_scaling_rule != "sqrt_global_batch":
        raise ValueError("lr_scaling_rule must be sqrt_global_batch")
    reference_world_size = int(lr_reference_world_size or world_size)
    if reference_world_size <= 0:
        raise ValueError("lr_reference_world_size must be positive")
    lr_multiplier = math.sqrt(world_size / reference_world_size)
    samples_per_rank = math.ceil(rows / world_size)
    steps_per_pass = math.ceil(samples_per_rank / batch_size)
    total_optimizer_steps = steps_per_pass * passes
    if requested_checkpoint_unit == "epoch":
        effective_checkpoint_interval = math.gcd(
            requested_checkpoint_interval, passes
        )
    else:
        effective_checkpoint_interval = min(
            requested_checkpoint_interval, total_optimizer_steps
        )
    training["checkpoint_interval_unit"] = requested_checkpoint_unit
    training["checkpoint_interval"] = effective_checkpoint_interval
    warmup_steps = max(1, round(total_optimizer_steps * 0.10))
    schedulers = training.setdefault("schedulers", {})
    for name in (
        "learning_rate",
        "last_layer_learning_rate",
        "weight_decay",
        "momentum",
        "teacher_temperature",
    ):
        scheduler = schedulers.setdefault(name, {})
        scheduler["max_decay_steps"] = total_optimizer_steps
        if name in {
            "learning_rate",
            "last_layer_learning_rate",
            "teacher_temperature",
        }:
            scheduler["warm_up_steps"] = warmup_steps
    lr_scaling_mode = "not_configured"
    for name in ("learning_rate", "last_layer_learning_rate"):
        scheduler = schedulers[name]
        value = scheduler.get("val_base")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            scheduler["val_base"] = float(value) * lr_multiplier
            lr_scaling_mode = "numeric_sqrt_scaled"
        elif isinstance(value, str):
            required_tokens = (
                "${dataset.batch_size}",
                "${train.num_gpus}",
                "${train.num_nodes}",
            )
            if not all(token in value for token in required_tokens):
                raise ValueError(
                    f"train.schedulers.{name}.val_base must be numeric or "
                    "depend on global batch when node scaling is enabled"
                )
            lr_scaling_mode = "base_spec_global_batch_formula"
        elif lr_multiplier != 1.0:
            raise ValueError(
                f"train.schedulers.{name}.val_base must be explicitly numeric "
                "or global-batch-aware when node scaling changes world size"
            )
    last_layer_scheduler = schedulers["last_layer_learning_rate"]
    requested_freeze_steps = int(last_layer_scheduler.get("freeze_steps", 0))
    if requested_freeze_steps < 0:
        raise ValueError(
            "train.schedulers.last_layer_learning_rate.freeze_steps must be nonnegative"
        )
    effective_freeze_steps = min(
        requested_freeze_steps,
        max(0, round(total_optimizer_steps * 0.10)),
    )
    last_layer_scheduler["freeze_steps"] = effective_freeze_steps
    dinov3_entrypoint = (
        Path(__file__).resolve().parents[1] / "entrypoint" / "dinov3.py"
    )
    python_executable = Path(sys.executable).resolve()
    contract = {
        "schema_version": "1.0",
        "base_spec": str(base_spec_path),
        "base_spec_sha256": _sha256(base_spec_path),
        "manifest": str(manifest_path),
        "manifest_rows": rows,
        "requested_data_passes": passes,
        "batch_size_per_gpu": batch_size,
        "world_size": world_size,
        "lr_scaling_rule": lr_scaling_rule,
        "lr_scaling_mode": lr_scaling_mode,
        "lr_reference_world_size": reference_world_size,
        "lr_multiplier": lr_multiplier,
        "num_nodes": int(training.get("num_nodes", 1)),
        "gpus_per_node": int(training.get("num_gpus", 1)),
        "steps_per_pass": steps_per_pass,
        "total_optimizer_steps": total_optimizer_steps,
        "scheduler_warmup_steps": warmup_steps,
        "scheduler_decay_steps": total_optimizer_steps,
        "pass_semantics": "every_manifest_sample_at_least_once_per_epoch",
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_resolved": str(resolved_parent_path),
        "parent_checkpoint_sha256": _sha256(parent_path),
        "manifest_sha256": _sha256(manifest_path),
        "resume_checkpoint": (
            None if resume_checkpoint is None else str(resume_checkpoint)
        ),
        "resume_checkpoint_sha256": (
            None if resume_checkpoint is None else _sha256(resume_checkpoint)
        ),
        "resume_checkpoint_bytes": (
            None if resume_checkpoint is None else resume_checkpoint.stat().st_size
        ),
        "round_checkpoint_policy": "final_ema_teacher",
        "requested_checkpoint_interval": requested_checkpoint_interval,
        "requested_checkpoint_interval_unit": requested_checkpoint_unit,
        "effective_checkpoint_interval": training["checkpoint_interval"],
        "effective_checkpoint_interval_unit": training[
            "checkpoint_interval_unit"
        ],
        "requested_last_layer_freeze_steps": requested_freeze_steps,
        "effective_last_layer_freeze_steps": effective_freeze_steps,
        "checkpoint_policy": checkpoint_policy,
        "launch_spec_policy": "private_copy_per_launcher",
        "python_executable": str(python_executable),
        "python_executable_sha256": _sha256(python_executable),
        "implementation_sha256": {
            "train_cli": _sha256(Path(__file__)),
            "train_script": _sha256(
                Path(__file__).resolve().parents[1] / "scripts" / "train.py"
            ),
            "pl_model": _sha256(
                Path(__file__).resolve().parents[1] / "model" / "pl_model.py"
            ),
            "manifest_dataset": _sha256(
                Path(__file__).resolve().parents[2].joinpath(
                    "dinov3", "dataloader", "dataset.py"
                )
            ),
            "data_module": _sha256(
                Path(__file__).resolve().parents[2].joinpath(
                    "dinov3", "dataloader", "pl_dinov3_data_module.py"
                )
            ),
            "checkpoint_retention": _sha256(
                Path(__file__).resolve().parents[2].joinpath(
                    "dinov3", "model", "checkpoint_retention.py"
                )
            ),
            "checkpoint_callback": _sha256(
                Path(__file__).resolve().parents[1] / "model" / "checkpoint.py"
            ),
            "runtime_spec_writer": _sha256(
                Path(__file__).resolve().parents[1] / "utils" / "runtime_spec.py"
            ),
            "dinov3_entrypoint": _sha256(dinov3_entrypoint),
            "tao_entrypoint": _sha256(
                Path(__file__).resolve().parents[3] / "core" / "entrypoint.py"
            ),
        },
    }
    return spec, contract


def _latest_resume_checkpoint(output_dir: Path) -> Path | None:
    """Return the latest loadable full-state checkpoint for an interrupted round."""
    candidates = [
        path
        for pattern in ("model_*.pth", "dinov3_model_latest.pth")
        for path in output_dir.glob(pattern)
        if path.name != "checkpoint.pth"
    ]

    def position(path: Path) -> tuple[int, int, int, str]:
        match = re.fullmatch(
            r"model_(?:epoch_)?(\d+)_(?:step_)?(\d+)\.pth", path.name
        )
        if match is not None:
            return (1, int(match.group(1)), int(match.group(2)), path.name)
        return (0, 0, 0, path.name)

    candidates = sorted(candidates, key=position, reverse=True)
    for candidate in candidates:
        try:
            payload = _load_torch_checkpoint(candidate, weights_only=False)
        except (
            OSError,
            RuntimeError,
            EOFError,
            ValueError,
            IndexError,
            pickle.UnpicklingError,
        ):
            continue
        if not _is_lightning_full_state(payload):
            continue
        return candidate.resolve()
    return None


def _load_torch_checkpoint(path: Path, *, weights_only: bool):
    """Load a checkpoint, retaining mmap for modern zip serialization."""
    import torch

    try:
        return torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=weights_only,
        )
    except RuntimeError:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=weights_only,
        )


def _is_lightning_full_state(payload: object) -> bool:
    """Recognize a resumable Lightning checkpoint rather than weights only."""
    if not isinstance(payload, dict):
        return False
    state_dict = payload.get("state_dict")
    return all(
        (
            isinstance(state_dict, dict) and bool(state_dict),
            isinstance(payload.get("epoch"), int),
            isinstance(payload.get("global_step"), int),
            isinstance(payload.get("optimizer_states"), list),
            isinstance(payload.get("loops"), dict),
        )
    )


def _recover_final_teacher(
    output_dir: Path, *, expected_epoch: int, expected_step: int
) -> Path | None:
    """Recover a missing teacher export from a valid final Lightning state."""
    full_state = _latest_resume_checkpoint(output_dir)
    if full_state is None:
        return None
    payload = _load_torch_checkpoint(full_state, weights_only=False)
    if (
        int(payload.get("epoch", -1)) != expected_epoch or
        int(payload.get("global_step", -1)) != expected_step
    ):
        return None
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        return None
    teacher = {
        key.removeprefix("teacher.backbone."): value
        for key, value in state_dict.items()
        if key.startswith("teacher.backbone.") and
        not key.removeprefix("teacher.backbone.").startswith("dino_head.") and
        key.removeprefix("teacher.backbone.") != "mask_token"
    }
    if not teacher:
        return None
    destination = output_dir / (
        f"teacher_epoch_{expected_epoch:03d}_step_{expected_step:05d}.pth"
    )
    temporary = destination.with_name(destination.name + ".tmp")
    import torch

    torch.save(teacher, temporary)
    temporary.replace(destination)
    return destination


def _final_teacher_checkpoint(output_dir: Path) -> Path:
    """Select the final EMA teacher exported by the completed SSL round."""

    def position(path: Path) -> tuple[int, int]:
        match = re.fullmatch(r"teacher_epoch_(\d+)_step_(\d+)\.pth", path.name)
        if match is None:
            return (-1, -1)
        return (int(match.group(1)), int(match.group(2)))

    candidates = sorted(output_dir.glob("teacher_epoch_*_step_*.pth"), key=position)
    if not candidates:
        raise RuntimeError(
            f"DINOv3 train produced no EMA teacher checkpoint under {output_dir}"
        )
    return candidates[-1]


def _preparation_request_digest(
    *,
    base_spec: str | Path,
    manifest: str | Path,
    parent_checkpoint: str | Path,
    passes: int,
    output_dir: Path,
    num_nodes: int,
    gpus_per_node: int,
    checkpoint_policy: str,
    lr_scaling_rule: str = "sqrt_global_batch",
    lr_reference_world_size: int | None = None,
) -> str:
    return _canonical_digest(
        {
            "base_spec": str(Path(base_spec).expanduser().absolute()),
            "manifest": str(Path(manifest).expanduser().absolute()),
            "parent_checkpoint": str(
                Path(parent_checkpoint).expanduser().absolute()
            ),
            "passes": passes,
            "output_dir": str(output_dir),
            "num_nodes": num_nodes,
            "gpus_per_node": gpus_per_node,
            "checkpoint_policy": checkpoint_policy,
            "lr_scaling_rule": lr_scaling_rule,
            "lr_reference_world_size": lr_reference_world_size,
        }
    )


def _prepare_training(
    *,
    base_spec: str | Path,
    manifest: str | Path,
    parent_checkpoint: str | Path,
    passes: int,
    output_dir: Path,
    num_nodes: int,
    gpus_per_node: int,
    checkpoint_policy: str,
    request_digest: str,
    launch_id: str | None,
    lr_scaling_rule: str = "sqrt_global_batch",
    lr_reference_world_size: int | None = None,
) -> dict:
    (output_dir / "_SUCCESS").unlink(missing_ok=True)
    spec, contract = build_training_spec(
        base_spec=base_spec,
        manifest=manifest,
        parent_checkpoint=parent_checkpoint,
        passes=passes,
        output_dir=output_dir,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        checkpoint_policy=checkpoint_policy,
        lr_scaling_rule=lr_scaling_rule,
        lr_reference_world_size=lr_reference_world_size,
    )
    prepared_spec = output_dir / "refinement_input.yaml"
    _atomic_write_text(prepared_spec, yaml.safe_dump(spec, sort_keys=False))
    contract.update(
        {
            "preparation_request_digest": request_digest,
            "launch_id": launch_id,
            "prepared_spec": str(prepared_spec),
            "prepared_spec_sha256": _sha256(prepared_spec),
            "runtime_spec": str(output_dir / "experiment.yaml"),
            "effective_overrides": {
                "results_dir": str(output_dir),
                "train.results_dir": str(output_dir),
                "train.num_nodes": num_nodes,
                "train.num_gpus": gpus_per_node,
            },
        }
    )
    _atomic_write_text(
        output_dir / "training_contract.json",
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
    )
    return contract


def _publish_prepared_marker(
    output_dir: Path,
    *,
    contract: dict,
    launch_id: str,
    request_digest: str,
) -> None:
    contract_path = output_dir / "training_contract.json"
    _atomic_write_text(
        output_dir / ".refinement_prepared.json",
        json.dumps(
            {
                "schema_version": "1.0",
                "launch_id": launch_id,
                "preparation_request_digest": request_digest,
                "prepared_spec_sha256": contract["prepared_spec_sha256"],
                "training_contract_sha256": _sha256(contract_path),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
    )


def _verify_prepared_contract(
    output_dir: Path,
    contract: dict,
    *,
    request_digest: str | None = None,
    launch_id: str | None = None,
    verify_resume_checkpoint: bool = False,
) -> None:
    prepared_spec = Path(str(contract.get("prepared_spec", ""))).resolve()
    expected_spec = (output_dir / "refinement_input.yaml").resolve()
    if prepared_spec != expected_spec or not prepared_spec.is_file():
        raise RuntimeError("Training contract does not bind the prepared input spec")
    if _sha256(prepared_spec) != contract.get("prepared_spec_sha256"):
        raise RuntimeError("Prepared input spec changed after publication")
    if request_digest is not None and contract.get(
        "preparation_request_digest"
    ) != request_digest:
        raise RuntimeError("Prepared training request does not match this launcher")
    if launch_id is not None and contract.get("launch_id") != launch_id:
        raise RuntimeError("Prepared training launch ID does not match this launcher")
    static_inputs = {
        "base_spec_sha256": "base_spec",
        "manifest_sha256": "manifest",
        "parent_checkpoint_sha256": "parent_checkpoint",
    }
    for identity, path_field in static_inputs.items():
        if identity not in contract:
            continue
        input_path = Path(str(contract.get(path_field, "")))
        if not input_path.is_file() or _sha256(input_path) != contract[identity]:
            raise RuntimeError(f"Prepared training input identity changed: {identity}")
    resume_checkpoint = contract.get("resume_checkpoint")
    if verify_resume_checkpoint and resume_checkpoint is not None:
        resume_path = Path(str(resume_checkpoint))
        if not resume_path.is_file():
            raise RuntimeError("Prepared same-round resume checkpoint changed")
        resume_changed = (
            resume_path.stat().st_size != contract.get("resume_checkpoint_bytes")
        )
        resume_changed = resume_changed or (
            _sha256(resume_path) != contract.get("resume_checkpoint_sha256")
        )
        if resume_changed:
            raise RuntimeError("Prepared same-round resume checkpoint changed")


def _completed_training(
    output_dir: Path,
    *,
    request_digest: str,
) -> dict | None:
    """Validate and return an identical completed request, if one exists."""
    contract_path = output_dir / "training_contract.json"
    if not contract_path.is_file():
        return None
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("preparation_request_digest") != request_digest:
        raise RuntimeError(
            "Output directory belongs to a different refinement training request"
        )
    _verify_prepared_contract(output_dir, contract)
    if "checkpoint_sha256" not in contract:
        return None
    commit_sha256 = _validate_training_commit(output_dir, contract)
    _publish_or_verify_success(output_dir, commit_sha256)
    return contract


def _validate_training_commit(output_dir: Path, contract: dict) -> str:
    """Verify the externally sealed final contract and its runtime artifacts."""
    contract_path = output_dir / "training_contract.json"
    commit_path = output_dir / "training_commit.json"
    if Path(str(contract.get("commit_record", ""))).resolve() != commit_path.resolve():
        raise RuntimeError("Finalized contract does not bind its commit record path")
    expected = {
        "training_contract_sha256": _sha256(contract_path),
        "checkpoint_sha256": contract.get("checkpoint_sha256"),
        "runtime_spec_sha256": contract.get("runtime_spec_sha256"),
    }
    checkpoint = output_dir / "checkpoint.pth"
    if not checkpoint.is_file() or _sha256(checkpoint) != expected[
        "checkpoint_sha256"
    ]:
        raise RuntimeError("Finalized training checkpoint identity changed")
    runtime_spec = Path(str(contract.get("runtime_spec", "")))
    if not runtime_spec.is_file():
        raise RuntimeError("Finalized TAO runtime spec identity changed")
    runtime_spec_changed = (
        runtime_spec.stat().st_size != contract.get("runtime_spec_bytes")
    )
    runtime_spec_changed = runtime_spec_changed or (
        _sha256(runtime_spec) != expected["runtime_spec_sha256"]
    )
    if runtime_spec_changed:
        raise RuntimeError("Finalized TAO runtime spec identity changed")
    if not commit_path.is_file():
        _atomic_write_text(
            commit_path,
            json.dumps(
                {"schema_version": "1.0", **expected},
                indent=2,
                sort_keys=True,
            ) + "\n",
        )
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    for name, value in expected.items():
        if not value or commit.get(name) != value:
            raise RuntimeError(f"Finalized training commit does not bind {name}")
    return _sha256(commit_path)


def _publish_or_verify_success(output_dir: Path, commit_sha256: str) -> None:
    """Publish a missing final seal but never replace a conflicting one."""
    marker = output_dir / "_SUCCESS"
    if marker.is_file():
        if marker.read_text(encoding="utf-8").strip() != commit_sha256:
            raise RuntimeError("Training success seal conflicts with final commit")
        return
    _atomic_write_text(marker, commit_sha256 + "\n")


def _await_prepared_training(
    output_dir: Path,
    *,
    launch_id: str,
    request_digest: str,
) -> dict:
    timeout_seconds = float(
        os.environ.get("TAO_REFINEMENT_PREPARE_TIMEOUT_SECONDS", "300")
    )
    if timeout_seconds <= 0:
        raise ValueError("TAO_REFINEMENT_PREPARE_TIMEOUT_SECONDS must be positive")
    marker_path = output_dir / ".refinement_prepared.json"
    contract_path = output_dir / "training_contract.json"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("launch_id") == launch_id:
                if marker.get("preparation_request_digest") != request_digest:
                    raise RuntimeError(
                        "Conflicting preparation requests share one launch ID"
                    )
                if not contract_path.is_file():
                    raise RuntimeError("Prepared training contract is missing")
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
                current_contract_sha256 = _sha256(contract_path)
                prepared_contract_sha256 = marker.get("training_contract_sha256")
                finalized_prepared_sha256 = contract.get(
                    "prepared_contract_sha256"
                )
                if current_contract_sha256 != prepared_contract_sha256 and (
                    finalized_prepared_sha256 != prepared_contract_sha256
                ):
                    raise RuntimeError("Prepared training contract identity changed")
                if contract.get("prepared_spec_sha256") != marker.get(
                    "prepared_spec_sha256"
                ):
                    raise RuntimeError("Prepared input identity changed")
                _verify_prepared_contract(
                    output_dir,
                    contract,
                    request_digest=request_digest,
                    launch_id=(
                        None if "checkpoint_sha256" in contract else launch_id
                    ),
                )
                return contract
        time.sleep(0.1)
    raise TimeoutError(
        f"Timed out waiting for rank-zero training preparation: {marker_path}"
    )


def finalize_training(output_dir: str | Path) -> Path:
    """Publish a stable, zero-copy hard link to the round's final EMA teacher."""
    output_dir = Path(output_dir).resolve()
    contract_path = output_dir / "training_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if "prepared_spec_sha256" in contract:
        _verify_prepared_contract(output_dir, contract)
    checkpoint = output_dir / "checkpoint.pth"
    if "checkpoint_sha256" in contract:
        commit_sha256 = _validate_training_commit(output_dir, contract)
        _publish_or_verify_success(output_dir, commit_sha256)
        return checkpoint

    prepared_contract_sha256 = _sha256(contract_path)
    expected_epoch = int(contract["requested_data_passes"]) - 1
    expected_step = int(contract["total_optimizer_steps"])
    try:
        selected = _final_teacher_checkpoint(output_dir)
    except RuntimeError:
        selected = _recover_final_teacher(
            output_dir,
            expected_epoch=expected_epoch,
            expected_step=expected_step,
        )
        if selected is None:
            raise
    match = re.fullmatch(r"teacher_epoch_(\d+)_step_(\d+)\.pth", selected.name)
    teacher_extent_matches = match is not None
    if match is not None:
        teacher_extent_matches = all(
            (
                int(match.group(1)) == expected_epoch,
                int(match.group(2)) == expected_step,
            )
        )
    if not teacher_extent_matches:
        recovered = _recover_final_teacher(
            output_dir,
            expected_epoch=expected_epoch,
            expected_step=expected_step,
        )
        if recovered is None:
            raise RuntimeError(
                "Final EMA teacher does not match the requested training extent: "
                f"expected epoch {expected_epoch}, step {expected_step}; "
                f"found {selected.name}"
            )
        selected = recovered
    try:
        _load_torch_checkpoint(selected, weights_only=True)
    except (
        OSError,
        RuntimeError,
        EOFError,
        ValueError,
        IndexError,
        pickle.UnpicklingError,
    ) as error:
        recovered = _recover_final_teacher(
            output_dir,
            expected_epoch=expected_epoch,
            expected_step=expected_step,
        )
        if recovered is None:
            raise RuntimeError(
                f"Final EMA teacher checkpoint is not loadable: {selected}"
            ) from error
        selected = recovered
    temporary_checkpoint = output_dir / "checkpoint.tmp.pth"
    temporary_checkpoint.unlink(missing_ok=True)
    temporary_checkpoint.hardlink_to(selected)
    temporary_checkpoint.replace(checkpoint)
    contract["prepared_contract_sha256"] = prepared_contract_sha256
    contract["checkpoint"] = str(checkpoint)
    contract["source_checkpoint_name"] = selected.name
    contract["checkpoint_sha256"] = _sha256(checkpoint)
    contract["checkpoint_bytes"] = checkpoint.stat().st_size
    runtime_spec = Path(str(contract.get("runtime_spec", "")))
    if not runtime_spec.is_file():
        raise RuntimeError("TAO train produced no runtime experiment spec")
    contract["runtime_spec_sha256"] = _sha256(runtime_spec)
    contract["runtime_spec_bytes"] = runtime_spec.stat().st_size
    commit_path = output_dir / "training_commit.json"
    contract["commit_record"] = str(commit_path)
    _atomic_write_text(
        contract_path, json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    commit_sha256 = _validate_training_commit(output_dir, contract)
    _publish_or_verify_success(output_dir, commit_sha256)
    return checkpoint


def main(argv: list[str] | None = None) -> int:
    """Build a derived DINOv3 spec and execute one refinement training round."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-spec")
    parser.add_argument("--manifest")
    parser.add_argument("--checkpoint")
    parser.add_argument("--passes", type=int)
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--gpus-per-node", type=int)
    parser.add_argument(
        "--lr-scaling-rule",
        choices=("sqrt_global_batch",),
        default="sqrt_global_batch",
    )
    parser.add_argument("--lr-reference-world-size", type=int)
    parser.add_argument(
        "--checkpoint-policy",
        choices=(
            "base_checkpoint_each_round",
            "previous_round_checkpoint",
            "caller_selected",
        ),
        default="caller_selected",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.finalize_only:
        finalize_training(output_dir)
        return 0
    required = {
        name: getattr(args, name)
        for name in ("base_spec", "manifest", "checkpoint", "passes")
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    num_nodes, gpus_per_node = _configured_resources(
        base_spec=args.base_spec,
        num_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
    )
    request_digest = _preparation_request_digest(
        base_spec=args.base_spec,
        manifest=args.manifest,
        parent_checkpoint=args.checkpoint,
        passes=args.passes,
        output_dir=output_dir,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        checkpoint_policy=args.checkpoint_policy,
        lr_scaling_rule=args.lr_scaling_rule,
        lr_reference_world_size=args.lr_reference_world_size,
    )
    if args.dry_run or args.prepare_only:
        if _completed_training(
            output_dir,
            request_digest=request_digest,
        ) is not None:
            return 0
        _prepare_training(
            base_spec=args.base_spec,
            manifest=args.manifest,
            parent_checkpoint=args.checkpoint,
            passes=args.passes,
            output_dir=output_dir,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            checkpoint_policy=args.checkpoint_policy,
            request_digest=request_digest,
            launch_id=None,
            lr_scaling_rule=args.lr_scaling_rule,
            lr_reference_world_size=args.lr_reference_world_size,
        )
        return 0

    node_rank, launch_id, child_environment = _distributed_launch_environment(
        num_nodes=num_nodes, gpus_per_node=gpus_per_node
    )
    if node_rank == 0:
        completed_contract = _completed_training(
            output_dir,
            request_digest=request_digest,
        )
        if completed_contract is not None:
            if launch_id is not None:
                _publish_prepared_marker(
                    output_dir,
                    contract=completed_contract,
                    launch_id=launch_id,
                    request_digest=request_digest,
                )
            return 0
        contract = _prepare_training(
            base_spec=args.base_spec,
            manifest=args.manifest,
            parent_checkpoint=args.checkpoint,
            passes=args.passes,
            output_dir=output_dir,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            checkpoint_policy=args.checkpoint_policy,
            request_digest=request_digest,
            launch_id=launch_id,
            lr_scaling_rule=args.lr_scaling_rule,
            lr_reference_world_size=args.lr_reference_world_size,
        )
        if launch_id is not None:
            _publish_prepared_marker(
                output_dir,
                contract=contract,
                launch_id=launch_id,
                request_digest=request_digest,
            )
    else:
        if launch_id is None:
            raise RuntimeError("Distributed nonzero rank has no launch ID")
        contract = _await_prepared_training(
            output_dir,
            launch_id=launch_id,
            request_digest=request_digest,
        )

    prepared_spec = Path(contract["prepared_spec"])
    _verify_prepared_contract(
        output_dir,
        contract,
        request_digest=request_digest,
        launch_id=(None if "checkpoint_sha256" in contract else launch_id),
        verify_resume_checkpoint=True,
    )
    if node_rank != 0 and "checkpoint_sha256" in contract:
        _validate_training_commit(output_dir, contract)
        return 0
    with _launch_spec(prepared_spec) as launch_spec:
        dinov3_entrypoint = (
            Path(__file__).resolve().parents[1] / "entrypoint" / "dinov3.py"
        )
        subprocess.run(
            [
                sys.executable,
                str(dinov3_entrypoint),
                "train",
                "-e",
                str(launch_spec),
                f"results_dir={output_dir}",
                f"train.results_dir={output_dir}",
                f"train.num_nodes={contract['num_nodes']}",
                f"train.num_gpus={contract['gpus_per_node']}",
            ],
            check=True,
            env=child_environment,
        )
    _verify_prepared_contract(
        output_dir,
        contract,
        request_digest=request_digest,
        launch_id=launch_id,
    )
    if node_rank == 0:
        finalize_training(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
