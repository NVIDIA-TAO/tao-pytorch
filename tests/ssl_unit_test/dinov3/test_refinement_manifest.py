# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for manifest-backed DINOv3 refinement training."""

import ast
from pathlib import Path
import hashlib
import json
import os
import sys
import tarfile
import zipfile

import pandas as pd
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.utilities import rank_zero_only
import pytest
import torch
import yaml

from nvidia_tao_pytorch.core.decorators import workflow
from nvidia_tao_pytorch.ssl.dinov3.data_refinement import train_cli
from nvidia_tao_pytorch.ssl.dinov3.data_refinement.train_cli import (
    _distributed_launch_environment,
    _final_teacher_checkpoint,
    _latest_resume_checkpoint,
    build_training_spec,
    finalize_training,
)
from nvidia_tao_pytorch.ssl.dinov3.dataloader.dataset import (
    DinoV3Dataset,
    ShardAwareDistributedSampler,
)
from nvidia_tao_pytorch.ssl.dinov3.model.checkpoint_retention import (
    prune_periodic_ssl_checkpoints,
)
from nvidia_tao_pytorch.ssl.dinov3.scripts.train import (
    _configure_refinement_status_rank,
    _refinement_trainer_plugins,
)
from nvidia_tao_pytorch.ssl.dinov3.utils.runtime_spec import publish_runtime_spec


def _transform(_image):
    return {"global_crops": ["global"], "local_crops": ["local"]}


@pytest.mark.parametrize(("relative_path", "reader", "count"), [
    ("dinov3/dataloader/dataset.py", "read_table", 1),
])
def test_deft_parquet_reads_disable_only_background_prefetch(
    relative_path, reader, count
):
    """Pin the image-teardown mitigation without disabling compute threads."""
    source_root = Path(__file__).resolve().parents[3] / "nvidia_tao_pytorch" / "ssl"
    module = ast.parse((source_root / relative_path).read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == reader
    ]
    assert len(calls) == count
    for call in calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert ast.literal_eval(keywords["pre_buffer"]) is False
        assert "use_threads" not in keywords


def test_single_node_adapter_preserves_scheduler_gpu_identity(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-assigned")
    monkeypatch.setenv("WORLD_SIZE", "8")
    rank, launch_id, environment = _distributed_launch_environment(
        num_nodes=1, gpus_per_node=1
    )
    assert rank == 0 and launch_id is None
    assert "WORLD_SIZE" not in environment
    assert "RANK" not in environment
    assert environment["TAO_REFINEMENT_LIGHTNING_LAUNCH"] == "1"
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-assigned"


@pytest.mark.parametrize("visible", ["", "GPU-a,GPU-b"])
def test_single_node_adapter_rejects_wrong_gpu_allocation(monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError, match="exactly match"):
        _distributed_launch_environment(num_nodes=1, gpus_per_node=1)


@pytest.mark.parametrize("variable", ["LOCAL_RANK", "LOCAL_WORLD_SIZE", "TORCHELASTIC_RUN_ID"])
def test_adapter_rejects_inherited_process_launcher(monkeypatch, variable):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-assigned")
    monkeypatch.setenv(variable, "1")
    with pytest.raises(ValueError, match="one adapter process per node"):
        _distributed_launch_environment(num_nodes=1, gpus_per_node=1)


def test_refinement_forces_lightning_worker_launcher_under_slurm(monkeypatch):
    """One Slurm adapter task per node still lets Lightning spawn local workers."""
    monkeypatch.setenv("TAO_REFINEMENT_LIGHTNING_LAUNCH", "1")
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "1")
    monkeypatch.setenv("SLURM_NNODES", "2")
    monkeypatch.setenv("SLURM_JOB_NAME", "deft")
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    plugins = _refinement_trainer_plugins()
    trainer = Trainer(
        accelerator="cpu",
        devices=2,
        num_nodes=2,
        strategy="ddp",
        plugins=plugins,
        logger=False,
        enable_checkpointing=False,
    )
    environment = trainer.strategy.cluster_environment
    assert isinstance(environment, LightningEnvironment)
    assert environment.creates_processes_externally is False


def test_non_slurm_nonzero_node_suppresses_pretrainer_status(
    tmp_path, monkeypatch
):
    """A node-level worker must not publish status before Trainer sets ranks."""
    monkeypatch.setenv("TAO_REFINEMENT_LIGHTNING_LAUNCH", "1")
    monkeypatch.setenv("NODE_RANK", "1")
    monkeypatch.setenv("NUM_GPU_PER_NODE", "2")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    monkeypatch.setattr(rank_zero_only, "rank", 0)
    monkeypatch.setattr(workflow, "update_results_dir", lambda cfg, **_: cfg)

    _configure_refinement_status_rank()
    assert rank_zero_only.rank == 2

    config = {"results_dir": str(tmp_path)}

    @workflow.monitor_status(
        name="DINOv3", mode="train", spec_writer=publish_runtime_spec
    )
    def run(_cfg):
        return None

    run(config)
    assert not (tmp_path / "experiment.yaml").exists()
    assert (tmp_path / "status.json").read_text(encoding="utf-8") == ""


def _full_state(epoch: int, step: int) -> dict:
    return {
        "epoch": epoch,
        "global_step": step,
        "state_dict": {"teacher.backbone.weight": torch.ones(2)},
        "optimizer_states": [{}],
        "loops": {},
    }


def test_dinov3_uses_lightning_launcher_with_strict_validation() -> None:
    entrypoint = (
        Path(__file__).resolve().parents[3]
        / "nvidia_tao_pytorch"
        / "ssl" / "dinov3" / "entrypoint" / "dinov3.py"
    )
    module = ast.parse(entrypoint.read_text(encoding="utf-8"))
    launch = next(node for node in ast.walk(module) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == "launch")
    keywords = {kw.arg for kw in launch.keywords}
    assert keywords == {"network", "strict_multinode"}
    assert "TAO_STRICT_MULTINODE" in entrypoint.read_text(encoding="utf-8")


def test_dataset_reads_exact_manifest_order(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (4, 4), "red").save(first)
    Image.new("RGB", (4, 4), "blue").save(second)
    manifest = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "sample_id": ["b", "a"],
            "storage_type": ["file", "file"],
            "path": [str(second), str(first)],
            "member": [None, None],
        }
    ).to_parquet(manifest, index=False)
    dataset = DinoV3Dataset(
        root="/", manifest_path=manifest, transform=_transform, train=False
    )
    assert len(dataset) == 2
    assert dataset[0]["input_path"] == str(second)
    assert dataset[1]["input_path"] == str(first)


def test_build_training_spec_sets_manifest_and_passes(tmp_path: Path) -> None:
    manifest = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "sample_id": ["a"],
            "storage_type": ["file"],
            "path": ["/data/a.jpg"],
            "member": [None],
        }
    ).to_parquet(manifest, index=False)
    checkpoint = tmp_path / "base.pth"
    checkpoint.write_text("base", encoding="utf-8")
    base_spec = tmp_path / "base.yaml"
    base_spec.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "batch_size": 1,
                    "train_dataset": {"images_dir": "/data"},
                },
                "train": {
                    "num_epochs": 1,
                    "num_nodes": 1,
                    "num_gpus": 1,
                    "checkpoint_interval": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    spec, contract = build_training_spec(
        base_spec=base_spec,
        manifest=manifest,
        parent_checkpoint=checkpoint,
        passes=12,
        output_dir=tmp_path / "output",
        checkpoint_policy="base_checkpoint_each_round",
    )
    assert spec["dataset"]["train_manifest"] == str(manifest.resolve())
    assert spec["train"]["num_epochs"] == 12
    assert spec["train"]["results_dir"] == str((tmp_path / "output").resolve())
    assert contract["manifest_rows"] == 1
    assert contract["requested_data_passes"] == 12
    assert contract["steps_per_pass"] == 1
    assert contract["total_optimizer_steps"] == 12
    assert contract["scheduler_warmup_steps"] == 1
    assert spec["train"]["schedulers"]["learning_rate"]["max_decay_steps"] == 12
    assert contract["round_checkpoint_policy"] == "final_ema_teacher"
    assert contract["checkpoint_policy"] == "base_checkpoint_each_round"
    assert contract["python_executable"] == str(Path(sys.executable).resolve())
    assert contract["requested_checkpoint_interval"] == 3
    assert contract["effective_checkpoint_interval"] == 3


def test_build_training_spec_rejects_distillation_before_training(tmp_path: Path) -> None:
    """Never publish a frozen distillation teacher as the learned DEFT model."""
    manifest = tmp_path / "train.parquet"
    manifest.touch()
    checkpoint = tmp_path / "base.pth"
    checkpoint.touch()
    base_spec = tmp_path / "base.yaml"
    base_spec.write_text(
        yaml.safe_dump({"model": {"distill": {"enable": True}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model.distill.enable=false"):
        build_training_spec(
            base_spec=base_spec,
            manifest=manifest,
            parent_checkpoint=checkpoint,
            passes=1,
            output_dir=tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()


def test_build_training_spec_scales_numeric_lr_with_global_batch(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "sample_id": ["a"],
            "storage_type": ["file"],
            "path": ["/data/a.jpg"],
        }
    ).to_parquet(manifest, index=False)
    checkpoint = tmp_path / "base.pth"
    checkpoint.write_text("base", encoding="utf-8")
    base_spec = tmp_path / "base.yaml"
    base_spec.write_text(
        yaml.safe_dump(
            {
                "dataset": {"batch_size": 2},
                "train": {
                    "num_nodes": 1,
                    "num_gpus": 1,
                    "schedulers": {
                        "learning_rate": {"val_base": 0.001},
                        "last_layer_learning_rate": {"val_base": 0.002},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    spec, contract = build_training_spec(
        base_spec=base_spec,
        manifest=manifest,
        parent_checkpoint=checkpoint,
        passes=1,
        output_dir=tmp_path / "output",
        num_nodes=2,
        gpus_per_node=2,
        lr_reference_world_size=1,
    )
    assert spec["train"]["schedulers"]["learning_rate"]["val_base"] == 0.002
    assert (
        spec["train"]["schedulers"]["last_layer_learning_rate"]["val_base"]
        == 0.004
    )
    assert contract["lr_multiplier"] == 2.0
    assert contract["lr_scaling_mode"] == "numeric_sqrt_scaled"


def test_node_scaling_requires_explicit_batch_aware_learning_rates(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "sample_id": ["a"],
            "storage_type": ["file"],
            "path": ["/data/a.jpg"],
        }
    ).to_parquet(manifest, index=False)
    checkpoint = tmp_path / "base.pth"
    checkpoint.write_text("base", encoding="utf-8")
    base_spec = tmp_path / "base.yaml"
    base_spec.write_text(
        yaml.safe_dump({"dataset": {"batch_size": 1}, "train": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="val_base must be explicitly numeric"):
        build_training_spec(
            base_spec=base_spec,
            manifest=manifest,
            parent_checkpoint=checkpoint,
            passes=1,
            output_dir=tmp_path / "output",
            num_nodes=2,
            gpus_per_node=2,
            lr_reference_world_size=1,
        )


def test_build_training_spec_preserves_checkpoint_symlink_name(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "sample_id": ["a"],
            "storage_type": ["file"],
            "path": ["/data/a.jpg"],
        }
    ).to_parquet(manifest, index=False)
    blob = tmp_path / "checkpoint-blob"
    blob.write_text("weights", encoding="utf-8")
    alias = tmp_path / "model.safetensors"
    alias.symlink_to(blob)
    base_spec = tmp_path / "base.yaml"
    base_spec.write_text(
        yaml.safe_dump(
            {
                "dataset": {"batch_size": 1},
                "train": {"num_nodes": 1, "num_gpus": 1},
            }
        ),
        encoding="utf-8",
    )
    spec, contract = build_training_spec(
        base_spec=base_spec,
        manifest=manifest,
        parent_checkpoint=alias,
        passes=1,
        output_dir=tmp_path / "output",
    )
    assert spec["train"]["pretrained_model_path"] == str(alias.absolute())
    assert contract["parent_checkpoint"] == str(alias.absolute())
    assert contract["parent_checkpoint_resolved"] == str(blob.resolve())


def test_build_training_spec_applies_resolved_distributed_allocation(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(128)],
            "storage_type": ["file"] * 128,
            "path": [f"/data/{index}.jpg" for index in range(128)],
        }
    ).to_parquet(manifest, index=False)
    checkpoint = tmp_path / "base.pth"
    checkpoint.write_text("base", encoding="utf-8")
    base_spec = tmp_path / "base.yaml"
    base_spec.write_text(
        yaml.safe_dump(
            {
                "dataset": {"batch_size": 2},
                "train": {"num_nodes": 1, "num_gpus": 1},
            }
        ),
        encoding="utf-8",
    )

    spec, contract = build_training_spec(
        base_spec=base_spec,
        manifest=manifest,
        parent_checkpoint=checkpoint,
        passes=2,
        output_dir=tmp_path / "output",
        num_nodes=4,
        gpus_per_node=2,
    )

    assert spec["train"]["num_nodes"] == 4
    assert spec["train"]["num_gpus"] == 2
    assert contract["num_nodes"] == 4
    assert contract["gpus_per_node"] == 2
    assert contract["world_size"] == 8
    assert contract["total_optimizer_steps"] == 16


def test_training_spec_checkpoint_cadence_reaches_non_divisible_final_pass(
    tmp_path: Path,
) -> None:
    """Train through a non-divisible cadence and finalize the actual last teacher."""
    from types import SimpleNamespace

    import pytorch_lightning as pl

    from nvidia_tao_pytorch.ssl.dinov3.model.checkpoint import DinoV3ModelCheckpoint

    manifest = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "sample_id": ["a"],
            "storage_type": ["file"],
            "path": ["/data/a.jpg"],
        }
    ).to_parquet(manifest, index=False)
    checkpoint = tmp_path / "base.pth"
    checkpoint.write_text("base", encoding="utf-8")
    base_spec = tmp_path / "base.yaml"
    base_spec.write_text(
        yaml.safe_dump(
            {
                "dataset": {"batch_size": 1},
                "train": {
                    "num_nodes": 1,
                    "num_gpus": 1,
                    "checkpoint_interval": 3,
                    "checkpoint_interval_unit": "step",
                },
            }
        ),
        encoding="utf-8",
    )
    spec, contract = build_training_spec(
        base_spec=base_spec,
        manifest=manifest,
        parent_checkpoint=checkpoint,
        passes=13,
        output_dir=tmp_path / "output",
    )
    assert spec["train"]["checkpoint_interval_unit"] == "step"
    assert spec["train"]["checkpoint_interval"] == 3
    assert contract["requested_checkpoint_interval"] == 3
    assert contract["effective_checkpoint_interval"] == 3
    assert contract["effective_checkpoint_interval_unit"] == "step"

    class TinyTeacher(pl.LightningModule):
        """Exercise real Lightning checkpoint scheduling without a GPU backbone."""

        def __init__(self):
            super().__init__()
            self.teacher = torch.nn.ModuleDict({"backbone": torch.nn.Linear(1, 1)})
            self.model_config = SimpleNamespace(distill=SimpleNamespace(enable=False))

        def training_step(self, batch, batch_idx):
            """Update the teacher so the final export has a verifiable identity."""
            return self.teacher["backbone"](batch).square().mean()

        def configure_optimizers(self):
            """Use one optimizer step per manifest batch."""
            return torch.optim.SGD(self.parameters(), lr=0.01)

    output_dir = tmp_path / "output"
    callback = DinoV3ModelCheckpoint(
        save_final_epoch=True,
        every_n_train_steps=spec["train"]["checkpoint_interval"],
        every_n_epochs=None,
        dirpath=str(output_dir),
        save_on_train_epoch_end=False,
        monitor=None,
        save_top_k=-1,
        save_last="link",
        filename="model_{epoch:03d}_{step:05d}",
        enable_version_counter=False,
    )
    model = TinyTeacher()
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=spec["train"]["num_epochs"],
        default_root_dir=str(output_dir), callbacks=[callback], logger=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(model, torch.utils.data.DataLoader(torch.ones(1, 1), batch_size=1))
    assert trainer.global_step == contract["total_optimizer_steps"] == 13
    assert callback._last_global_step_saved == trainer.global_step
    saved_steps = sorted(
        int(path.stem.rsplit("_", 1)[1]) for path in output_dir.glob("model_*.pth")
    )
    assert saved_steps == [3, 6, 9, 12, 13]
    full_state = torch.load(_latest_resume_checkpoint(output_dir), weights_only=False)
    assert (full_state["epoch"], full_state["global_step"]) == (12, 13)
    runtime_spec = output_dir / "experiment.yaml"
    runtime_spec.write_text(yaml.safe_dump(spec), encoding="utf-8")
    contract["runtime_spec"] = str(runtime_spec)
    (output_dir / "training_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    final_checkpoint = finalize_training(output_dir)
    final_teacher = torch.load(final_checkpoint, weights_only=True)
    for name, value in model.teacher["backbone"].state_dict().items():
        assert torch.equal(final_teacher[name], value)
    assert (output_dir / "_SUCCESS").is_file()
    assert finalize_training(output_dir) == final_checkpoint


def test_multinode_launch_uses_private_spec_and_explicit_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "storage_type": ["file", "file"],
            "path": ["/data/a.jpg", "/data/b.jpg"],
        }
    ).to_parquet(manifest, index=False)
    checkpoint = tmp_path / "base.pth"
    checkpoint.write_text("base", encoding="utf-8")
    base_spec = tmp_path / "base.yaml"
    base_spec.write_text(
        yaml.safe_dump(
            {
                "dataset": {"batch_size": 1},
                "train": {
                    "num_nodes": 1,
                    "num_gpus": 1,
                    "checkpoint_interval": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    temporary_dir = tmp_path / "node-local"
    temporary_dir.mkdir()
    observed: dict[str, object] = {}

    def fake_run(command: list[str], *, check: bool, env: dict[str, str]) -> None:
        launch_spec = Path(command[command.index("-e") + 1])
        observed.update(
            command=command, launch_spec=launch_spec, check=check, env=env
        )
        assert launch_spec != output_dir / "refinement_input.yaml"
        assert yaml.safe_load(launch_spec.read_text(encoding="utf-8"))["train"][
            "num_nodes"
        ] == 2
        launch_spec.write_text("entrypoint_mutated: true\n", encoding="utf-8")
        (output_dir / "experiment.yaml").write_text(
            "runtime_spec: true\n", encoding="utf-8"
        )
        torch.save(
            {"weight": torch.ones(1)},
            output_dir / "teacher_epoch_001_step_00002.pth",
        )

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("NODE_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("NUM_GPU_PER_NODE", "8")
    monkeypatch.setenv("TAO_REFINEMENT_LAUNCH_ID", "test-launch")
    monkeypatch.setenv("TMPDIR", str(temporary_dir))
    monkeypatch.setattr(train_cli.subprocess, "run", fake_run)

    arguments = [
        "--base-spec",
        str(base_spec),
        "--manifest",
        str(manifest),
        "--checkpoint",
        str(checkpoint),
        "--passes",
        "2",
        "--num-nodes",
        "2",
        "--gpus-per-node",
        "8",
        "--checkpoint-policy",
        "base_checkpoint_each_round",
        "--output-dir",
        str(output_dir),
    ]
    assert train_cli.main(arguments) == 0

    command = observed["command"]
    assert isinstance(command, list)
    assert command[0] == sys.executable
    assert command[1].endswith("/ssl/dinov3/entrypoint/dinov3.py")
    assert f"results_dir={output_dir.resolve()}" in command
    assert f"train.results_dir={output_dir.resolve()}" in command
    assert "train.num_nodes=2" in command
    assert "train.num_gpus=8" in command
    assert observed["check"] is True
    child_environment = observed["env"]
    assert isinstance(child_environment, dict)
    assert child_environment["TAO_STRICT_MULTINODE"] == "1"
    assert child_environment["NUM_GPU_PER_NODE"] == "8"
    assert not Path(observed["launch_spec"]).exists()
    canonical_spec = yaml.safe_load(
        (output_dir / "refinement_input.yaml").read_text(encoding="utf-8")
    )
    assert canonical_spec["results_dir"] == str(output_dir.resolve())
    assert canonical_spec["train"]["results_dir"] == str(output_dir.resolve())
    assert (output_dir / "experiment.yaml").read_text(
        encoding="utf-8"
    ) == "runtime_spec: true\n"
    contract = json.loads(
        (output_dir / "training_contract.json").read_text(encoding="utf-8")
    )
    assert contract["launch_spec_policy"] == "private_copy_per_launcher"
    assert contract["checkpoint_policy"] == "base_checkpoint_each_round"
    assert contract["prepared_spec"] == str(output_dir / "refinement_input.yaml")
    assert contract["prepared_spec_sha256"] == "sha256:" + hashlib.sha256(
        (output_dir / "refinement_input.yaml").read_bytes()
    ).hexdigest()

    (output_dir / "training_commit.json").unlink()
    (output_dir / "_SUCCESS").unlink()

    def unexpected_run(
        command: list[str], *, check: bool, env: dict[str, str]
    ) -> None:
        del command, check, env
        raise AssertionError("Completed recovery must not relaunch training")

    monkeypatch.setattr(train_cli.subprocess, "run", unexpected_run)
    monkeypatch.setenv("TAO_REFINEMENT_LAUNCH_ID", "test-launch-recovery")
    assert train_cli.main(arguments) == 0
    commit_mtime = (output_dir / "training_commit.json").stat().st_mtime_ns
    success_mtime = (output_dir / "_SUCCESS").stat().st_mtime_ns

    monkeypatch.setenv("NODE_RANK", "1")
    assert train_cli.main(arguments) == 0
    assert (output_dir / "training_commit.json").stat().st_mtime_ns == commit_mtime
    assert (output_dir / "_SUCCESS").stat().st_mtime_ns == success_mtime


def test_multinode_nonzero_rank_consumes_preparation_without_finalizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "sample_id": ["a"],
            "storage_type": ["file"],
            "path": ["/data/a.jpg"],
        }
    ).to_parquet(manifest, index=False)
    checkpoint = tmp_path / "base.pth"
    checkpoint.write_text("base", encoding="utf-8")
    base_spec = tmp_path / "base.yaml"
    base_spec.write_text(
        yaml.safe_dump(
            {
                "dataset": {"batch_size": 1},
                "train": {"num_nodes": 2, "num_gpus": 8},
            }
        ),
        encoding="utf-8",
    )
    output_dir = (tmp_path / "output").resolve()
    output_dir.mkdir()
    request_digest = train_cli._preparation_request_digest(
        base_spec=base_spec,
        manifest=manifest,
        parent_checkpoint=checkpoint,
        passes=2,
        output_dir=output_dir,
        num_nodes=2,
        gpus_per_node=8,
        checkpoint_policy="base_checkpoint_each_round",
    )
    contract = train_cli._prepare_training(
        base_spec=base_spec,
        manifest=manifest,
        parent_checkpoint=checkpoint,
        passes=2,
        output_dir=output_dir,
        num_nodes=2,
        gpus_per_node=8,
        checkpoint_policy="base_checkpoint_each_round",
        request_digest=request_digest,
        launch_id="test-launch",
    )
    train_cli._publish_prepared_marker(
        output_dir,
        contract=contract,
        launch_id="test-launch",
        request_digest=request_digest,
    )
    launched = []

    def fake_run(command: list[str], *, check: bool, env: dict[str, str]) -> None:
        launched.append((command, check, env))

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("NODE_RANK", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("NUM_GPU_PER_NODE", "8")
    monkeypatch.setenv("TAO_REFINEMENT_LAUNCH_ID", "test-launch")
    monkeypatch.setattr(train_cli.subprocess, "run", fake_run)

    assert train_cli.main(
        [
            "--base-spec",
            str(base_spec),
            "--manifest",
            str(manifest),
            "--checkpoint",
            str(checkpoint),
            "--passes",
            "2",
            "--num-nodes",
            "2",
            "--gpus-per-node",
            "8",
            "--checkpoint-policy",
            "base_checkpoint_each_round",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    assert len(launched) == 1
    assert not (output_dir / "checkpoint.pth").exists()
    assert not (output_dir / "_SUCCESS").exists()
    final_contract = json.loads(
        (output_dir / "training_contract.json").read_text(encoding="utf-8")
    )
    assert "checkpoint_sha256" not in final_contract


def test_training_rejects_prepared_spec_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "sample_id": ["a"],
            "storage_type": ["file"],
            "path": ["/data/a.jpg"],
        }
    ).to_parquet(manifest, index=False)
    checkpoint = tmp_path / "base.pth"
    checkpoint.write_text("base", encoding="utf-8")
    base_spec = tmp_path / "base.yaml"
    base_spec.write_text(
        yaml.safe_dump(
            {
                "dataset": {"batch_size": 1},
                "train": {"num_nodes": 1, "num_gpus": 1},
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    def fake_run(command: list[str], *, check: bool, env: dict[str, str]) -> None:
        del command, check, env
        (output_dir / "refinement_input.yaml").write_text(
            "tampered: true\n", encoding="utf-8"
        )

    monkeypatch.setattr(train_cli.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="changed after publication"):
        train_cli.main(
            [
                "--base-spec",
                str(base_spec),
                "--manifest",
                str(manifest),
                "--checkpoint",
                str(checkpoint),
                "--passes",
                "1",
                "--output-dir",
                str(output_dir),
            ]
        )


def test_completed_training_is_idempotent_and_input_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "sample_id": ["a"],
            "storage_type": ["file"],
            "path": ["/data/a.jpg"],
        }
    ).to_parquet(manifest, index=False)
    checkpoint = tmp_path / "base.pth"
    checkpoint.write_text("base", encoding="utf-8")
    base_spec = tmp_path / "base.yaml"
    base_spec.write_text(
        yaml.safe_dump(
            {
                "dataset": {"batch_size": 1},
                "train": {"num_nodes": 1, "num_gpus": 1},
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    arguments = [
        "--base-spec",
        str(base_spec),
        "--manifest",
        str(manifest),
        "--checkpoint",
        str(checkpoint),
        "--passes",
        "1",
        "--output-dir",
        str(output_dir),
    ]
    launches = []

    def fake_run(command: list[str], *, check: bool, env: dict[str, str]) -> None:
        launches.append((command, check, env))
        (output_dir / "experiment.yaml").write_text(
            "runtime_spec: true\n", encoding="utf-8"
        )
        torch.save(
            {"weight": torch.ones(1)},
            output_dir / "teacher_epoch_000_step_00001.pth",
        )

    monkeypatch.setattr(train_cli.subprocess, "run", fake_run)
    assert train_cli.main(arguments) == 0
    assert len(launches) == 1

    (output_dir / "_SUCCESS").unlink()
    assert train_cli.main(arguments) == 0
    assert len(launches) == 1
    assert (output_dir / "_SUCCESS").is_file()

    base_spec.write_text("dataset:\n  batch_size: 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="input identity changed"):
        train_cli.main(arguments)


def test_training_refuses_output_directory_from_different_request(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "training_contract.json").write_text(
        json.dumps({"preparation_request_digest": "sha256:different"}),
        encoding="utf-8",
    )
    manifest = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "sample_id": ["a"],
            "storage_type": ["file"],
            "path": ["/data/a.jpg"],
        }
    ).to_parquet(manifest, index=False)
    checkpoint = tmp_path / "base.pth"
    checkpoint.write_text("base", encoding="utf-8")
    base_spec = tmp_path / "base.yaml"
    base_spec.write_text(
        yaml.safe_dump(
            {
                "dataset": {"batch_size": 1},
                "train": {"num_nodes": 1, "num_gpus": 1},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="different refinement training request"):
        train_cli.main(
            [
                "--base-spec",
                str(base_spec),
                "--manifest",
                str(manifest),
                "--checkpoint",
                str(checkpoint),
                "--passes",
                "1",
                "--output-dir",
                str(output_dir),
                "--prepare-only",
            ]
        )

def test_multinode_launch_rejects_resource_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("NODE_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("NUM_GPU_PER_NODE", "8")
    monkeypatch.setenv("TAO_REFINEMENT_LAUNCH_ID", "test-launch")
    with pytest.raises(ValueError, match="WORLD_SIZE is the node count"):
        _distributed_launch_environment(num_nodes=2, gpus_per_node=8)


def test_multinode_launch_requires_complete_rendezvous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "WORLD_SIZE",
        "NODE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NUM_GPU_PER_NODE",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="missing"):
        _distributed_launch_environment(num_nodes=2, gpus_per_node=8)


def test_multinode_launch_requires_attempt_unique_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("NODE_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("NUM_GPU_PER_NODE", "8")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.delenv("TAO_REFINEMENT_LAUNCH_ID", raising=False)
    with pytest.raises(ValueError, match="attempt-unique"):
        _distributed_launch_environment(num_nodes=2, gpus_per_node=8)


def test_multinode_launch_preserves_exact_scheduler_device_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("NODE_RANK", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("NUM_GPU_PER_NODE", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    monkeypatch.setenv("TAO_REFINEMENT_LAUNCH_ID", "12345.1")

    _, _, environment = _distributed_launch_environment(
        num_nodes=2, gpus_per_node=2
    )

    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b"
    assert environment["TAO_STRICT_MULTINODE"] == "1"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c")
    with pytest.raises(ValueError, match="exactly match"):
        _distributed_launch_environment(num_nodes=2, gpus_per_node=2)


def test_interrupted_training_refuses_changed_inputs_before_resume(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "sample_id": ["a"],
            "storage_type": ["file"],
            "path": ["/data/a.jpg"],
        }
    ).to_parquet(manifest, index=False)
    checkpoint = tmp_path / "base.pth"
    checkpoint.write_text("base", encoding="utf-8")
    base_spec = tmp_path / "base.yaml"
    base_spec.write_text(
        yaml.safe_dump(
            {
                "dataset": {"batch_size": 1},
                "train": {"num_nodes": 1, "num_gpus": 1},
            }
        ),
        encoding="utf-8",
    )
    output_dir = (tmp_path / "output").resolve()
    output_dir.mkdir()
    request_digest = train_cli._preparation_request_digest(
        base_spec=base_spec,
        manifest=manifest,
        parent_checkpoint=checkpoint,
        passes=1,
        output_dir=output_dir,
        num_nodes=1,
        gpus_per_node=1,
        checkpoint_policy="caller_selected",
    )
    train_cli._prepare_training(
        base_spec=base_spec,
        manifest=manifest,
        parent_checkpoint=checkpoint,
        passes=1,
        output_dir=output_dir,
        num_nodes=1,
        gpus_per_node=1,
        checkpoint_policy="caller_selected",
        request_digest=request_digest,
        launch_id=None,
    )
    (output_dir / "model_000_00001.pth").write_text(
        "same-round state", encoding="utf-8"
    )
    checkpoint.write_text("different base", encoding="utf-8")

    with pytest.raises(RuntimeError, match="input identity changed"):
        train_cli.main(
            [
                "--base-spec",
                str(base_spec),
                "--manifest",
                str(manifest),
                "--checkpoint",
                str(checkpoint),
                "--passes",
                "1",
                "--output-dir",
                str(output_dir),
                "--prepare-only",
            ]
        )


def test_training_contract_binds_same_round_resume_checkpoint(tmp_path: Path) -> None:
    manifest = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "sample_id": ["a"],
            "storage_type": ["file"],
            "path": ["/data/a.jpg"],
        }
    ).to_parquet(manifest, index=False)
    checkpoint = tmp_path / "base.pth"
    checkpoint.write_text("base", encoding="utf-8")
    base_spec = tmp_path / "base.yaml"
    base_spec.write_text(
        yaml.safe_dump(
            {
                "dataset": {"batch_size": 1},
                "train": {"num_nodes": 1, "num_gpus": 1},
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    resume = output_dir / "model_000_00001.pth"
    torch.save(_full_state(0, 1), resume)

    _, contract = build_training_spec(
        base_spec=base_spec,
        manifest=manifest,
        parent_checkpoint=checkpoint,
        passes=1,
        output_dir=output_dir,
    )

    assert contract["resume_checkpoint"] == str(resume.resolve())
    assert contract["resume_checkpoint_bytes"] == resume.stat().st_size
    assert contract["resume_checkpoint_sha256"] == "sha256:" + hashlib.sha256(
        resume.read_bytes()
    ).hexdigest()


def test_manifest_requires_canonical_locator(tmp_path: Path) -> None:
    manifest = tmp_path / "invalid.parquet"
    pd.DataFrame({"sample_id": ["a"]}).to_parquet(manifest, index=False)
    with pytest.raises(ValueError, match="storage_type/path/member"):
        DinoV3Dataset(root="/", manifest_path=manifest, transform=_transform)


@pytest.mark.parametrize("storage_type", ["tar", "zip"])
def test_dataset_reads_archive_member(tmp_path: Path, storage_type: str) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "green").save(image_path)
    member = "images/image.png"
    archive_path = tmp_path / f"images.{storage_type}"
    if storage_type == "tar":
        with tarfile.open(archive_path, "w") as archive:
            archive.add(image_path, arcname=member)
    else:
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.write(image_path, arcname=member)
    manifest = tmp_path / "archive.parquet"
    pd.DataFrame(
        {
            "sample_id": ["archive-image"],
            "storage_type": [storage_type],
            "path": [str(archive_path)],
            "member": [member],
        }
    ).to_parquet(manifest, index=False)
    dataset = DinoV3Dataset(
        root="/", manifest_path=manifest, transform=_transform, train=False
    )
    assert dataset[0]["input_path"] == f"{archive_path}::{member}"


def test_dataset_reuses_tar_index_and_accepts_file_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "green").save(image_path)
    archive_path = tmp_path / "images.tar"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(image_path, arcname="image.png")
    manifest = tmp_path / "archive.parquet"
    pd.DataFrame(
        {
            "sample_id": ["archive-image"],
            "storage_type": ["tar"],
            "path": [archive_path.as_uri()],
            "member": ["image.png"],
        }
    ).to_parquet(manifest, index=False)
    original_open = tarfile.open
    calls = []

    def counted_open(*args, **kwargs):
        calls.append(args[0])
        return original_open(*args, **kwargs)

    monkeypatch.setattr(tarfile, "open", counted_open)
    dataset = DinoV3Dataset(
        root="/", manifest_path=manifest, transform=_transform, train=True
    )
    dataset[0]
    dataset[0]
    assert calls == [archive_path]


def test_shard_aware_sampler_keeps_archive_rows_contiguous() -> None:
    class Dataset:
        groups = ["tar:a"] * 3 + ["tar:b"] * 3 + ["row:6"]

        def __len__(self):
            return len(self.groups)

        def sampling_group(self, index: int) -> str:
            return self.groups[index]

    dataset = Dataset()
    sampler = ShardAwareDistributedSampler(
        dataset, num_replicas=1, rank=0, shuffle=True, seed=7
    )
    groups = [dataset.sampling_group(index) for index in sampler]
    positions = {
        group: [index for index, value in enumerate(groups) if value == group]
        for group in set(groups)
    }
    for values in positions.values():
        assert values == list(range(min(values), max(values) + 1))


def test_shard_sampler_honors_disabled_shuffle() -> None:
    class Dataset:
        def __len__(self):
            return 20

        def sampling_group(self, index):
            return "archive"

    sampler = ShardAwareDistributedSampler(
        Dataset(), num_replicas=1, rank=0, shuffle=False
    )
    assert list(sampler) == list(range(20))
    sampler.set_epoch(1)
    assert list(sampler) == list(range(20))


def test_shard_sampler_accepts_empty_partition() -> None:
    class Dataset:
        def __len__(self):
            return 0

        def sampling_group(self, index):
            raise AssertionError("empty dataset must not be indexed")

    sampler = ShardAwareDistributedSampler(
        Dataset(), num_replicas=2, rank=0, shuffle=False
    )
    assert len(sampler) == 0
    assert list(sampler) == []


@pytest.mark.parametrize("storage_type", ["tar", "zip"])
def test_archive_descriptors_are_reopened_after_fork(tmp_path, monkeypatch, storage_type):
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "green").save(image)
    path = tmp_path / f"images.{storage_type}"
    if storage_type == "tar":
        with tarfile.open(path, "w") as archive:
            archive.add(image, arcname="image.png")
    else:
        with zipfile.ZipFile(path, "w") as archive:
            archive.write(image, arcname="image.png")
    dataset = DinoV3Dataset(root=tmp_path, transform=_transform)
    parent_handle = dataset._archive(storage_type, path)
    parent_pid = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: parent_pid + 1)
    child_handle = dataset._archive(storage_type, path)
    assert child_handle is not parent_handle
    assert dataset._archive(storage_type, path) is child_handle


def test_shard_aware_sampler_assigns_archives_to_bounded_rank_sets() -> None:
    class Dataset:
        groups = ["tar:a"] * 100 + ["tar:b"] * 100 + ["tar:c"] * 100

        def __len__(self):
            return len(self.groups)

        def sampling_group(self, index: int) -> str:
            return self.groups[index]

    dataset = Dataset()
    per_rank = []
    for rank in range(8):
        sampler = ShardAwareDistributedSampler(
            dataset,
            num_replicas=8,
            rank=rank,
            shuffle=True,
            seed=11,
        )
        indices = list(sampler)
        assert len(indices) == len(sampler)
        assert len(indices) == 38
        assert (len(indices) + 10 - 1) // 10 == 4
        per_rank.append(indices)
    assert set().union(*map(set, per_rank)) == set(range(len(dataset)))
    assert sum(map(len, per_rank)) == 304
    for group in set(dataset.groups):
        ranks = {
            rank
            for rank, indices in enumerate(per_rank)
            if any(dataset.sampling_group(index) == group for index in indices)
        }
        assert len(ranks) < 8


def test_dataset_rejects_compressed_random_access_tar(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "green").save(image_path)
    archive_path = tmp_path / "images.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(image_path, arcname="image.png")
    manifest = tmp_path / "archive.parquet"
    pd.DataFrame(
        {
            "sample_id": ["archive-image"],
            "storage_type": ["tar"],
            "path": [str(archive_path)],
            "member": ["image.png"],
        }
    ).to_parquet(manifest, index=False)
    dataset = DinoV3Dataset(
        root="/", manifest_path=manifest, transform=_transform, train=True
    )
    with pytest.raises(ValueError, match="uncompressed"):
        dataset[0]


def test_dataset_rejects_remote_file_uri_authority(tmp_path: Path) -> None:
    manifest = tmp_path / "remote.parquet"
    pd.DataFrame(
        {
            "sample_id": ["remote-image"],
            "storage_type": ["file"],
            "path": ["file://remote-host/data/image.png"],
        }
    ).to_parquet(manifest, index=False)
    dataset = DinoV3Dataset(
        root="/", manifest_path=manifest, transform=_transform, train=True
    )
    with pytest.raises(ValueError, match="Remote file URI"):
        dataset[0]


def test_resume_and_round_checkpoint_roles_are_not_ambiguous(tmp_path: Path) -> None:
    full_state = tmp_path / "model_epoch_001_step_00010.pth"
    teacher = tmp_path / "teacher_epoch_001_step_00010.pth"
    student = tmp_path / "student_ema_epoch_001_step_00010.pth"
    torch.save(_full_state(1, 10), full_state)
    for path in (teacher, student):
        path.write_text(path.name, encoding="utf-8")
    assert _latest_resume_checkpoint(tmp_path) == full_state.resolve()
    assert _final_teacher_checkpoint(tmp_path) == teacher


def test_resume_skips_newer_truncated_checkpoint(tmp_path: Path) -> None:
    valid = tmp_path / "model_epoch_001_step_00010.pth"
    truncated = tmp_path / "model_epoch_002_step_00020.pth"
    torch.save(_full_state(1, 10), valid)
    truncated.write_bytes(b"truncated")
    assert _latest_resume_checkpoint(tmp_path) == valid.resolve()


def test_resume_skips_newer_weights_only_checkpoint(tmp_path: Path) -> None:
    valid = tmp_path / "model_epoch_001_step_00010.pth"
    weights_only = tmp_path / "model_epoch_002_step_00020.pth"
    torch.save(_full_state(1, 10), valid)
    torch.save({"weight": torch.ones(1)}, weights_only)
    assert _latest_resume_checkpoint(tmp_path) == valid.resolve()


def test_finalize_publishes_final_teacher_as_hard_link(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher_epoch_011_step_00010.pth"
    torch.save({"weight": torch.ones(1)}, teacher)
    contract_path = tmp_path / "training_contract.json"
    runtime_spec = tmp_path / "experiment.yaml"
    runtime_spec.write_text("runtime: true\n", encoding="utf-8")
    contract_path.write_text(
        json.dumps(
            {
                "requested_data_passes": 12,
                "total_optimizer_steps": 10,
                "runtime_spec": str(runtime_spec),
            }
        ),
        encoding="utf-8",
    )
    prepared_digest = "sha256:" + hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest()
    checkpoint = finalize_training(tmp_path)
    assert torch.load(checkpoint, weights_only=True)["weight"].item() == 1.0
    assert checkpoint.stat().st_ino == teacher.stat().st_ino
    expected = "sha256:" + hashlib.sha256(teacher.read_bytes()).hexdigest()
    contract = json.loads(
        (tmp_path / "training_contract.json").read_text(encoding="utf-8")
    )
    assert contract["checkpoint"] == str(checkpoint)
    assert contract["prepared_contract_sha256"] == prepared_digest
    assert contract["checkpoint_bytes"] == teacher.stat().st_size
    assert contract["checkpoint_sha256"] == expected
    commit = tmp_path / "training_commit.json"
    assert commit.is_file()
    commit_sha256 = "sha256:" + hashlib.sha256(commit.read_bytes()).hexdigest()
    assert (tmp_path / "_SUCCESS").read_text(
        encoding="utf-8"
    ).strip() == commit_sha256
    assert contract["runtime_spec_sha256"] == "sha256:" + hashlib.sha256(
        runtime_spec.read_bytes()
    ).hexdigest()

    (tmp_path / "_SUCCESS").unlink()
    assert finalize_training(tmp_path) == checkpoint
    assert (tmp_path / "_SUCCESS").read_text(
        encoding="utf-8"
    ).strip() == commit_sha256

    commit.unlink()
    assert finalize_training(tmp_path) == checkpoint
    assert "sha256:" + hashlib.sha256(commit.read_bytes()).hexdigest() == (
        commit_sha256
    )

    contract["checkpoint_policy"] = "tampered"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not bind training_contract_sha256"):
        finalize_training(tmp_path)


def test_finalize_recovers_missing_teacher_from_final_full_state(
    tmp_path: Path,
) -> None:
    runtime_spec = tmp_path / "experiment.yaml"
    runtime_spec.write_text("runtime: true\n", encoding="utf-8")
    (tmp_path / "training_contract.json").write_text(
        json.dumps(
            {
                "requested_data_passes": 2,
                "total_optimizer_steps": 10,
                "runtime_spec": str(runtime_spec),
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        _full_state(1, 10),
        tmp_path / "model_epoch_001_step_00010.pth",
    )

    checkpoint = finalize_training(tmp_path)

    assert torch.load(checkpoint, weights_only=True)["weight"].tolist() == [1.0, 1.0]
    assert (tmp_path / "teacher_epoch_001_step_00010.pth").is_file()


def test_finalize_recovers_corrupt_final_teacher_from_full_state(
    tmp_path: Path,
) -> None:
    runtime_spec = tmp_path / "experiment.yaml"
    runtime_spec.write_text("runtime: true\n", encoding="utf-8")
    (tmp_path / "training_contract.json").write_text(
        json.dumps(
            {
                "requested_data_passes": 2,
                "total_optimizer_steps": 10,
                "runtime_spec": str(runtime_spec),
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "teacher_epoch_001_step_00010.pth").write_bytes(b"truncated")
    torch.save(
        _full_state(1, 10),
        tmp_path / "model_epoch_001_step_00010.pth",
    )

    checkpoint = finalize_training(tmp_path)

    assert torch.load(checkpoint, weights_only=True)["weight"].tolist() == [1.0, 1.0]


def test_finalize_rejects_teacher_before_requested_final_pass(tmp_path: Path) -> None:
    (tmp_path / "teacher_epoch_011_step_00010.pth").write_text(
        "teacher", encoding="utf-8"
    )
    (tmp_path / "training_contract.json").write_text(
        json.dumps({"requested_data_passes": 13, "total_optimizer_steps": 10}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="requested training extent"):
        finalize_training(tmp_path)


def test_finalize_rejects_teacher_before_requested_final_step(tmp_path: Path) -> None:
    (tmp_path / "teacher_epoch_011_step_00009.pth").write_text(
        "teacher", encoding="utf-8"
    )
    (tmp_path / "training_contract.json").write_text(
        json.dumps({"requested_data_passes": 12, "total_optimizer_steps": 10}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="requested training extent"):
        finalize_training(tmp_path)


def test_periodic_checkpoint_retention_keeps_newest_families(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TAO_SSL_CHECKPOINT_KEEP_LAST_N", "2")
    for step in (1, 2, 3):
        for prefix in ("model", "student_epoch", "teacher_epoch"):
            path = tmp_path / f"{prefix}_{step:03d}_step_{step:05d}.pth"
            path.write_text(str(step), encoding="utf-8")
    prune_periodic_ssl_checkpoints(str(tmp_path))
    for prefix in ("model", "student_epoch", "teacher_epoch"):
        retained = [
            path.read_text(encoding="utf-8")
            for path in sorted(tmp_path.glob(f"{prefix}_*.pth"))
        ]
        assert retained == ["2", "3"]
