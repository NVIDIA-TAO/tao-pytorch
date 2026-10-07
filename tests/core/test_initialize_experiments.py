# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for initialize_train_experiment determinism plumbing."""

import os
import random

import numpy as np
import pytest
import torch
import torch.backends.cudnn as cudnn

import nvidia_tao_pytorch.core.initialize_experiments as initialize_module
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment


@pytest.fixture
def restore_determinism_state():
    """Snapshot and restore process-global determinism state across a test."""
    prior_benchmark = cudnn.benchmark
    prior_cudnn_det = cudnn.deterministic
    prior_use_det = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    prior_sdp = (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
    )
    prior_python_rng = random.getstate()
    prior_numpy_rng = np.random.get_state()
    prior_torch_rng = torch.random.get_rng_state()
    prior_cuda_rng = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    )
    tracked_environment = (
        "CUBLAS_WORKSPACE_CONFIG",
        "PL_GLOBAL_SEED",
        "PL_SEED_WORKERS",
    )
    prior_environment = {
        name: os.environ.get(name) for name in tracked_environment
    }
    try:
        yield
    finally:
        cudnn.benchmark = prior_benchmark
        cudnn.deterministic = prior_cudnn_det
        torch.use_deterministic_algorithms(prior_use_det, warn_only=prior_warn_only)
        torch.backends.cuda.enable_flash_sdp(prior_sdp[0])
        torch.backends.cuda.enable_mem_efficient_sdp(prior_sdp[1])
        torch.backends.cuda.enable_math_sdp(prior_sdp[2])
        random.setstate(prior_python_rng)
        np.random.set_state(prior_numpy_rng)
        torch.random.set_rng_state(prior_torch_rng)
        if prior_cuda_rng is not None:
            torch.cuda.set_rng_state_all(prior_cuda_rng)
        for name, value in prior_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _build_cfg(tmp_path, deterministic):
    """Minimal cfg dict accepted by initialize_train_experiment."""
    return {
        "results_dir": str(tmp_path),
        "train": {
            "num_epochs": 1,
            "validation_interval": 1,
            "checkpoint_interval": 1,
            "checkpoint_interval_unit": "epoch",
            # These tests exercise determinism flags, not RNG seeding. Disabling
            # seeding also avoids leaving a deferred CUDA seed callback when the
            # process has not initialized CUDA yet.
            "seed": -1,
            "cudnn": {"benchmark": False, "deterministic": deterministic},
            "resume_training_checkpoint_path": None,
            "num_gpus": 1,
            "gpu_ids": [0],
        },
    }


def test_deterministic_true_wires_global_flags(tmp_path, monkeypatch, restore_determinism_state):
    """When cudnn.deterministic=True the helper must enable every determinism switch.

    Regression guard for TLT-5860: prior to the fix, only cuDNN conv determinism
    was set; torch.use_deterministic_algorithms / CUBLAS_WORKSPACE_CONFIG /
    Trainer(deterministic=...) were all left untouched.
    """
    monkeypatch.setenv("TAO_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    _, trainer_kwargs = initialize_train_experiment(_build_cfg(tmp_path, deterministic=True))

    assert cudnn.deterministic is True
    assert cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.is_deterministic_algorithms_warn_only_enabled() is True
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert trainer_kwargs["deterministic"] == "warn"


def test_deterministic_false_leaves_global_flags_off(tmp_path, monkeypatch, restore_determinism_state):
    """deterministic=False must not export CUBLAS_WORKSPACE_CONFIG, call
    use_deterministic_algorithms, or pass a truthy value to Trainer."""
    monkeypatch.setenv("TAO_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    torch.use_deterministic_algorithms(False)

    _, trainer_kwargs = initialize_train_experiment(_build_cfg(tmp_path, deterministic=False))

    assert cudnn.deterministic is False
    assert trainer_kwargs["deterministic"] is False
    assert torch.are_deterministic_algorithms_enabled() is False
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_cublas_workspace_config_respects_user_value(tmp_path, monkeypatch, restore_determinism_state):
    """setdefault semantics: a user-exported CUBLAS_WORKSPACE_CONFIG wins."""
    monkeypatch.setenv("TAO_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    initialize_train_experiment(_build_cfg(tmp_path, deterministic=True))

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_default_rejects_epoch_checkpoint_beyond_training(tmp_path, monkeypatch):
    """Existing model callers retain the historical cadence validation."""
    config = _build_cfg(tmp_path, deterministic=False)
    config["train"]["checkpoint_interval"] = 2
    monkeypatch.setenv("TAO_VISIBLE_DEVICES", "0")
    with pytest.raises(AssertionError, match="Checkpoint interval"):
        initialize_train_experiment(config)


def test_dino_opt_ins_allow_terminal_save_without_directory_resume(
    tmp_path, monkeypatch
):
    """DEFT candidates can finish off cadence and never inherit sibling state."""
    config = _build_cfg(tmp_path, deterministic=False)
    config["train"]["checkpoint_interval"] = 2
    monkeypatch.setenv("TAO_VISIBLE_DEVICES", "0")

    def unexpected_scan(_results_dir):
        raise AssertionError("automatic checkpoint scanning must be disabled")

    monkeypatch.setattr(initialize_module, "get_latest_checkpoint", unexpected_scan)
    resume, _ = initialize_train_experiment(
        config,
        allow_off_cadence_final_checkpoint=True,
        auto_resume=False,
    )
    assert resume is None


@pytest.mark.parametrize("resume_path", [None, "", "explicit.pth"])
@pytest.mark.parametrize("auto_resume", [False, True])
@pytest.mark.parametrize("discovered", [None, "discovered.pth"])
def test_resume_path_normalization(
    tmp_path, monkeypatch, restore_determinism_state,
    resume_path, auto_resume, discovered,
):
    """Unset paths mean fresh training unless directory discovery is enabled."""
    config = _build_cfg(tmp_path, deterministic=False)
    config["train"]["resume_training_checkpoint_path"] = resume_path
    monkeypatch.setenv("TAO_VISIBLE_DEVICES", "0")
    scans = []

    def discover(results_dir):
        scans.append(results_dir)
        return discovered

    monkeypatch.setattr(initialize_module, "get_latest_checkpoint", discover)
    resume, _ = initialize_train_experiment(config, auto_resume=auto_resume)
    should_scan = auto_resume and not resume_path
    assert scans == ([str(tmp_path)] if should_scan else [])
    assert resume == (resume_path or (discovered if should_scan else None))


@pytest.mark.parametrize("resume_path", [None, ""])
def test_disabled_resume_reaches_lightning_training_step(
    tmp_path, monkeypatch, restore_determinism_state, resume_path,
):
    """Exercise Lightning's real checkpoint parser, not just the helper return."""
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset

    class TinyModel(pl.LightningModule):
        """Small CPU model for the fresh-start checkpoint boundary."""

        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 1)

        def training_step(self, batch, batch_idx):
            """Perform one dummy-data training step."""
            return self.layer(batch[0]).square().mean()

        def configure_optimizers(self):
            """Use a minimal optimizer with no external assets."""
            return torch.optim.SGD(self.parameters(), lr=0.01)

    config = _build_cfg(tmp_path, deterministic=False)
    config["train"]["resume_training_checkpoint_path"] = resume_path
    monkeypatch.setenv("TAO_VISIBLE_DEVICES", "0")
    resume, _ = initialize_train_experiment(config, auto_resume=False)
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_steps=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, default_root_dir=str(tmp_path),
    )
    trainer.fit(
        TinyModel(), DataLoader(TensorDataset(torch.ones(4, 2)), batch_size=2),
        ckpt_path=resume,
    )
    assert trainer.global_step == 1
