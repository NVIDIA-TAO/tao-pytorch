# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for initialize_train_experiment determinism plumbing."""

import os

import pytest
import torch
import torch.backends.cudnn as cudnn

from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment


@pytest.fixture
def restore_determinism_state():
    """Snapshot and restore process-global determinism state across a test."""
    prior_benchmark = cudnn.benchmark
    prior_cudnn_det = cudnn.deterministic
    prior_use_det = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    prior_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        yield
    finally:
        cudnn.benchmark = prior_benchmark
        cudnn.deterministic = prior_cudnn_det
        torch.use_deterministic_algorithms(prior_use_det, warn_only=prior_warn_only)
        if prior_cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = prior_cublas


def _build_cfg(tmp_path, deterministic):
    """Minimal cfg dict accepted by initialize_train_experiment."""
    return {
        "results_dir": str(tmp_path),
        "train": {
            "num_epochs": 1,
            "validation_interval": 1,
            "checkpoint_interval": 1,
            "checkpoint_interval_unit": "epoch",
            "seed": 1234,
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
