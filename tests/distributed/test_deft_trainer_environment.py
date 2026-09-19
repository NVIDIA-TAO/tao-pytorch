# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The DEFT node launcher must pair with Lightning-owned worker creation."""

from nvidia_tao_pytorch.ssl.dinov3.scripts.train import _refinement_trainer_plugins


def test_deft_overrides_external_scheduler_environment(monkeypatch):
    """A Slurm allocation must not make Lightning expect pre-launched workers."""
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("TAO_REFINEMENT_LIGHTNING_LAUNCH", "1")
    plugins = _refinement_trainer_plugins()
    assert len(plugins) == 1
    assert plugins[0].creates_processes_externally is False


def test_ordinary_launch_preserves_lightning_autodetection(monkeypatch):
    """Only the explicit DEFT launch contract opts into this plugin."""
    monkeypatch.delenv("TAO_REFINEMENT_LIGHTNING_LAUNCH", raising=False)
    assert _refinement_trainer_plugins() is None
