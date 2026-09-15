# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for workflow status artifact publication."""

from pathlib import Path

import pytest

pytest.importorskip("omegaconf")
from omegaconf import OmegaConf  # noqa: E402

from nvidia_tao_pytorch.ssl.dinov3.utils.runtime_spec import publish_runtime_spec
from nvidia_tao_pytorch.core.decorators import workflow


@pytest.mark.parametrize("rank_variable", ["RANK", "NODE_RANK", "LOCAL_RANK"])
def test_runtime_spec_is_published_only_by_rank_zero(
    rank_variable: str,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = OmegaConf.create({"train": {"num_nodes": 2}})
    destination = tmp_path / "experiment.yaml"

    for name in ("RANK", "NODE_RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(rank_variable, "1")
    publish_runtime_spec(config, str(tmp_path))
    assert not destination.exists()

    monkeypatch.setenv(rank_variable, "0")
    publish_runtime_spec(config, str(tmp_path))
    assert OmegaConf.load(destination) == config
    assert not list(tmp_path.glob(".experiment.*.tmp"))


def test_default_status_writer_is_unchanged_on_nonzero_rank(tmp_path, monkeypatch):
    """Existing model callers keep their original runtime-spec writing path."""
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setattr(workflow, "update_results_dir", lambda cfg, **_: cfg)
    config = OmegaConf.create({"results_dir": str(tmp_path)})

    @workflow.monitor_status(name="NVDINOv2", mode="train")
    def run(_cfg):
        return None

    run(config)
    assert OmegaConf.load(tmp_path / "experiment.yaml") == config


def test_status_writer_opt_in_uses_dinov3_rank_policy(tmp_path, monkeypatch):
    """The DINOv3 callback alone suppresses nonzero-rank publication."""
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setattr(workflow, "update_results_dir", lambda cfg, **_: cfg)
    config = OmegaConf.create({"results_dir": str(tmp_path)})

    @workflow.monitor_status(name="DINOv3", mode="train", spec_writer=publish_runtime_spec)
    def run(_cfg):
        return None

    run(config)
    assert not (tmp_path / "experiment.yaml").exists()
