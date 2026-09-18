# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the TAO-native GRIT subtask."""

import json

import pandas as pd
import pytest
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.dinov3.entrypoint.dinov3 import get_subtask_list
from nvidia_tao_pytorch.ssl.dinov3.scripts.grit_score import main


@pytest.mark.parametrize("torchrun", [False, True])
def test_tao_grit_subtask_publishes_scores_and_status(tmp_path, monkeypatch, torchrun):
    for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK", "NODE_RANK"):
        monkeypatch.delenv(name, raising=False)
    if torchrun:
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("NODE_RANK", "0")
    manifest = tmp_path / "consensus.parquet"
    pd.DataFrame({
        "sample_id": ["a", "b"], "task": ["domain", "domain"],
        "global_consensus": [0.2, 0.8], "dense_consensus": [0.3, 0.9],
    }).to_parquet(manifest, index=False)
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.results_dir = str(tmp_path / "results")
    cfg.grit_score.input_parquet = str(manifest)
    cfg.grit_score.precomputed_consensus = True
    parquet_reads = []
    real_read_parquet = pd.read_parquet

    def observed_read_parquet(*args, **kwargs):
        parquet_reads.append(kwargs.copy())
        return real_read_parquet(*args, **kwargs)

    monkeypatch.setattr(
        "nvidia_tao_pytorch.ssl.dinov3.data_refinement.cli.pd.read_parquet",
        observed_read_parquet,
    )
    assert "grit_score" in get_subtask_list()
    main(cfg)
    output = tmp_path / "results" / "grit_score"
    assert len(real_read_parquet(output / "grit_scores.parquet")) == 2
    assert parquet_reads
    assert all(options.get("pre_buffer") is False for options in parquet_reads)
    assert (output / "experiment.yaml").is_file()
    assert (output / "status.json").is_file()
    metadata = json.loads((output / "score_commit.json").read_text())
    assert metadata["observation_mode"] == "precomputed_consensus"
    snapshot = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    with pytest.raises(RuntimeError, match="committed output"):
        main(cfg)
    repeated = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert repeated == snapshot


@pytest.mark.parametrize("environment", [
    {"WORLD_SIZE": "2", "RANK": "0"},
    {"WORLD_SIZE": "2", "RANK": "1"},
    {"LOCAL_WORLD_SIZE": "2"},
    {"RANK": "1"},
    {"LOCAL_RANK": "1"},
    {"NODE_RANK": "1"},
])
def test_tao_grit_rejects_distributed_workers_before_publication(
    tmp_path, monkeypatch, environment
):
    """Neither rank zero nor other workers may modify shared score artifacts."""
    for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK", "NODE_RANK"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.results_dir = str(tmp_path / "results")
    with pytest.raises(ValueError, match="single process"):
        main(cfg)
    assert not (tmp_path / "results").exists()
