# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 rank-owned runtime-spec publication tests."""

from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("omegaconf")
from omegaconf import OmegaConf  # noqa: E402

from nvidia_tao_pytorch.ssl.dinov3.utils.runtime_spec import (  # noqa: E402
    publish_runtime_spec,
)


@pytest.mark.parametrize("rank_variable", ["RANK", "NODE_RANK", "LOCAL_RANK"])
def test_nonzero_rank_does_not_publish(
    rank_variable, tmp_path, monkeypatch
):
    for name in ("RANK", "NODE_RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(rank_variable, "1")

    publish_runtime_spec(OmegaConf.create({"value": "nonzero"}), str(tmp_path))

    assert not (tmp_path / "experiment.yaml").exists()


def test_empty_rank_exports_publish_as_rank_zero(tmp_path, monkeypatch):
    """Scheduler wrappers commonly export rank variables as empty strings."""
    for name in ("RANK", "NODE_RANK", "LOCAL_RANK"):
        monkeypatch.setenv(name, "")
    config = OmegaConf.create({"value": "rank-zero"})

    publish_runtime_spec(config, str(tmp_path))

    assert OmegaConf.load(tmp_path / "experiment.yaml") == config
    assert not list(tmp_path.glob(".experiment.yaml.*.tmp"))


def test_concurrent_rank_zero_writers_use_unique_atomic_files(
    tmp_path, monkeypatch
):
    """Retries sharing a results directory cannot unlink one another's temp file."""
    for name in ("RANK", "NODE_RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    config = OmegaConf.create({"value": "same-sealed-spec"})

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(publish_runtime_spec, config, str(tmp_path))
            for _ in range(8)
        ]
        for future in futures:
            future.result()

    assert OmegaConf.load(tmp_path / "experiment.yaml") == config
    assert not list(tmp_path.glob(".experiment.yaml.*.tmp"))
