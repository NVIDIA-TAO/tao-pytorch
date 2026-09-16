# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-owned runtime spec publication for DINOv3 training."""

import os
from pathlib import Path
import tempfile

from omegaconf import OmegaConf


def _nonzero_rank(name: str) -> bool:
    """Treat absent and empty scheduler rank exports as rank zero."""
    return int(os.environ.get(name) or "0") != 0


def publish_runtime_spec(cfg, results_dir: str) -> None:
    """Durably publish one DINOv3 runtime spec from global rank zero."""
    if any(_nonzero_rank(name) for name in ("RANK", "NODE_RANK", "LOCAL_RANK")):
        return

    destination = Path(results_dir) / "experiment.yaml"
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        OmegaConf.save(cfg, temporary_path)
        with temporary_path.open("rb") as temporary_file:
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)
