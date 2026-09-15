# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-owned runtime spec publication for DINOv3 training."""

import os
from pathlib import Path

from omegaconf import OmegaConf


def publish_runtime_spec(cfg, results_dir: str) -> None:
    """Atomically publish one DINOv3 runtime spec from global rank zero."""
    rank_coordinates = ("RANK", "NODE_RANK", "LOCAL_RANK")
    if any(int(os.getenv(name, "0")) != 0 for name in rank_coordinates):
        return
    destination = Path(results_dir) / "experiment.yaml"
    temporary_path = destination.with_name(".experiment.yaml.tmp")
    try:
        OmegaConf.save(cfg, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
