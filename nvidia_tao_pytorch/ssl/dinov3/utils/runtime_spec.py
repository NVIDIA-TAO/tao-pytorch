# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-owned runtime spec publication for DINOv3 training."""

import os
from pathlib import Path

from omegaconf import OmegaConf

from nvidia_tao_pytorch.ssl.dinov3.utils.atomic import atomic_path


def _nonzero_rank(name: str) -> bool:
    """Treat absent and empty scheduler rank exports as rank zero."""
    return int(os.environ.get(name) or "0") != 0


def publish_runtime_spec(cfg, results_dir: str) -> None:
    """Durably publish one DINOv3 runtime spec from global rank zero."""
    if any(_nonzero_rank(name) for name in ("RANK", "NODE_RANK", "LOCAL_RANK")):
        return

    destination = Path(results_dir) / "experiment.yaml"
    with atomic_path(destination) as temporary_path:
        OmegaConf.save(cfg, temporary_path)
