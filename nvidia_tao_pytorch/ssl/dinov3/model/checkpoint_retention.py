# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disk-bounded retention for periodic SSL checkpoint families."""

from __future__ import annotations

import os
from pathlib import Path
import re


def _checkpoint_extent(path: Path) -> tuple[int, int, str]:
    match = re.search(r"(?:epoch_)?(\d+)(?:_step_)?(\d+)\.pth$", path.name)
    if match is None:
        return (-1, -1, path.name)
    return (int(match.group(1)), int(match.group(2)), path.name)


def prune_periodic_ssl_checkpoints(results_dir: str) -> None:
    """Keep only the newest requested SSL checkpoint families when configured."""
    raw_keep = os.environ.get("TAO_SSL_CHECKPOINT_KEEP_LAST_N")
    if raw_keep is None:
        return
    keep = int(raw_keep)
    if keep <= 0:
        raise ValueError("TAO_SSL_CHECKPOINT_KEEP_LAST_N must be positive")
    root = Path(results_dir)
    patterns = (
        "model_*.pth",
        "student_epoch_*_step_*.pth",
        "teacher_epoch_*_step_*.pth",
        "student_ema_epoch_*_step_*.pth",
    )
    for pattern in patterns:
        candidates = sorted(
            root.glob(pattern),
            key=_checkpoint_extent,
        )
        for stale in candidates[:-keep]:
            stale.unlink()
