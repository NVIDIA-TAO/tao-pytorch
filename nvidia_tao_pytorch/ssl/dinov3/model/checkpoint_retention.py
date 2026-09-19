# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disk-bounded retention for periodic SSL checkpoint families."""

from __future__ import annotations

from pathlib import Path
import re


_CHECKPOINT_PATTERNS = (
    re.compile(r"model_epoch_(\d+)_step_(\d+)\.pth"),
    re.compile(r"(?:student|teacher|student_ema)_epoch_(\d+)_step_(\d+)\.pth"),
)


def _checkpoint_extent(path: Path) -> tuple[int, int, str]:
    match = next(
        (pattern.fullmatch(path.name) for pattern in _CHECKPOINT_PATTERNS
         if pattern.fullmatch(path.name)),
        None,
    )
    if match is None:
        raise ValueError(f"Not a periodic DINOv3 checkpoint: {path.name}")
    return (int(match.group(1)), int(match.group(2)), path.name)


def prune_periodic_ssl_checkpoints(results_dir: str, keep_last_n: int) -> None:
    """Keep only the newest requested SSL checkpoint families when configured."""
    keep = int(keep_last_n)
    if keep == 0:
        return
    if keep < 0:
        raise ValueError("keep_last_n must be non-negative")
    root = Path(results_dir)
    patterns = (
        "model_epoch_*_step_*.pth",
        "student_epoch_*_step_*.pth",
        "teacher_epoch_*_step_*.pth",
        "student_ema_epoch_*_step_*.pth",
    )
    for pattern in patterns:
        candidates = sorted(
            (
                path for path in root.glob(pattern)
                if any(regex.fullmatch(path.name) for regex in _CHECKPOINT_PATTERNS)
            ),
            key=_checkpoint_extent,
        )
        for stale in candidates[:-keep]:
            stale.unlink()
