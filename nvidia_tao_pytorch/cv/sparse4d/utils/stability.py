# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical and output-format stability helpers for Sparse4D."""

from contextlib import contextmanager
import math
import threading
from typing import Iterable, Iterator

import torch


_NVSCHEMA_FPS_LOCK = threading.RLock()


@torch.no_grad()
def scrub_nan_gradients(parameters: Iterable[torch.nn.Parameter]) -> int:
    """Replace NaN gradient entries with zero while preserving infinities.

    Lightning calls :meth:`~pytorch_lightning.LightningModule.on_after_backward`
    before mixed-precision gradients are unscaled. Keeping ``+/-Inf`` intact lets
    the scaler retain its normal overflow detection and scale-reduction behavior.

    Args:
        parameters: Parameters whose gradients should be inspected.

    Returns:
        Number of gradient tensors inspected. This is intentionally host-sync
        free; it does not count individual NaN entries.
    """
    inspected = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = (
            parameter.grad._values() if parameter.grad.is_sparse else parameter.grad
        )
        torch.nan_to_num_(
            gradient,
            nan=0.0,
            posinf=float("inf"),
            neginf=float("-inf"),
        )
        inspected += 1
    return inspected


@contextmanager
def nvschema_fps_override(fps: float = 0.0) -> Iterator[None]:
    """Temporarily override the installed NVSchema converter frame rate.

    ``spatialai_data_utils`` currently exposes its frame rate only as a module
    constant. The conversion runs synchronously at inference/test epoch end, so
    a scoped override gives Sparse4D a public config knob without modifying that
    shared package. A value of zero preserves the package default.

    Args:
        fps: Positive output frame rate, or zero to use the package default.

    Raises:
        ValueError: If a non-zero value is not finite and positive.
    """
    fps = float(fps or 0.0)
    if not math.isfinite(fps) or fps < 0.0:
        raise ValueError(f"nvschema_fps must be finite and positive, got {fps}")

    # The converter exposes FPS as process-global state. Serialize scoped
    # overrides (including default-rate conversions) so overlapping callers
    # cannot observe or restore each other's values.
    with _NVSCHEMA_FPS_LOCK:
        if fps == 0.0:
            yield
            return

        from spatialai_data_utils.converters import nusc_results_to_nvschema

        original_fps = nusc_results_to_nvschema.FPS
        nusc_results_to_nvschema.FPS = fps
        try:
            yield
        finally:
            nusc_results_to_nvschema.FPS = original_fps
