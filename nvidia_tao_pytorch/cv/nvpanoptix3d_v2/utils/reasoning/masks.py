# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mask utilities shared by the NVPanoptix3Dv2-Reasoning training path."""

from torch import Tensor


def build_target_masks(
    pan_inst_id: Tensor,
    target_inst_id: Tensor,
) -> Tensor:
    """Build ``[B, S, H, W]`` binary masks from per-view instance ids."""
    tgt = target_inst_id.view(-1, 1, 1, 1)
    return (pan_inst_id == tgt) & (tgt > 0)


def gather_view_masks(
    pan_inst_id: Tensor,
    target_inst_id: Tensor,
    sample_idx: Tensor,
    view_idx: Tensor,
) -> Tensor:
    """Per-binding GT instance masks.

    Args:
        pan_inst_id: ``[B, S, H, W]`` per-view instance ids.
        target_inst_id: ``[B]`` instance UID per sample.
        sample_idx: ``[N]`` sample id per binding.
        view_idx: ``[N]`` view id per binding.

    Returns:
        ``[N, H, W]`` boolean GT masks (``pan_inst_id == target`` at each binding's view).
    """
    gt_all = build_target_masks(pan_inst_id, target_inst_id)  # [B, S, H, W]
    if sample_idx.numel() == 0:
        _, _, H, W = gt_all.shape
        return gt_all.new_zeros((0, H, W), dtype=gt_all.dtype)
    return gt_all[sample_idx, view_idx]
