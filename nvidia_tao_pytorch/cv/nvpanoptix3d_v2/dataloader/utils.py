# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities shared by both NVPanoptix3Dv2 dataloader variants."""

from typing import List, Tuple

import numpy as np
from PIL import Image


def get_config_value(config, key, default=None):
    """Read a key from a dictionary or an attribute-based config object."""
    if config is None:
        return default
    if hasattr(config, key):
        return getattr(config, key)
    if isinstance(config, dict):
        return config.get(key, default)
    return default


def center_crop_and_resize(
    img: Image.Image,
    intrinsics: np.ndarray,
    depth: np.ndarray,
    panoptic_id,
    target_hw: Tuple[int, int],
):
    """Principal-point crop, aspect-preserving resize, and center padding.

    The transform preserves spatial correspondence among RGB, depth, panoptic
    labels, and camera intrinsics. ``intrinsics`` is modified in place.
    """
    target_h, target_w = target_hw
    image_w, image_h = img.size

    center_x = int(round(intrinsics[0, 2]))
    center_y = int(round(intrinsics[1, 2]))
    margin_x = min(center_x, image_w - center_x)
    margin_y = min(center_y, image_h - center_y)

    left, top = center_x - margin_x, center_y - margin_y
    right, bottom = center_x + margin_x, center_y + margin_y

    img = img.crop((left, top, right, bottom))
    depth = depth[top:bottom, left:right]
    if panoptic_id is not None:
        panoptic_id = panoptic_id[top:bottom, left:right]
    intrinsics[0, 2] -= left
    intrinsics[1, 2] -= top

    crop_w, crop_h = img.size
    scale = min(target_w / crop_w, target_h / crop_h)
    new_w = max(1, min(target_w, round(crop_w * scale)))
    new_h = max(1, min(target_h, round(crop_h * scale)))

    img = img.resize((new_w, new_h), Image.LANCZOS)
    # Nearest-neighbor interpolation keeps invalid depth at zero and preserves
    # real values at depth discontinuities.
    depth = np.array(
        Image.fromarray(depth).resize((new_w, new_h), Image.NEAREST),
        dtype=np.float32,
    )
    if panoptic_id is not None:
        panoptic_id = np.array(
            Image.fromarray(panoptic_id.astype(np.int32)).resize(
                (new_w, new_h), Image.NEAREST,
            ),
            dtype=np.int32,
        )
    intrinsics[0, :] *= new_w / crop_w
    intrinsics[1, :] *= new_h / crop_h

    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2

    image_canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    image_canvas.paste(img, (pad_left, pad_top))
    img = image_canvas

    depth_canvas = np.zeros((target_h, target_w), dtype=np.float32)
    depth_canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = depth
    depth = depth_canvas

    if panoptic_id is not None:
        panoptic_canvas = np.zeros((target_h, target_w), dtype=np.int32)
        panoptic_canvas[
            pad_top:pad_top + new_h, pad_left:pad_left + new_w
        ] = panoptic_id
        panoptic_id = panoptic_canvas

    intrinsics[0, 2] += pad_left
    intrinsics[1, 2] += pad_top
    return img, intrinsics, depth, panoptic_id


def rgb2id(rgb: np.ndarray) -> np.ndarray:
    """Decode an RGB-encoded ScanNet++ panoptic ID map."""
    return (
        rgb[..., 0].astype(np.int32) +
        256 * rgb[..., 1].astype(np.int32) +
        65536 * rgb[..., 2].astype(np.int32)
    )


def normalize_resolution_arg(resolution) -> List[Tuple[int, int]]:
    """Normalize one ``(H, W)`` pair or a list of resolution pairs."""
    if resolution is None:
        raise ValueError("resolution must not be None")
    first = resolution[0]
    if isinstance(first, (list, tuple)):
        buckets = [(int(value[0]), int(value[1])) for value in resolution]
    else:
        buckets = [(int(resolution[0]), int(resolution[1]))]
    if not buckets:
        raise ValueError("resolution list is empty")
    return buckets
